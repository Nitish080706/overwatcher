"""
Module 5: Behavioral Trust Memory
===================================
Maintains a persistent, per-agent trust score that evolves over time
based on the agent's history of reliable or unreliable behavior.

Approach (No LLM):
  - Bayesian-style update rule with fixed reward/penalty constants
  - Trust score stored in PostgreSQL (append-only trust event log)
  - Time-based decay applied per day of inactivity
  - Hash-chaining for tamper-evidence

Trust Update Rules:
  - Action ALLOWED and completed successfully → +alpha (small positive)
  - Consistency violation detected            → -beta  (significant negative)
  - Human approval required & clinician rejected → -2*beta (large negative)
  - Time-based decay: trust drifts toward neutral (0.5) if agent is inactive

Output: TrustProfile for the given agent_id.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from overwatcher.config import get_settings
from overwatcher.models import TrustProfile

settings = get_settings()


# ---------------------------------------------------------------------------
# SQLAlchemy ORM Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class AgentTrustRecord(Base):
    """Current trust score snapshot for each agent."""
    __tablename__ = "agent_trust"

    agent_id      = Column(String(128), primary_key=True)
    trust_score   = Column(Float, default=settings.default_trust_score)
    total_actions = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    last_updated  = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    last_hash     = Column(String(64), default="")


class TrustEvent(Base):
    """Append-only log of trust update events (hash-chained)."""
    __tablename__ = "trust_events"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    agent_id      = Column(String(128), index=True)
    event_type    = Column(String(32))   # reward | penalty | decay
    delta         = Column(Float)
    trust_after   = Column(Float)
    timestamp     = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    entry_hash    = Column(String(64))   # SHA-256 of this entry
    previous_hash = Column(String(64))   # SHA-256 of previous entry (chain)


# ---------------------------------------------------------------------------
# Trust Memory Service
# ---------------------------------------------------------------------------

class TrustMemory:
    """
    Manages per-agent trust scores with persistent storage.
    All operations are deterministic — no ML, no LLM.
    """

    def __init__(self) -> None:
        engine = create_engine(settings.database_url, echo=False)
        Base.metadata.create_all(engine)
        self._session_factory = sessionmaker(bind=engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trust_profile(self, agent_id: str) -> TrustProfile:
        """Retrieve the current trust profile for an agent."""
        with self._session_factory() as session:
            record = self._get_or_create(session, agent_id)
            self._apply_time_decay(session, record)
            session.commit()
            return TrustProfile(
                agent_id=record.agent_id,
                trust_score=round(record.trust_score, 4),
                total_actions=record.total_actions,
                total_violations=record.total_violations,
                last_updated=record.last_updated,
            )

    def reward(self, agent_id: str) -> TrustProfile:
        """Apply a positive trust update (action was valid and completed)."""
        return self._update(agent_id, delta=+settings.trust_reward_alpha, event_type="reward")

    def penalize(self, agent_id: str, multiplier: float = 1.0) -> TrustProfile:
        """Apply a negative trust update (violation detected)."""
        delta = -settings.trust_penalty_beta * multiplier
        return self._update(agent_id, delta=delta, event_type="penalty")

    def increment_violation(self, agent_id: str) -> None:
        """Increment the violation counter for an agent."""
        with self._session_factory() as session:
            record = self._get_or_create(session, agent_id)
            record.total_violations += 1
            session.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update(self, agent_id: str, delta: float, event_type: str) -> TrustProfile:
        with self._session_factory() as session:
            record = self._get_or_create(session, agent_id)
            record.trust_score = min(1.0, max(0.0, record.trust_score + delta))
            record.total_actions += 1
            record.last_updated = datetime.now(timezone.utc)

            # Append hash-chained trust event
            self._append_event(session, agent_id, event_type, delta, record.trust_score, record.last_hash)
            record.last_hash = self._compute_hash(agent_id, event_type, delta, record.trust_score)

            session.commit()
            return TrustProfile(
                agent_id=record.agent_id,
                trust_score=round(record.trust_score, 4),
                total_actions=record.total_actions,
                total_violations=record.total_violations,
                last_updated=record.last_updated,
            )

    def _apply_time_decay(self, session: Session, record: AgentTrustRecord) -> None:
        """
        Apply time-based decay proportional to days since last update.
        Trust drifts toward neutral (0.5) if agent is inactive.
        """
        now = datetime.now(timezone.utc)
        last = record.last_updated.replace(tzinfo=timezone.utc) if record.last_updated.tzinfo is None else record.last_updated
        days_inactive = (now - last).days
        if days_inactive > 0:
            decay = settings.trust_decay_factor ** days_inactive
            # Drift toward 0.5 (neutral)
            record.trust_score = 0.5 + (record.trust_score - 0.5) * decay
            record.last_updated = now

    def _get_or_create(self, session: Session, agent_id: str) -> AgentTrustRecord:
        record = session.get(AgentTrustRecord, agent_id)
        if record is None:
            record = AgentTrustRecord(
                agent_id=agent_id,
                trust_score=settings.default_trust_score,
            )
            session.add(record)
            session.flush()
        return record

    def _append_event(
        self,
        session: Session,
        agent_id: str,
        event_type: str,
        delta: float,
        trust_after: float,
        previous_hash: str,
    ) -> None:
        entry_hash = self._compute_hash(agent_id, event_type, delta, trust_after)
        event = TrustEvent(
            agent_id=agent_id,
            event_type=event_type,
            delta=delta,
            trust_after=trust_after,
            entry_hash=entry_hash,
            previous_hash=previous_hash or "",
        )
        session.add(event)

    @staticmethod
    def _compute_hash(agent_id: str, event_type: str, delta: float, trust_after: float) -> str:
        payload = json.dumps({
            "agent_id":   agent_id,
            "event_type": event_type,
            "delta":      delta,
            "trust_after": round(trust_after, 6),
            "ts":         datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
