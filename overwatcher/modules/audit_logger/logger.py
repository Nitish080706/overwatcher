"""
Module 8: Audit Logger
========================
Writes every OVERWATCHER decision to an immutable, hash-chained audit log.

Design:
  - Append-only PostgreSQL table — no UPDATE or DELETE operations permitted
  - SHA-256 hash chaining: each entry includes the hash of the previous entry,
    making retroactive modification detectable
  - Structured log format (structlog) for machine-readable compliance output
  - Every decision, its contributing factors, and the policy rule that fired
    are recorded — supports HIPAA audit trail requirements

Output: audit_entry_id (UUID) for each logged decision.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from overwatcher.config import get_settings
from overwatcher.models import OverwatcherResponse

settings = get_settings()
logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class AuditLogEntry(Base):
    """
    Immutable audit log entry.
    This table must NEVER have UPDATE or DELETE permissions granted to
    the application database user in production.
    """
    __tablename__ = "audit_logs"

    id              = Column(String(36), primary_key=True)  # UUID
    session_id      = Column(String(128), index=True)
    agent_id        = Column(String(128), index=True)
    clinician_id    = Column(String(128))
    timestamp       = Column(DateTime(timezone=True))
    decision        = Column(String(32))
    rule_fired      = Column(String(64))
    risk_score      = Column(String(8))
    trust_score     = Column(String(8))
    anomaly_score   = Column(String(8))
    is_consistent   = Column(String(5))
    violations      = Column(Text)   # JSON list
    reasons         = Column(Text)   # JSON list
    raw_request     = Column(Text)
    proposed_action = Column(Text)   # JSON
    entry_hash      = Column(String(64))
    previous_hash   = Column(String(64))


# ---------------------------------------------------------------------------
# Audit Logger Service
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Append-only, hash-chained audit logger.
    No LLM, no ML — pure deterministic logging.
    """

    def __init__(self) -> None:
        engine = create_engine(settings.database_url, echo=False)
        Base.metadata.create_all(engine)
        self._session_factory = sessionmaker(bind=engine)
        self._last_hash: str = "GENESIS"  # starting hash for the chain

    def log(self, response: OverwatcherResponse, clinician_id: str, raw_request: str) -> str:
        """
        Write one audit entry. Returns the entry ID.

        Args:
            response:     The full OverwatcherResponse from the pipeline.
            clinician_id: ID of the requesting clinician.
            raw_request:  Original free-text request.

        Returns:
            UUID string of the created audit entry.
        """
        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        payload = {
            "id":             entry_id,
            "session_id":     response.session_id,
            "agent_id":       response.agent_id,
            "clinician_id":   clinician_id,
            "timestamp":      now.isoformat(),
            "decision":       response.decision.value,
            "rule_fired":     response.policy_result.rule_fired,
            "risk_score":     str(round(response.risk_score.composite_score, 4)),
            "trust_score":    str(round(response.trust_profile.trust_score, 4)),
            "anomaly_score":  str(round(response.anomaly_result.anomaly_score, 4)),
            "is_consistent":  str(response.consistency_result.is_consistent),
            "violations":     json.dumps(response.consistency_result.violations),
            "reasons":        json.dumps(response.policy_result.reasons),
            "raw_request":    raw_request,
            "previous_hash":  self._last_hash,
        }

        entry_hash = self._compute_hash(payload)
        payload["entry_hash"] = entry_hash

        # Write to DB
        with self._session_factory() as session:
            entry = AuditLogEntry(**payload)
            session.add(entry)
            session.commit()

        # Update chain
        self._last_hash = entry_hash

        # Structured log output (machine-readable)
        logger.info(
            "overwatcher_decision",
            entry_id=entry_id,
            agent_id=response.agent_id,
            decision=response.decision.value,
            rule_fired=response.policy_result.rule_fired,
            risk_score=response.risk_score.composite_score,
            trust_score=response.trust_profile.trust_score,
            anomaly_score=response.anomaly_result.anomaly_score,
            is_consistent=response.consistency_result.is_consistent,
        )

        return entry_id

    @staticmethod
    def _compute_hash(payload: dict) -> str:
        """SHA-256 hash of the payload for chain integrity."""
        serialized = json.dumps(
            {k: v for k, v in payload.items() if k != "entry_hash"},
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()
