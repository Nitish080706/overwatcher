"""
OVERWATCHER Pipeline
=====================
Orchestrates all 8 modules in sequence for every incoming agent action request.
This is the single entry point — call pipeline.run(request) to process an action.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from overwatcher.models import (
    OverwatcherRequest,
    OverwatcherResponse,
)
from overwatcher.modules.intent_parser.parser import IntentParser
from overwatcher.modules.identity.verifier import IdentityVerifier
from overwatcher.modules.risk_scoring.engine import RiskScoringEngine
from overwatcher.modules.consistency.checker import ConsistencyChecker
from overwatcher.modules.trust_memory.memory import TrustMemory
from overwatcher.modules.behavior_analyzer.analyzer import BehaviorAnalyzer
from overwatcher.modules.policy_engine.engine import PolicyEngine
from overwatcher.modules.audit_logger.logger import AuditLogger


class OverwatcherPipeline:
    """
    Wires together all 8 OVERWATCHER modules in order.
    Each module is instantiated once and reused across requests.
    """

    def __init__(self) -> None:
        self._intent_parser   = IntentParser()
        self._identity        = IdentityVerifier()
        self._risk_engine     = RiskScoringEngine()
        self._consistency     = ConsistencyChecker()
        self._trust_memory    = TrustMemory()
        self._behavior        = BehaviorAnalyzer()
        self._policy          = PolicyEngine()
        self._audit           = AuditLogger()

        # Train behavior analyzer on startup (replace with loaded model in prod)
        self._behavior.load_or_train()

    def run(self, request: OverwatcherRequest) -> OverwatcherResponse:
        """
        Process one agent action request through the full pipeline.

        Pipeline order:
          1. Intent Parser
          2. Identity & Device Verification
          3. Risk Scoring Engine
          4. Consistency Checker
          5. Behavioral Trust Memory (read)
          6. Behavior Analyzer
          7. Dynamic Policy Engine
          8. Audit Logger
          9. Trust Memory update (post-decision)

        Returns:
            OverwatcherResponse with the final decision and all signals.
        """
        action = request.proposed_action

        # ── Module 1: Parse intent ────────────────────────────────────────
        parsed_intent = self._intent_parser.parse(request.raw_request)

        # ── Module 2: Verify identity & device ───────────────────────────
        identity_ctx = self._identity.verify(request.auth_token, request.device_id)

        # ── Module 3: Score risk ──────────────────────────────────────────
        risk_score = self._risk_engine.score(action, urgency=parsed_intent.urgency)

        # ── Module 4: Check consistency ───────────────────────────────────
        consistency = self._consistency.check(parsed_intent, action)

        # ── Module 5: Get trust profile ───────────────────────────────────
        trust_profile = self._trust_memory.get_trust_profile(action.agent_id)

        # ── Module 6: Analyze behavior sequence ───────────────────────────
        anomaly = self._behavior.analyze(
            session_id=request.session_id,
            operation=action.operation,
            resource_type=action.resource_type,
        )

        # ── Module 7: Policy decision ─────────────────────────────────────
        policy_result = self._policy.decide(
            identity=identity_ctx,
            risk=risk_score,
            consistency=consistency,
            trust=trust_profile,
            anomaly=anomaly,
            operation=action.operation,
            resource=action.resource_type,
        )

        # ── Build response ────────────────────────────────────────────────
        response = OverwatcherResponse(
            session_id=request.session_id,
            agent_id=action.agent_id,
            decision=policy_result.decision,
            policy_result=policy_result,
            parsed_intent=parsed_intent,
            consistency_result=consistency,
            risk_score=risk_score,
            trust_profile=trust_profile,
            anomaly_result=anomaly,
            audit_entry_id="",  # filled below
            timestamp=datetime.now(timezone.utc),
        )

        # ── Module 8: Audit log ───────────────────────────────────────────
        audit_id = self._audit.log(
            response=response,
            clinician_id=identity_ctx.user_id,
            raw_request=request.raw_request,
        )
        response.audit_entry_id = audit_id

        # ── Module 5: Update trust based on decision ──────────────────────
        from overwatcher.models import PolicyDecision
        if policy_result.decision == PolicyDecision.ALLOW:
            self._trust_memory.reward(action.agent_id)
        elif policy_result.decision == PolicyDecision.BLOCK:
            if not consistency.is_consistent:
                self._trust_memory.increment_violation(action.agent_id)
            self._trust_memory.penalize(action.agent_id, multiplier=2.0)
        elif policy_result.decision == PolicyDecision.HUMAN_APPROVAL:
            # Trust update deferred until clinician approves/rejects
            pass

        return response
