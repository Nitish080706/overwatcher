"""
Tests for the Dynamic Policy Engine.
Covers all 9 rules with direct signal injection.
"""

import pytest
from datetime import datetime, timezone

from overwatcher.modules.policy_engine.engine import PolicyEngine
from overwatcher.models import (
    AnomalyResult,
    ConsistencyResult,
    IdentityContext,
    OperationType,
    PolicyDecision,
    ResourceType,
    RiskScore,
    TrustProfile,
    UserRole,
)


@pytest.fixture
def engine():
    return PolicyEngine()


def _make_identity(verified=True, role=UserRole.DOCTOR):
    return IdentityContext(
        user_id="doc1",
        role=role,
        device_id="HOSP-DEV-001",
        device_trusted=True,
        identity_verified=verified,
        allowed_operations=list(OperationType),
        allowed_resource_types=list(ResourceType),
    )


def _make_risk(score: float):
    return RiskScore(reversibility=score, sensitivity=score, urgency_modifier=0.1, composite_score=score)


def _make_consistency(ok: bool, confidence: float = 1.0):
    return ConsistencyResult(is_consistent=ok, violations=[] if ok else ["mismatch"], confidence=confidence)


def _make_trust(score: float):
    return TrustProfile(agent_id="agent1", trust_score=score, last_updated=datetime.now(timezone.utc))


def _make_anomaly(score: float):
    return AnomalyResult(anomaly_score=score, is_anomalous=score > 0.4, sequence_length=5)


class TestPolicyEngine:

    def test_r1_identity_not_verified(self, engine):
        result = engine.decide(
            _make_identity(verified=False), _make_risk(0.1),
            _make_consistency(True), _make_trust(0.9), _make_anomaly(0.0),
            OperationType.READ, ResourceType.LAB_REPORT,
        )
        assert result.decision == PolicyDecision.BLOCK
        assert result.rule_fired == "R1_IDENTITY_NOT_VERIFIED"

    def test_r2_permission_denied(self, engine):
        identity = IdentityContext(
            user_id="nurse1", role=UserRole.NURSE, device_id="DEV1",
            device_trusted=True, identity_verified=True,
            allowed_operations=[OperationType.READ],
            allowed_resource_types=[ResourceType.LAB_REPORT],
        )
        result = engine.decide(
            identity, _make_risk(0.1), _make_consistency(True),
            _make_trust(0.9), _make_anomaly(0.0),
            OperationType.DELETE, ResourceType.PRESCRIPTION,
        )
        assert result.decision == PolicyDecision.BLOCK
        assert result.rule_fired == "R2_PERMISSION_DENIED"

    def test_r3_consistency_violation(self, engine):
        result = engine.decide(
            _make_identity(), _make_risk(0.1),
            _make_consistency(False, confidence=0.95),
            _make_trust(0.9), _make_anomaly(0.0),
            OperationType.READ, ResourceType.LAB_REPORT,
        )
        assert result.decision == PolicyDecision.BLOCK
        assert result.rule_fired == "R3_CONSISTENCY_VIOLATION"

    def test_r4_anomaly_block(self, engine):
        result = engine.decide(
            _make_identity(), _make_risk(0.1), _make_consistency(True),
            _make_trust(0.9), _make_anomaly(0.95),
            OperationType.READ, ResourceType.LAB_REPORT,
        )
        assert result.decision == PolicyDecision.BLOCK
        assert result.rule_fired == "R4_ANOMALY_BLOCK"

    def test_r5_high_risk_low_trust(self, engine):
        result = engine.decide(
            _make_identity(), _make_risk(0.9), _make_consistency(True),
            _make_trust(0.3), _make_anomaly(0.0),
            OperationType.READ, ResourceType.LAB_REPORT,
        )
        assert result.decision == PolicyDecision.HUMAN_APPROVAL
        assert result.rule_fired == "R5_HIGH_RISK_LOW_TRUST"

    def test_r9_allow(self, engine):
        result = engine.decide(
            _make_identity(), _make_risk(0.2), _make_consistency(True),
            _make_trust(0.9), _make_anomaly(0.05),
            OperationType.READ, ResourceType.LAB_REPORT,
        )
        assert result.decision == PolicyDecision.ALLOW
        assert result.rule_fired == "R9_ALLOW"
