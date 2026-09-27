"""
Module 7: Dynamic Policy Engine
=================================
Fuses all module signals into a single, deterministic routing decision.
No ML override — pure rule engine with configurable thresholds.

Decision priority (evaluated in order):
  1. BLOCK   — identity not verified
  2. BLOCK   — IAM permission denied for this operation/resource
  3. BLOCK   — consistency violation (high confidence)
  4. BLOCK   — anomaly score above block threshold
  5. HUMAN   — high risk AND low trust
  6. HUMAN   — high risk score alone
  7. HUMAN   — high anomaly score
  8. STEP-UP — moderate risk OR moderate anomaly
  9. ALLOW   — all checks pass

All thresholds are configurable via .env — no hardcoding.
Output: PolicyResult with decision + rule_fired + all contributing signals.
"""

from overwatcher.config import get_settings
from overwatcher.models import (
    AnomalyResult,
    ConsistencyResult,
    IdentityContext,
    OperationType,
    PolicyDecision,
    PolicyResult,
    ResourceType,
    RiskScore,
    TrustProfile,
)

settings = get_settings()


class PolicyEngine:
    """
    Deterministic rule engine — no LLM, no ML at inference time.
    Thresholds are loaded from settings (configurable per hospital deployment).
    """

    def decide(
        self,
        identity:    IdentityContext,
        risk:        RiskScore,
        consistency: ConsistencyResult,
        trust:       TrustProfile,
        anomaly:     AnomalyResult,
        operation:   OperationType,
        resource:    ResourceType,
    ) -> PolicyResult:
        """
        Evaluate all signals and return a PolicyResult.
        Rules are evaluated in strict priority order.
        """
        reasons: list[str] = []

        # ----------------------------------------------------------------
        # Rule 1: Identity not verified → BLOCK immediately
        # ----------------------------------------------------------------
        if not identity.identity_verified:
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                rule_fired="R1_IDENTITY_NOT_VERIFIED",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=["Identity verification failed — token invalid or device untrusted"],
            )

        # ----------------------------------------------------------------
        # Rule 2: IAM permission denied → BLOCK
        # ----------------------------------------------------------------
        if (
            operation not in identity.allowed_operations
            or resource not in identity.allowed_resource_types
        ):
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                rule_fired="R2_PERMISSION_DENIED",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=[
                    f"Role '{identity.role.value}' does not have permission "
                    f"for {operation.value} on {resource.value}"
                ],
            )

        # ----------------------------------------------------------------
        # Rule 3: Consistency violation (high confidence) → BLOCK
        # ----------------------------------------------------------------
        if (
            not consistency.is_consistent
            and consistency.confidence >= settings.threshold_block_consistency
        ):
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                rule_fired="R3_CONSISTENCY_VIOLATION",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=False,
                reasons=consistency.violations,
            )

        # ----------------------------------------------------------------
        # Rule 4: Anomaly score above block threshold → BLOCK
        # ----------------------------------------------------------------
        if anomaly.anomaly_score >= settings.threshold_anomaly_block:
            reasons.append(f"Sequence anomaly score {anomaly.anomaly_score:.2f} exceeds block threshold")
            return PolicyResult(
                decision=PolicyDecision.BLOCK,
                rule_fired="R4_ANOMALY_BLOCK",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=reasons,
            )

        # ----------------------------------------------------------------
        # Rule 5: High risk AND low trust → Human Approval
        # ----------------------------------------------------------------
        if (
            risk.composite_score >= settings.threshold_human_approval_risk
            and trust.trust_score < settings.threshold_human_approval_trust
        ):
            reasons.append(
                f"High risk ({risk.composite_score:.2f}) combined with "
                f"low trust ({trust.trust_score:.2f})"
            )
            return PolicyResult(
                decision=PolicyDecision.HUMAN_APPROVAL,
                rule_fired="R5_HIGH_RISK_LOW_TRUST",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=reasons,
            )

        # ----------------------------------------------------------------
        # Rule 6: High risk score alone → Human Approval
        # ----------------------------------------------------------------
        if risk.composite_score >= settings.threshold_human_approval_risk:
            reasons.append(f"Risk score {risk.composite_score:.2f} exceeds human approval threshold")
            return PolicyResult(
                decision=PolicyDecision.HUMAN_APPROVAL,
                rule_fired="R6_HIGH_RISK",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=reasons,
            )

        # ----------------------------------------------------------------
        # Rule 7: High anomaly score → Human Approval
        # ----------------------------------------------------------------
        if anomaly.anomaly_score >= settings.threshold_anomaly_human:
            reasons.append(f"Sequence anomaly score {anomaly.anomaly_score:.2f} requires human review")
            return PolicyResult(
                decision=PolicyDecision.HUMAN_APPROVAL,
                rule_fired="R7_ANOMALY_HUMAN",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=reasons,
            )

        # ----------------------------------------------------------------
        # Rule 8: Moderate risk OR moderate anomaly → Step-up Verification
        # ----------------------------------------------------------------
        if (
            risk.composite_score >= settings.threshold_stepup_risk
            or anomaly.anomaly_score >= settings.threshold_stepup_anomaly
            or not consistency.is_consistent  # low-confidence inconsistency
        ):
            if risk.composite_score >= settings.threshold_stepup_risk:
                reasons.append(f"Moderate risk score {risk.composite_score:.2f}")
            if anomaly.anomaly_score >= settings.threshold_stepup_anomaly:
                reasons.append(f"Moderate anomaly score {anomaly.anomaly_score:.2f}")
            if not consistency.is_consistent:
                reasons.append("Low-confidence consistency check")
            return PolicyResult(
                decision=PolicyDecision.STEP_UP,
                rule_fired="R8_STEPUP",
                risk_score=risk.composite_score,
                trust_score=trust.trust_score,
                anomaly_score=anomaly.anomaly_score,
                is_consistent=consistency.is_consistent,
                reasons=reasons,
            )

        # ----------------------------------------------------------------
        # Rule 9: All checks pass → Allow
        # ----------------------------------------------------------------
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            rule_fired="R9_ALLOW",
            risk_score=risk.composite_score,
            trust_score=trust.trust_score,
            anomaly_score=anomaly.anomaly_score,
            is_consistent=consistency.is_consistent,
            reasons=["All checks passed"],
        )
