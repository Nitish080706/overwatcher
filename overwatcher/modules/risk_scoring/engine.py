"""
Module 3: Risk Scoring Engine
==============================
Scores the agent's proposed action along three axes:
  1. Reversibility  — how hard is it to undo this action?
  2. Sensitivity    — how sensitive is the data being accessed/modified?
  3. Urgency        — does urgency context change the risk profile?

Approach (No LLM):
  - Expert-defined Clinical Action Taxonomy (static lookup table)
  - Fixed weights set by clinical domain experts (configurable via .env)
  - Pure arithmetic — fully deterministic, fully auditable

Output: RiskScore with composite_score in [0, 1].
"""

from overwatcher.config import get_settings
from overwatcher.models import (
    OperationType,
    ProposedAction,
    ResourceType,
    RiskScore,
    UrgencyLevel,
)

settings = get_settings()

# ---------------------------------------------------------------------------
# Clinical Action Taxonomy
# (operation, resource_type) → (reversibility, sensitivity)
#
# reversibility: 0.0 = fully reversible (read), 1.0 = irreversible (delete)
# sensitivity:   0.0 = public data,       1.0 = highly sensitive PHI
# ---------------------------------------------------------------------------
ACTION_TAXONOMY: dict[tuple[OperationType, ResourceType], tuple[float, float]] = {
    # READ operations — generally low risk
    (OperationType.READ, ResourceType.LAB_REPORT):        (0.0, 0.5),
    (OperationType.READ, ResourceType.MRI):               (0.0, 0.5),
    (OperationType.READ, ResourceType.PRESCRIPTION):      (0.0, 0.6),
    (OperationType.READ, ResourceType.PATIENT_RECORD):    (0.0, 0.7),
    (OperationType.READ, ResourceType.MESSAGE):           (0.0, 0.3),
    (OperationType.READ, ResourceType.APPOINTMENT):       (0.0, 0.1),
    (OperationType.READ, ResourceType.BILLING):           (0.0, 0.4),
    (OperationType.READ, ResourceType.DEVICE_COMMAND):    (0.0, 0.3),

    # SEND operations — moderate risk (PHI disclosure risk)
    (OperationType.SEND, ResourceType.LAB_REPORT):        (0.3, 0.6),
    (OperationType.SEND, ResourceType.MRI):               (0.3, 0.6),
    (OperationType.SEND, ResourceType.PRESCRIPTION):      (0.4, 0.7),
    (OperationType.SEND, ResourceType.PATIENT_RECORD):    (0.4, 0.9),
    (OperationType.SEND, ResourceType.MESSAGE):           (0.2, 0.3),

    # UPDATE operations — high risk (data mutation)
    (OperationType.UPDATE, ResourceType.PRESCRIPTION):    (0.9, 0.9),
    (OperationType.UPDATE, ResourceType.PATIENT_RECORD):  (0.8, 0.8),
    (OperationType.UPDATE, ResourceType.LAB_REPORT):      (0.7, 0.7),
    (OperationType.UPDATE, ResourceType.APPOINTMENT):     (0.5, 0.2),
    (OperationType.UPDATE, ResourceType.DEVICE_COMMAND):  (0.8, 0.7),

    # CREATE operations
    (OperationType.CREATE, ResourceType.PRESCRIPTION):    (0.7, 0.8),
    (OperationType.CREATE, ResourceType.APPOINTMENT):     (0.4, 0.1),
    (OperationType.CREATE, ResourceType.LAB_REPORT):      (0.5, 0.5),

    # WRITE operations
    (OperationType.WRITE, ResourceType.PATIENT_RECORD):   (0.8, 0.9),
    (OperationType.WRITE, ResourceType.LAB_REPORT):       (0.7, 0.7),

    # DELETE operations — highest risk
    (OperationType.DELETE, ResourceType.PATIENT_RECORD):  (1.0, 1.0),
    (OperationType.DELETE, ResourceType.PRESCRIPTION):    (1.0, 0.9),
    (OperationType.DELETE, ResourceType.LAB_REPORT):      (1.0, 0.8),
    (OperationType.DELETE, ResourceType.APPOINTMENT):     (0.6, 0.2),

    # CANCEL operations
    (OperationType.CANCEL, ResourceType.APPOINTMENT):     (0.5, 0.1),
    (OperationType.CANCEL, ResourceType.PRESCRIPTION):    (0.8, 0.8),

    # DEVICE COMMAND — very high risk (physical patient safety)
    (OperationType.UPDATE, ResourceType.DEVICE_COMMAND):  (0.9, 0.8),
    (OperationType.CREATE, ResourceType.DEVICE_COMMAND):  (0.8, 0.8),
}

# Default fallback for unknown (operation, resource) pairs — conservative
_DEFAULT_SCORES = (0.6, 0.6)

# Urgency modifier — emergency context can increase effective risk
URGENCY_MODIFIER: dict[UrgencyLevel, float] = {
    UrgencyLevel.ROUTINE:   0.1,
    UrgencyLevel.URGENT:    0.3,
    UrgencyLevel.EMERGENCY: 0.5,
}


class RiskScoringEngine:
    """
    Computes a composite risk score for a proposed action.
    Fully deterministic — no ML, no LLM.
    """

    def __init__(self) -> None:
        self._w_rev = settings.risk_weight_reversibility
        self._w_sen = settings.risk_weight_sensitivity
        self._w_urg = settings.risk_weight_urgency

    def score(
        self,
        action: ProposedAction,
        urgency: UrgencyLevel = UrgencyLevel.ROUTINE,
    ) -> RiskScore:
        """
        Score a proposed action.

        Args:
            action:  The agent's proposed tool call.
            urgency: Urgency level from the parsed intent.

        Returns:
            RiskScore with all component scores and the composite.
        """
        key = (action.operation, action.resource_type)
        reversibility, sensitivity = ACTION_TAXONOMY.get(key, _DEFAULT_SCORES)
        urgency_mod = URGENCY_MODIFIER.get(urgency, 0.1)

        composite = (
            self._w_rev * reversibility
            + self._w_sen * sensitivity
            + self._w_urg * urgency_mod
        )
        # Clamp to [0, 1]
        composite = min(1.0, max(0.0, composite))

        return RiskScore(
            reversibility=reversibility,
            sensitivity=sensitivity,
            urgency_modifier=urgency_mod,
            composite_score=round(composite, 4),
        )
