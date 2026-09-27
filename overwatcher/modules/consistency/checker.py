"""
Module 4: Intent-Action Consistency Checker
============================================
Compares the agent's ACTUAL proposed action against the SIGNED golden intent
produced by the Intent Parser to detect scope expansion or prompt injection.

Approach (No LLM):
  1. Structural checks (deterministic):
     - operation must match
     - resource_type must match
     - recipients must be a subset of declared recipients
     - patient_id must match if specified
  2. Fuzzy field matching (static encoder — no generation):
     - Uses sentence-transformers (ClinicalBERT/BioBERT encoder only)
     - Cosine similarity between intent data_fields and proposed data_fields
     - Threshold configurable in .env

The HMAC signature on the ParsedIntent is verified before any comparison
to ensure the golden intent itself wasn't tampered with.

Output: ConsistencyResult with is_consistent flag and list of violations.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from overwatcher.config import get_settings
from overwatcher.models import (
    ConsistencyResult,
    ParsedIntent,
    ProposedAction,
)

settings = get_settings()


class ConsistencyChecker:
    """
    Checks that the agent's proposed action is consistent with the
    signed golden intent. This is the primary prompt injection defense.
    """

    def __init__(self) -> None:
        # Load encoder-only clinical embedding model (no generation)
        self._encoder = SentenceTransformer(settings.consistency_embedding_model)
        self._threshold = settings.consistency_similarity_threshold

    def check(
        self,
        intent: ParsedIntent,
        action: ProposedAction,
    ) -> ConsistencyResult:
        """
        Main entry point.

        Args:
            intent: The signed ParsedIntent from the Intent Parser.
            action: The proposed action intercepted at the OS/API level.

        Returns:
            ConsistencyResult with violations list and overall flag.
        """
        violations: list[str] = []

        # Step 0: Verify the HMAC signature on the golden intent
        if not self._verify_signature(intent):
            violations.append("CRITICAL: Golden intent signature is invalid — possible tampering")
            return ConsistencyResult(
                is_consistent=False,
                violations=violations,
                similarity_score=0.0,
                confidence=1.0,
            )

        # Step 1: Structural checks (fully deterministic)
        self._check_operation(intent, action, violations)
        self._check_resource_type(intent, action, violations)
        self._check_patient_id(intent, action, violations)
        self._check_recipients(intent, action, violations)

        # Step 2: Fuzzy data field check (static encoder, no generation)
        similarity_score = self._check_data_fields(intent, action, violations)

        is_consistent = len(violations) == 0

        return ConsistencyResult(
            is_consistent=is_consistent,
            violations=violations,
            similarity_score=round(max(0.0, min(1.0, similarity_score)), 4),
            confidence=0.95 if is_consistent else 1.0,
        )

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def _check_operation(
        self,
        intent: ParsedIntent,
        action: ProposedAction,
        violations: list[str],
    ) -> None:
        if action.operation != intent.operation:
            violations.append(
                f"Operation mismatch: intent={intent.operation.value}, "
                f"action={action.operation.value}"
            )

    def _check_resource_type(
        self,
        intent: ParsedIntent,
        action: ProposedAction,
        violations: list[str],
    ) -> None:
        if action.resource_type != intent.resource_type:
            violations.append(
                f"Resource type mismatch: intent={intent.resource_type.value}, "
                f"action={action.resource_type.value}"
            )

    def _check_patient_id(
        self,
        intent: ParsedIntent,
        action: ProposedAction,
        violations: list[str],
    ) -> None:
        if intent.patient_id and action.patient_id:
            if intent.patient_id.lower() != action.patient_id.lower():
                violations.append(
                    f"Patient ID mismatch: intent={intent.patient_id}, "
                    f"action={action.patient_id}"
                )

    def _check_recipients(
        self,
        intent: ParsedIntent,
        action: ProposedAction,
        violations: list[str],
    ) -> None:
        """Action recipients must be a subset of intent recipients."""
        intent_set = {r.lower() for r in intent.recipients}
        action_set = {r.lower() for r in action.recipients}
        extra = action_set - intent_set
        if extra:
            violations.append(
                f"Undeclared recipients in action: {extra}. "
                f"Only {intent_set} were declared."
            )

    # ------------------------------------------------------------------
    # Fuzzy field check (encoder only — no generation)
    # ------------------------------------------------------------------

    def _check_data_fields(
        self,
        intent: ParsedIntent,
        action: ProposedAction,
        violations: list[str],
    ) -> float:
        """
        Encode intent data fields and proposed data fields with a clinical
        sentence encoder. If any proposed field is semantically distant from
        all intent fields, it's flagged as a violation.

        Returns the minimum similarity score found (1.0 if no fields to check).
        """
        if not intent.data_fields_requested or not action.data_fields:
            return 1.0

        intent_embeddings = self._encoder.encode(intent.data_fields_requested)
        action_embeddings = self._encoder.encode(action.data_fields)

        sims = cosine_similarity(action_embeddings, intent_embeddings)
        # For each proposed field, check if max similarity to any intent field is above threshold
        min_score = 1.0
        for i, field in enumerate(action.data_fields):
            max_sim = float(np.max(sims[i]))
            min_score = min(min_score, max_sim)
            if max_sim < self._threshold:
                violations.append(
                    f"Data field '{field}' not semantically covered by intent "
                    f"(similarity={max_sim:.2f} < threshold={self._threshold})"
                )

        return min_score

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def _verify_signature(self, intent: ParsedIntent) -> bool:
        """
        Recompute the HMAC over the intent payload and compare with the
        stored signature. Returns False if signature is missing or invalid.
        """
        if not intent.signature:
            return False

        payload = json.dumps({
            "operation":   intent.operation.value,
            "resource_type": intent.resource_type.value,
            "patient_id":  intent.patient_id,
            "recipients":  sorted(intent.recipients),
            "data_fields": sorted(intent.data_fields_requested),
            "urgency":     intent.urgency.value,
        }, sort_keys=True)

        expected = hmac.new(
            settings.secret_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, intent.signature)
