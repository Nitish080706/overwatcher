"""
Module 1: Intent Parser
=======================
Converts a clinician's free-text request into a structured ParsedIntent schema.

Approach (No LLM):
  - spaCy NER pipeline for entity extraction (patient names, record types, staff)
  - spaCy Matcher / PhraseMatcher for verb-action mapping
  - Dependency parser for verb-object-recipient resolution
  - Deterministic rule engine to map extracted entities → ParsedIntent fields
  - HMAC signing of the final ParsedIntent to produce a tamper-proof "golden intent"

Low-confidence parses (confidence < threshold) are flagged immediately
and routed to Step-up Verification by the Policy Engine.
"""

import hashlib
import hmac
import json
from typing import Optional

import spacy
from spacy.matcher import Matcher

from overwatcher.config import get_settings
from overwatcher.models import (
    OperationType,
    ParsedIntent,
    ResourceType,
    UrgencyLevel,
)

settings = get_settings()

# ---------------------------------------------------------------------------
# Action verb → OperationType mapping
# ---------------------------------------------------------------------------
VERB_TO_OPERATION: dict[str, OperationType] = {
    "fetch":    OperationType.READ,
    "get":      OperationType.READ,
    "retrieve": OperationType.READ,
    "read":     OperationType.READ,
    "show":     OperationType.READ,
    "send":     OperationType.SEND,
    "share":    OperationType.SEND,
    "forward":  OperationType.SEND,
    "message":  OperationType.SEND,
    "update":   OperationType.UPDATE,
    "modify":   OperationType.UPDATE,
    "change":   OperationType.UPDATE,
    "edit":     OperationType.UPDATE,
    "delete":   OperationType.DELETE,
    "remove":   OperationType.DELETE,
    "cancel":   OperationType.CANCEL,
    "create":   OperationType.CREATE,
    "add":      OperationType.CREATE,
    "write":    OperationType.WRITE,
    "order":    OperationType.CREATE,
}

# ---------------------------------------------------------------------------
# Noun phrase → ResourceType mapping
# ---------------------------------------------------------------------------
NOUN_TO_RESOURCE: dict[str, ResourceType] = {
    "blood report":      ResourceType.LAB_REPORT,
    "lab report":        ResourceType.LAB_REPORT,
    "test result":       ResourceType.LAB_REPORT,
    "cbc":               ResourceType.LAB_REPORT,
    "complete blood count": ResourceType.LAB_REPORT,
    "prescription":      ResourceType.PRESCRIPTION,
    "medication":        ResourceType.PRESCRIPTION,
    "dosage":            ResourceType.PRESCRIPTION,
    "drug":              ResourceType.PRESCRIPTION,
    "mri":               ResourceType.MRI,
    "mri scan":          ResourceType.MRI,
    "ct scan":           ResourceType.MRI,
    "x-ray":             ResourceType.MRI,
    "patient record":    ResourceType.PATIENT_RECORD,
    "medical record":    ResourceType.PATIENT_RECORD,
    "chart":             ResourceType.PATIENT_RECORD,
    "ehr":               ResourceType.PATIENT_RECORD,
    "message":           ResourceType.MESSAGE,
    "appointment":       ResourceType.APPOINTMENT,
    "schedule":          ResourceType.APPOINTMENT,
    "billing":           ResourceType.BILLING,
    "invoice":           ResourceType.BILLING,
    "device":            ResourceType.DEVICE_COMMAND,
    "monitor":           ResourceType.DEVICE_COMMAND,
    "ventilator":        ResourceType.DEVICE_COMMAND,
    "infusion pump":     ResourceType.DEVICE_COMMAND,
}

# ---------------------------------------------------------------------------
# Urgency keywords
# ---------------------------------------------------------------------------
URGENCY_MAP: dict[str, UrgencyLevel] = {
    "emergency":  UrgencyLevel.EMERGENCY,
    "critical":   UrgencyLevel.EMERGENCY,
    "code blue":  UrgencyLevel.EMERGENCY,
    "stat":       UrgencyLevel.URGENT,
    "urgent":     UrgencyLevel.URGENT,
    "asap":       UrgencyLevel.URGENT,
    "immediately": UrgencyLevel.URGENT,
    "routine":    UrgencyLevel.ROUTINE,
}


class IntentParser:
    """
    Parses a clinician's natural-language request into a structured ParsedIntent.
    Uses spaCy NER + rule-based Matcher. No LLM involved.
    """

    def __init__(self) -> None:
        # Load the spaCy transformer model (or medium model for lighter deployments)
        # Use en_core_web_trf for best accuracy, en_core_web_md as fallback
        try:
            self._nlp = spacy.load("en_core_web_trf")
        except OSError:
            self._nlp = spacy.load("en_core_web_md")

        self._matcher = Matcher(self._nlp.vocab)
        self._setup_matcher_patterns()

    def _setup_matcher_patterns(self) -> None:
        """Register spaCy Matcher patterns for action verb detection."""
        for verb in VERB_TO_OPERATION:
            pattern = [{"LOWER": verb}]
            self._matcher.add(f"VERB_{verb.upper()}", [pattern])

    def parse(self, raw_request: str) -> ParsedIntent:
        """
        Main entry point. Returns a ParsedIntent with a confidence score.
        Low confidence → caller should route to Step-up Verification.
        """
        doc = self._nlp(raw_request.lower().strip())

        operation     = self._extract_operation(doc)
        resource_type = self._extract_resource(doc)
        patient_id    = self._extract_patient(doc)
        recipients    = self._extract_recipients(doc)
        urgency       = self._extract_urgency(doc)
        data_fields   = self._extract_data_fields(doc)

        # Confidence: drops if any key field is missing
        confidence = self._compute_confidence(
            operation, resource_type, patient_id
        )

        intent = ParsedIntent(
            operation=operation or OperationType.READ,
            resource_type=resource_type or ResourceType.PATIENT_RECORD,
            patient_id=patient_id,
            recipients=recipients,
            data_fields_requested=data_fields,
            urgency=urgency,
            confidence=confidence,
            raw_request=raw_request,
        )

        # Sign the intent so downstream modules can verify it hasn't been tampered
        intent.signature = self._sign_intent(intent)
        return intent

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_operation(self, doc) -> Optional[OperationType]:
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            token_text = doc[start].text.lower()
            if token_text in VERB_TO_OPERATION:
                return VERB_TO_OPERATION[token_text]
        return None

    def _extract_resource(self, doc) -> Optional[ResourceType]:
        text = doc.text.lower()
        # Sort by length (longest match first) for greedy matching
        for phrase in sorted(NOUN_TO_RESOURCE.keys(), key=len, reverse=True):
            if phrase in text:
                return NOUN_TO_RESOURCE[phrase]
        return None

    def _extract_patient(self, doc) -> Optional[str]:
        """
        Extract patient identifier from NER (PERSON entities).
        In a real deployment this would resolve against the hospital patient DB.
        """
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.title()
        return None

    def _extract_recipients(self, doc) -> list[str]:
        """
        Extract recipient names/roles. Looks for prepositions like 'to', 'for'
        followed by PERSON entities.
        """
        recipients = []
        for i, token in enumerate(doc):
            if token.lower_ in ("to", "for") and i + 1 < len(doc):
                # Check the span following 'to/for' for PERSON ents
                for ent in doc.ents:
                    if ent.start > i and ent.label_ == "PERSON":
                        recipients.append(ent.text.title())
                        break
        return list(set(recipients))

    def _extract_urgency(self, doc) -> UrgencyLevel:
        text = doc.text.lower()
        for keyword, level in URGENCY_MAP.items():
            if keyword in text:
                return level
        return UrgencyLevel.ROUTINE

    def _extract_data_fields(self, doc) -> list[str]:
        """
        Extract specific data fields mentioned (e.g., 'hemoglobin', 'dosage').
        Returns noun chunks that are not already captured as resource_type.
        """
        fields = []
        for chunk in doc.noun_chunks:
            text = chunk.text.lower()
            if text not in NOUN_TO_RESOURCE:
                fields.append(text)
        return fields[:5]  # cap at 5 fields

    def _compute_confidence(
        self,
        operation: Optional[OperationType],
        resource_type: Optional[ResourceType],
        patient_id: Optional[str],
    ) -> float:
        score = 1.0
        if operation is None:
            score -= 0.4
        if resource_type is None:
            score -= 0.35
        if patient_id is None:
            score -= 0.15
        return max(0.0, score)

    def _sign_intent(self, intent: ParsedIntent) -> str:
        """
        HMAC-SHA256 sign the intent payload using the app secret key.
        This signature is verified by the Consistency Checker to ensure
        the golden intent was not tampered with by a prompt injection.
        """
        payload = json.dumps({
            "operation":   intent.operation.value,
            "resource_type": intent.resource_type.value,
            "patient_id":  intent.patient_id,
            "recipients":  sorted(intent.recipients),
            "data_fields": sorted(intent.data_fields_requested),
            "urgency":     intent.urgency.value,
        }, sort_keys=True)

        sig = hmac.new(
            settings.secret_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return sig
