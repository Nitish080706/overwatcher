"""
Tests for the OVERWATCHER Intent Parser.
"""

import pytest
from overwatcher.modules.intent_parser.parser import IntentParser
from overwatcher.models import OperationType, ResourceType, UrgencyLevel


@pytest.fixture
def parser():
    return IntentParser()


def test_parse_fetch_blood_report(parser):
    result = parser.parse("Fetch John's blood report and send it to Nurse Sarah")
    assert result.operation in (OperationType.READ, OperationType.SEND)
    assert result.resource_type == ResourceType.LAB_REPORT
    assert result.confidence > 0.0
    assert result.signature is not None


def test_parse_update_prescription(parser):
    result = parser.parse("Update the prescription dosage for patient Mary")
    assert result.operation == OperationType.UPDATE
    assert result.resource_type == ResourceType.PRESCRIPTION


def test_parse_urgency_emergency(parser):
    result = parser.parse("Emergency: retrieve patient vitals immediately")
    assert result.urgency == UrgencyLevel.EMERGENCY


def test_parse_low_confidence_flags(parser):
    # Ambiguous request — should have low confidence
    result = parser.parse("do something with the thing")
    assert result.confidence < 0.6


def test_intent_is_signed(parser):
    result = parser.parse("Get lab results for patient 123")
    assert result.signature is not None
    assert len(result.signature) == 64  # SHA-256 hex digest
