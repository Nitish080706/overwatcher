"""
Tests for the Consistency Checker — especially prompt injection scenarios.
"""

import pytest
from overwatcher.modules.consistency.checker import ConsistencyChecker
from overwatcher.modules.intent_parser.parser import IntentParser
from overwatcher.models import (
    OperationType,
    ProposedAction,
    ResourceType,
)


@pytest.fixture
def checker():
    return ConsistencyChecker()


@pytest.fixture
def parser():
    return IntentParser()


def _make_action(op, res, patient=None, recipients=None, fields=None):
    return ProposedAction(
        agent_id="agent1",
        operation=op,
        resource_type=res,
        patient_id=patient,
        recipients=recipients or [],
        data_fields=fields or [],
        raw_tool_call={},
    )


class TestConsistencyChecker:

    def test_consistent_action(self, checker, parser):
        intent = parser.parse("Fetch John's blood report")
        action = _make_action(OperationType.READ, ResourceType.LAB_REPORT, patient="John")
        result = checker.check(intent, action)
        assert result.is_consistent is True
        assert result.violations == []

    def test_operation_mismatch_detected(self, checker, parser):
        """Prompt injection: agent tries to DELETE when intent says READ."""
        intent = parser.parse("Fetch John's blood report")
        action = _make_action(OperationType.DELETE, ResourceType.LAB_REPORT, patient="John")
        result = checker.check(intent, action)
        assert result.is_consistent is False
        assert any("mismatch" in v.lower() or "operation" in v.lower() for v in result.violations)

    def test_undeclared_recipient_detected(self, checker, parser):
        """Prompt injection: agent adds an undeclared external recipient."""
        intent = parser.parse("Send John's report to Nurse Sarah")
        action = _make_action(
            OperationType.SEND, ResourceType.LAB_REPORT,
            recipients=["Nurse Sarah", "external@hacker.com"],
        )
        result = checker.check(intent, action)
        assert result.is_consistent is False
        assert any("recipient" in v.lower() for v in result.violations)

    def test_resource_type_mismatch(self, checker, parser):
        """Agent tries to access a different resource than requested."""
        intent = parser.parse("Fetch John's blood report")
        action = _make_action(OperationType.READ, ResourceType.PATIENT_RECORD, patient="John")
        result = checker.check(intent, action)
        assert result.is_consistent is False

    def test_tampered_signature_detected(self, checker, parser):
        """If the golden intent signature is tampered, checker should reject it."""
        intent = parser.parse("Fetch John's blood report")
        intent.signature = "deadbeef" * 8  # invalid signature
        action = _make_action(OperationType.READ, ResourceType.LAB_REPORT, patient="John")
        result = checker.check(intent, action)
        assert result.is_consistent is False
        assert any("signature" in v.lower() for v in result.violations)
