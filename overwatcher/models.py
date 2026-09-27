"""
Core data models (Pydantic schemas) shared across all OVERWATCHER modules.
These are the contracts between modules — no free-form dicts.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OperationType(str, Enum):
    READ   = "read"
    WRITE  = "write"
    SEND   = "send"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    CANCEL = "cancel"


class ResourceType(str, Enum):
    LAB_REPORT     = "lab_report"
    PRESCRIPTION   = "prescription"
    MRI            = "mri"
    PATIENT_RECORD = "patient_record"
    MESSAGE        = "message"
    APPOINTMENT    = "appointment"
    DEVICE_COMMAND = "device_command"
    BILLING        = "billing"
    AUDIT_LOG      = "audit_log"


class UrgencyLevel(str, Enum):
    ROUTINE   = "routine"
    URGENT    = "urgent"
    EMERGENCY = "emergency"


class PolicyDecision(str, Enum):
    ALLOW           = "allow"
    STEP_UP         = "step_up_verification"
    HUMAN_APPROVAL  = "human_approval"
    BLOCK           = "block"


class UserRole(str, Enum):
    DOCTOR    = "doctor"
    NURSE     = "nurse"
    ADMIN     = "admin"
    LAB_TECH  = "lab_tech"
    PHARMACIST = "pharmacist"
    PATIENT   = "patient"


# ---------------------------------------------------------------------------
# Core Schemas
# ---------------------------------------------------------------------------

class ParsedIntent(BaseModel):
    """Output of the Intent Parser — the signed golden intent."""
    operation: OperationType
    resource_type: ResourceType
    patient_id: Optional[str] = None
    recipients: list[str] = Field(default_factory=list)
    data_fields_requested: list[str] = Field(default_factory=list)
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    confidence: float = Field(ge=0.0, le=1.0)
    raw_request: str
    signature: Optional[str] = None           # HMAC signature added at intake


class IdentityContext(BaseModel):
    """Output of Identity & Device Verification."""
    user_id: str
    role: UserRole
    device_id: str
    device_trusted: bool
    identity_verified: bool
    allowed_operations: list[OperationType] = Field(default_factory=list)
    allowed_resource_types: list[ResourceType] = Field(default_factory=list)


class ProposedAction(BaseModel):
    """The actual action the agent is proposing to execute."""
    agent_id: str
    operation: OperationType
    resource_type: ResourceType
    patient_id: Optional[str] = None
    recipients: list[str] = Field(default_factory=list)
    data_fields: list[str] = Field(default_factory=list)
    raw_tool_call: dict                       # Raw OS-level / API-level tool call


class RiskScore(BaseModel):
    """Output of the Risk Scoring Engine."""
    reversibility: float = Field(ge=0.0, le=1.0)
    sensitivity: float   = Field(ge=0.0, le=1.0)
    urgency_modifier: float = Field(ge=0.0, le=1.0)
    composite_score: float  = Field(ge=0.0, le=1.0)


class ConsistencyResult(BaseModel):
    """Output of the Intent-Action Consistency Checker."""
    is_consistent: bool
    violations: list[str] = Field(default_factory=list)
    similarity_score: float = Field(ge=0.0, le=1.0, default=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class TrustProfile(BaseModel):
    """Per-agent trust profile from Behavioral Trust Memory."""
    agent_id: str
    trust_score: float = Field(ge=0.0, le=1.0)
    total_actions: int = 0
    total_violations: int = 0
    last_updated: datetime


class AnomalyResult(BaseModel):
    """Output of the Transformer/HMM Behavior Analyzer."""
    anomaly_score: float = Field(ge=0.0, le=1.0)
    is_anomalous: bool
    sequence_length: int


class PolicyResult(BaseModel):
    """Final decision from the Dynamic Policy Engine."""
    decision: PolicyDecision
    rule_fired: str
    risk_score: float
    trust_score: float
    anomaly_score: float
    is_consistent: bool
    reasons: list[str] = Field(default_factory=list)


class OverwatcherRequest(BaseModel):
    """Full request entering the OVERWATCHER pipeline."""
    session_id: str
    raw_request: str
    proposed_action: ProposedAction
    auth_token: str
    device_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OverwatcherResponse(BaseModel):
    """Full response from the OVERWATCHER pipeline."""
    session_id: str
    agent_id: str
    decision: PolicyDecision
    policy_result: PolicyResult
    parsed_intent: ParsedIntent
    consistency_result: ConsistencyResult
    risk_score: RiskScore
    trust_profile: TrustProfile
    anomaly_result: AnomalyResult
    audit_entry_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
