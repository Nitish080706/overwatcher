"""
Module 2: Identity & Device Verification
=========================================
Authenticates the requesting user and assesses device trustworthiness.

Approach (No LLM):
  - JWT token validation via python-jose
  - Role → permission set lookup (static table)
  - Device trust checked via registered device registry (Redis/DB)
  - Output: IdentityContext with verified role and allowed operations

This module runs FIRST in the pipeline. If identity_verified is False,
the Policy Engine immediately BLOCKs — nothing else runs.
"""

from __future__ import annotations
from typing import Optional

from jose import JWTError, jwt

from overwatcher.config import get_settings
from overwatcher.models import (
    IdentityContext,
    OperationType,
    ResourceType,
    UserRole,
)

settings = get_settings()

# ---------------------------------------------------------------------------
# Role-based permission table
# Each role defines exactly which operations and resource types are allowed.
# This is the core IAM layer — no LLM, pure lookup.
# ---------------------------------------------------------------------------
ROLE_PERMISSIONS: dict[UserRole, dict] = {
    UserRole.DOCTOR: {
        "operations": [
            OperationType.READ,
            OperationType.WRITE,
            OperationType.SEND,
            OperationType.UPDATE,
            OperationType.CREATE,
        ],
        "resources": [
            ResourceType.LAB_REPORT,
            ResourceType.PRESCRIPTION,
            ResourceType.MRI,
            ResourceType.PATIENT_RECORD,
            ResourceType.MESSAGE,
            ResourceType.APPOINTMENT,
            ResourceType.DEVICE_COMMAND,
        ],
    },
    UserRole.NURSE: {
        "operations": [
            OperationType.READ,
            OperationType.SEND,
            OperationType.UPDATE,
        ],
        "resources": [
            ResourceType.LAB_REPORT,
            ResourceType.MRI,
            ResourceType.PATIENT_RECORD,
            ResourceType.MESSAGE,
            ResourceType.APPOINTMENT,
        ],
    },
    UserRole.LAB_TECH: {
        "operations": [
            OperationType.READ,
            OperationType.WRITE,
            OperationType.UPDATE,
        ],
        "resources": [
            ResourceType.LAB_REPORT,
        ],
    },
    UserRole.PHARMACIST: {
        "operations": [
            OperationType.READ,
            OperationType.UPDATE,
        ],
        "resources": [
            ResourceType.PRESCRIPTION,
        ],
    },
    UserRole.ADMIN: {
        "operations": [
            OperationType.READ,
            OperationType.CREATE,
            OperationType.UPDATE,
        ],
        "resources": [
            ResourceType.APPOINTMENT,
            ResourceType.BILLING,
            ResourceType.MESSAGE,
        ],
    },
    UserRole.PATIENT: {
        "operations": [
            OperationType.READ,
        ],
        "resources": [
            ResourceType.LAB_REPORT,
            ResourceType.APPOINTMENT,
            ResourceType.PRESCRIPTION,
        ],
    },
}


class IdentityVerifier:
    """
    Verifies JWT token, extracts user role, and checks device trust.
    No LLM involved — pure rule-based lookup.
    """

    def verify(self, auth_token: str, device_id: str) -> IdentityContext:
        """
        Main entry point. Returns an IdentityContext.
        If verification fails at any step, identity_verified=False is returned.
        """
        user_id, role = self._decode_token(auth_token)
        if user_id is None or role is None:
            return self._rejected_context(device_id)

        device_trusted = self._check_device_trust(device_id)

        permissions = ROLE_PERMISSIONS.get(role, {"operations": [], "resources": []})

        return IdentityContext(
            user_id=user_id,
            role=role,
            device_id=device_id,
            device_trusted=device_trusted,
            identity_verified=True,
            allowed_operations=permissions["operations"],
            allowed_resource_types=permissions["resources"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_token(self, token: str) -> tuple[Optional[str], Optional[UserRole]]:
        """Decode and validate a JWT. Returns (user_id, role) or (None, None)."""
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            user_id = payload.get("sub")
            role_str = payload.get("role")
            role = UserRole(role_str) if role_str else None
            return user_id, role
        except (JWTError, ValueError):
            return None, None

    def _check_device_trust(self, device_id: str) -> bool:
        """
        Check whether the device is registered and trusted.
        In production: query MDM/device registry (Redis or DB).
        Stub: always returns True for known device IDs.
        TODO: Replace with real MDM integration.
        """
        # Placeholder — in production, query Redis or DB for device record
        KNOWN_DEVICES = {"HOSP-DEV-001", "HOSP-DEV-002", "HOSP-DEV-003"}
        return device_id in KNOWN_DEVICES

    def _rejected_context(self, device_id: str) -> IdentityContext:
        """Returns a context that signals complete rejection."""
        return IdentityContext(
            user_id="unknown",
            role=UserRole.PATIENT,
            device_id=device_id,
            device_trusted=False,
            identity_verified=False,
            allowed_operations=[],
            allowed_resource_types=[],
        )

    def has_permission(
        self,
        context: IdentityContext,
        operation: OperationType,
        resource: ResourceType,
    ) -> bool:
        """
        Convenience method to check if a verified identity is allowed
        to perform a specific operation on a specific resource type.
        """
        return (
            context.identity_verified
            and operation in context.allowed_operations
            and resource in context.allowed_resource_types
        )
