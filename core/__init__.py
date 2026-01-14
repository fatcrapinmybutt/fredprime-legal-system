"""Core system modules for FRED Supreme Litigation OS.

Provides foundational functionality for the litigation system.
"""

__version__ = "1.0.0"

from .exceptions import (
    LitigationOSError,
    ConfigurationError,
    EvidenceError,
    ValidationError,
    DocumentProcessingError,
    ComplianceError,
    FilingError,
    StateManagementError,
    IntegrityError,
    WorkflowError,
    ResourceNotFoundError,
    InvalidInputError,
    SystemNotReadyError,
)

from .constants import (
    VERSION,
    SYSTEM_NAME,
    EVIDENCE_TYPES,
    DOCUMENT_TYPES,
    CASE_TYPES,
    MCR_COMPLIANCE_RULES,
)

__all__ = [
    "LitigationOSError",
    "ConfigurationError",
    "EvidenceError",
    "ValidationError",
    "DocumentProcessingError",
    "ComplianceError",
    "FilingError",
    "StateManagementError",
    "IntegrityError",
    "WorkflowError",
    "ResourceNotFoundError",
    "InvalidInputError",
    "SystemNotReadyError",
    "VERSION",
    "SYSTEM_NAME",
    "EVIDENCE_TYPES",
    "DOCUMENT_TYPES",
    "CASE_TYPES",
    "MCR_COMPLIANCE_RULES",
]

