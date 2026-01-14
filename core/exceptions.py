"""Custom exception classes for FRED Supreme Litigation OS.

Provides structured error handling across the litigation system.
"""


class LitigationOSError(Exception):
    """Base exception for all litigation system errors."""

    pass


class ConfigurationError(LitigationOSError):
    """Raised when system configuration is invalid or missing."""

    pass


class EvidenceError(LitigationOSError):
    """Raised when evidence processing fails."""

    pass


class ValidationError(LitigationOSError):
    """Raised when validation checks fail."""

    pass


class DocumentProcessingError(LitigationOSError):
    """Raised when document processing fails."""

    pass


class ComplianceError(LitigationOSError):
    """Raised when compliance checks fail (MCR/MCL)."""

    pass


class FilingError(LitigationOSError):
    """Raised when court filing operations fail."""

    pass


class StateManagementError(LitigationOSError):
    """Raised when case state management fails."""

    pass


class IntegrityError(LitigationOSError):
    """Raised when data integrity checks fail."""

    pass


class WorkflowError(LitigationOSError):
    """Raised when workflow execution fails."""

    pass


class ResourceNotFoundError(LitigationOSError):
    """Raised when a required resource cannot be found."""

    def __init__(self, resource_type: str, resource_id: str, message: str = ""):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            message or f"{resource_type} not found: {resource_id}"
        )


class InvalidInputError(LitigationOSError):
    """Raised when user input is invalid."""

    pass


class SystemNotReadyError(LitigationOSError):
    """Raised when system prerequisites are not met."""

    pass
