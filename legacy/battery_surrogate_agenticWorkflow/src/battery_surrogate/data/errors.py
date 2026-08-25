"""Shared error types for the workflow package."""


class MissingOpError(FileNotFoundError):
    """Raised when an OP folder or OP record is missing."""


class MissingCoordinatesError(FileNotFoundError):
    """Raised when the thermal coordinate files are missing or malformed."""


class InsufficientDiskError(RuntimeError):
    """Raised when the free disk space is not enough for ingest or build."""


class SchemaChangelogError(RuntimeError):
    """Raised when the schema version is not listed in the changelog."""


class UnknownInputsignaleValueError(ValueError):
    """Raised when Inputsignale contains a value that the parser cannot classify."""
