"""BATO exception hierarchy.

Errors are raised, logged and allowed to propagate. Code must never silence
them (no ``except Exception: pass``). Catch a specific subclass only where a
documented fallback exists, and record that fallback in the stage report.
"""


class BatoError(Exception):
    """Base class for all BATO errors."""


class ConfigError(BatoError):
    """Configuration file is missing, malformed or fails validation."""


class InvalidInputError(BatoError):
    """A caller-supplied request (AOI, date, ...) was rejected."""


class AOIValidationError(InvalidInputError):
    """The area of interest is malformed or outside the configured limits."""


class EventDateError(InvalidInputError):
    """The event date is outside the range the system can process."""


class ProviderError(BatoError):
    """A data provider (e.g. the CDSE catalogue) failed or returned unusable data."""


class CatalogueQueryError(ProviderError):
    """A catalogue request failed: network error, HTTP error, or malformed response."""


class AuthenticationError(ProviderError):
    """Credentials are missing or were rejected. Messages never contain secrets."""


class MetadataError(BatoError):
    """One catalogue record lacks, or has inconsistent, required metadata.

    Raised per record and caught by the catalogue classifier, which records
    the record as REJECTED_INVALID_METADATA with this message as the reason.
    """


class ProhibitedInputError(BatoError):
    """A data-rule violation: an evaluation-only or prohibited source was about
    to be used as a pipeline input. Never catch this to continue processing."""
