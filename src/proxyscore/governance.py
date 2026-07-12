"""Governance context and reproducibility manifests for score artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from typing import Any, cast

import numpy as np
import pandas as pd

GOVERNANCE_SCHEMA_VERSION = "1.0"
REDACTED = "[REDACTED]"

_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "auth",
    "connection_string",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)


class GovernanceVersionError(ValueError):
    """Raised when a governance manifest uses an unsupported schema version."""


def _package_version() -> str:
    try:
        return importlib_metadata.version("proxyscore")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0"


def _utc_iso(value: datetime | None = None) -> str:
    stamp = value or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return _utc_iso(value.to_pydatetime() if isinstance(value, pd.Timestamp) else value)
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    return any(marker in normalized for marker in _SECRET_MARKERS)


def redact_secrets(value: Any) -> Any:
    """Return a JSON-safe copy with likely secret values removed.

    Redaction is key-name based by design. The manifest should contain provenance and
    aggregate counts, never row-level data or credentials.
    """
    converted = _json_value(value)
    if isinstance(converted, dict):
        return {
            str(key): REDACTED if _is_secret_key(str(key)) else redact_secrets(item)
            for key, item in converted.items()
        }
    if isinstance(converted, list):
        return [redact_secrets(item) for item in converted]
    return converted


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    converted = cast(dict[str, Any], redact_secrets(value))
    try:
        json.dumps(converted, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only JSON-serializable values") from exc
    return converted


def _string_list(values: Sequence[str] | None, name: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings, not a single string")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings")
        item = value.strip()
        if item:
            cleaned.append(item)
    return cleaned


def configuration_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-safe configuration."""
    canonical = json.dumps(
        redact_secrets(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GovernanceContext:
    """Business ownership and permitted-use context for a proxy score."""

    score_name: str | None = None
    score_version: str | None = None
    owner: str | None = None
    intended_uses: Sequence[str] | None = None
    prohibited_uses: Sequence[str] | None = None
    population: str | None = None
    data_window: str | None = None
    outcome_window: str | None = None
    decision_owner: str | None = None
    reviewer: str | None = None
    tags: Sequence[str] | None = None
    dataset_id: str | None = None
    code_revision_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intended_uses", _string_list(self.intended_uses, "intended_uses"))
        object.__setattr__(
            self,
            "prohibited_uses",
            _string_list(self.prohibited_uses, "prohibited_uses"),
        )
        object.__setattr__(self, "tags", _string_list(self.tags, "tags"))
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))
        for field_name in (
            "score_name",
            "score_version",
            "owner",
            "population",
            "data_window",
            "outcome_window",
            "decision_owner",
            "reviewer",
            "dataset_id",
            "code_revision_id",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GovernanceContext:
        """Create a context from a mapping, rejecting unknown field names."""
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise TypeError(f"unknown governance context field(s): {unknown}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Return a redacted JSON-safe context mapping."""
        return _json_object(asdict(self), "governance context")

    def missing_fields(self) -> list[str]:
        """Required governance fields that have not been supplied."""
        required = {
            "score_name": self.score_name,
            "score_version": self.score_version,
            "owner": self.owner,
            "intended_uses": self.intended_uses,
            "prohibited_uses": self.prohibited_uses,
            "population": self.population,
            "data_window": self.data_window,
            "outcome_window": self.outcome_window,
            "decision_owner": self.decision_owner,
            "reviewer": self.reviewer,
        }
        missing: list[str] = []
        for name, value in required.items():
            if value is None or value == "" or value == []:
                missing.append(name)
        return missing


@dataclass(frozen=True)
class GovernanceManifest:
    """Versioned governance and reproducibility metadata for an artifact."""

    schema_version: str
    generated_at: str
    package_version: str
    context: dict[str, Any]
    row_counts: dict[str, int]
    checks: dict[str, Any]
    thresholds: dict[str, Any]
    configuration_fingerprint: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a redacted JSON-safe manifest mapping."""
        return _json_object(asdict(self), "governance manifest")

    def to_json(self, indent: int = 2) -> str:
        """Serialize the manifest deterministically."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GovernanceManifest:
        """Validate and load a supported manifest mapping."""
        version = values.get("schema_version")
        if version != GOVERNANCE_SCHEMA_VERSION:
            raise GovernanceVersionError(
                f"unsupported governance schema {version!r}; "
                f"this package supports {GOVERNANCE_SCHEMA_VERSION!r}"
            )
        try:
            manifest = cls(**dict(values))
        except TypeError as exc:
            raise GovernanceVersionError(f"malformed governance manifest: {exc}") from exc
        manifest.validate()
        return manifest

    @classmethod
    def from_json(cls, document: str) -> GovernanceManifest:
        """Load a manifest from a JSON document."""
        try:
            values = json.loads(document)
        except json.JSONDecodeError as exc:
            raise GovernanceVersionError(f"invalid governance JSON: {exc}") from exc
        if not isinstance(values, dict):
            raise GovernanceVersionError("governance JSON must contain an object")
        return cls.from_dict(values)

    def validate(self) -> None:
        """Reject malformed governance state."""
        if self.schema_version != GOVERNANCE_SCHEMA_VERSION:
            raise GovernanceVersionError(f"unsupported governance schema {self.schema_version!r}")
        if not isinstance(self.context, dict):
            raise GovernanceVersionError("governance context must be an object")
        if not isinstance(self.row_counts, dict) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.row_counts.values()
        ):
            raise GovernanceVersionError("governance row_counts must contain non-negative integers")
        if not isinstance(self.checks, dict):
            raise GovernanceVersionError("governance checks must be an object")
        if not isinstance(self.thresholds, dict):
            raise GovernanceVersionError("governance thresholds must be an object")
        if (
            not isinstance(self.configuration_fingerprint, str)
            or not self.configuration_fingerprint
        ):
            raise GovernanceVersionError("governance configuration_fingerprint must be non-empty")
        if not isinstance(self.warnings, list) or any(
            not isinstance(item, str) for item in self.warnings
        ):
            raise GovernanceVersionError("governance warnings must be strings")


def _coerce_context(
    governance: GovernanceContext | Mapping[str, Any] | None,
) -> GovernanceContext:
    if governance is None:
        return GovernanceContext()
    if isinstance(governance, GovernanceContext):
        return governance
    if isinstance(governance, Mapping):
        return GovernanceContext.from_dict(governance)
    raise TypeError("governance must be a GovernanceContext, mapping, or None")


def create_governance_manifest(
    governance: GovernanceContext | Mapping[str, Any] | None = None,
    *,
    row_counts: Mapping[str, int] | None = None,
    checks: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    strict: bool = False,
) -> GovernanceManifest:
    """Build a versioned manifest without storing row-level data or secrets."""
    context = _coerce_context(governance)
    missing = context.missing_fields()
    warnings = [f"Missing governance field: {field_name}" for field_name in missing]
    if strict and missing:
        raise ValueError("missing required governance field(s): " + ", ".join(missing))

    safe_row_counts = _json_object(row_counts or {}, "row_counts")
    row_count_values: dict[str, int] = {}
    for key, value in safe_row_counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("row_counts must contain non-negative integers")
        row_count_values[key] = value

    safe_checks = _json_object(checks or {}, "checks")
    safe_thresholds = _json_object(thresholds or {}, "thresholds")
    fingerprint_input = {
        "context": context.to_dict(),
        "thresholds": safe_thresholds,
        "configuration": _json_object(configuration or {}, "configuration"),
    }
    manifest = GovernanceManifest(
        schema_version=GOVERNANCE_SCHEMA_VERSION,
        generated_at=_utc_iso(generated_at),
        package_version=_package_version(),
        context=context.to_dict(),
        row_counts=row_count_values,
        checks=safe_checks,
        thresholds=safe_thresholds,
        configuration_fingerprint=configuration_fingerprint(fingerprint_input),
        warnings=warnings,
    )
    manifest.validate()
    return manifest
