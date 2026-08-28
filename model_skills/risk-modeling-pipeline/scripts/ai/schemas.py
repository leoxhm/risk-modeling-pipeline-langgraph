"""Shared JSON-safe schemas for AI advisory outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Severity = Literal["info", "warning", "blocker"]


@dataclass(frozen=True)
class Finding:
    """One reviewable conclusion tied to deterministic evidence."""

    code: str
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def report_status(findings: list[Finding]) -> str:
    """Return a compact status derived only from finding severities."""

    severities = {finding.severity for finding in findings}
    if "blocker" in severities:
        return "blocked"
    if "warning" in severities:
        return "needs_review"
    return "ready"


def finding_dicts(findings: list[Finding]) -> list[dict[str, Any]]:
    return [finding.to_dict() for finding in findings]
