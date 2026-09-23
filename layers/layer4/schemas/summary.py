from typing import Any, Literal

from pydantic import BaseModel


class EvidenceRef(BaseModel):
    """循证依据引用：来源、发现、依据。"""

    source: str
    finding: str
    basis: str


class StructuredSummary(BaseModel):
    """每层输出附带的结构化临床摘要（通用格式，L4 填充）。"""

    layer_id: Literal["L4"] = "L4"
    timestamp: str
    summary: str
    structured_labels: dict[str, Any]
    evidence_refs: list[EvidenceRef] = []
    confidence: float = 0.0
    flags: list[str] = []
