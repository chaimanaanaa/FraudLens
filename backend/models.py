from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Passage(BaseModel):
    text: str
    score: float
    source: str
    label: str = "unknown"


# ── /extract ──────────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (no data URI prefix needed)")
    filename: Optional[str] = None


class ExtractResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_base64: str
    filename: Optional[str]
    extracted_text: str
    raw_model_output: str
    model_used: str


# ── /retrieve ─────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    extracted_text: str
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    top_k: int = 5


class RetrieveResponse(BaseModel):
    extracted_text: str
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    passages: List[Passage]
    strategy: str
    top_k: int


# ── /analyze ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    extracted_text: str
    passages: List[Passage]
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    is_retry: bool = False


class AnalyzeResponse(BaseModel):
    # pass-through for downstream agents
    extracted_text: str
    passages: List[Passage]
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    # analysis fields
    verdict: str
    confidence_score: float
    fraud_type: Optional[str] = None
    reasons: List[str]
    recommended_actions: List[str]
    cited_sources: List[str]
    is_retry: bool = False

    @field_validator("verdict")
    @classmethod
    def verdict_must_be_valid(cls, v: str) -> str:
        allowed = {"SCAM", "SUSPICIOUS", "LEGITIMATE"}
        if v.upper() not in allowed:
            raise ValueError(f"verdict must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("confidence_score")
    @classmethod
    def confidence_must_be_below_one(cls, v: float) -> float:
        return min(v, 0.95)


# ── /validate ─────────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    extracted_text: str
    passages: List[Passage]
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    verdict: str
    confidence_score: float
    fraud_type: Optional[str] = None
    reasons: List[str]
    recommended_actions: List[str]
    cited_sources: List[str]
    is_retry: bool = False


class ValidationChecks(BaseModel):
    confidence_below_max: bool
    sources_cited: bool
    no_refund_promise: bool
    reasons_sufficient: bool
    actions_present: bool


class ValidateResponse(BaseModel):
    is_valid: bool
    validation_issues: List[str]
    regenerated: bool
    checks: Optional[ValidationChecks] = None
    # final analysis (original or regenerated)
    verdict: str
    confidence_score: float
    fraud_type: Optional[str] = None
    reasons: List[str]
    recommended_actions: List[str]
    cited_sources: List[str]
    # pass-through for Streamlit display
    extracted_text: str
    passages: List[Passage]
