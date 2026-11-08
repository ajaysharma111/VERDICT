from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    id: str
    symbol: str
    exchange: str = "NSE"
    source: str          # DEMO | ANGELONE | BACKTEST
    timestamp: str
    finding: str
    value: str
    interpretation: str


class Proposal(BaseModel):
    action: Literal["BUY", "NO_TRADE"]
    symbol: str
    strategy: str
    entry: Optional[Decimal] = None
    stop: Optional[Decimal] = None
    target: Optional[Decimal] = None
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class Argument(BaseModel):
    role: Literal["PROSECUTOR", "DEFENDER"]
    points: list[str]
    evidence_ids: list[str]
    severity: float = Field(ge=0, le=1)  # strength of the case, 0-1

class JudgeAssessment(BaseModel):
    decision: Literal["APPROVE", "REDUCE", "REJECT"]
    reasoning: str
    supporting_ids: list[str]
    contradicting_ids: list[str]
    confidence: float = Field(ge=0, le=1)

class Verdict(BaseModel):
    decision: Literal["APPROVE", "REDUCE", "REJECT"]
    multiplier: Decimal = Field(ge=0, le=1)
    reasoning: str
    supporting_ids: list[str]
    contradicting_ids: list[str]
    i_am_wrong_if: str
