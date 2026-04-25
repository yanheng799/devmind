"""Prediction result models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from devmind.models.types import Direction, TimeHorizon


class ReasoningStep(BaseModel):
    """Single step in reasoning chain.

    Attributes:
        step_number: Order in the reasoning chain
        description: What was considered
        evidence: Supporting evidence
        conclusion: Intermediate conclusion
        tool_calls: Tools used in this step
    """

    step_number: int = Field(..., ge=1, description="Step order")
    description: str = Field(..., description="What was considered")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    conclusion: str = Field(..., description="Intermediate conclusion")
    tool_calls: list[str] = Field(default_factory=list, description="Tools used")


class ReasoningChain(BaseModel):
    """Complete reasoning chain for prediction.

    Attributes:
        steps: List of reasoning steps
        final_conclusion: Overall conclusion
        confidence_factors: Factors affecting confidence
        contradictory_evidence: Evidence against the prediction
    """

    steps: list[ReasoningStep] = Field(..., min_length=1, description="Reasoning steps")
    final_conclusion: str = Field(..., description="Overall conclusion")
    confidence_factors: dict[str, str] = Field(
        default_factory=dict,
        description="Factors affecting confidence",
    )
    contradictory_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence against prediction",
    )


class RiskFactor(BaseModel):
    """Risk factor for prediction.

    Attributes:
        category: Risk category
        description: Risk description
        severity: HIGH, MEDIUM, or LOW
        mitigation: Potential mitigation strategies
    """

    category: str = Field(..., description="Risk category")
    description: str = Field(..., description="Risk description")
    severity: str = Field(..., description="Severity: high/medium/low")
    mitigation: str | None = Field(default=None, description="Mitigation strategy")


class PredictionResult(BaseModel):
    """Stock price prediction result.

    Attributes:
        prediction_id: Unique identifier
        stock_code: Target stock code
        stock_name: Stock name
        direction: BULLISH, BEARISH, or NEUTRAL
        probability: Estimated probability (0-1)
        target_range: Expected price range
        time_horizon: Prediction timeframe
        confidence: Overall confidence (0-1)
        reasoning_chain: Complete reasoning chain
        risk_factors: List of risk factors
        similar_events: Similar historical events
        created_at: Prediction timestamp
        status: Prediction status
    """

    prediction_id: str = Field(..., description="Unique prediction identifier")
    stock_code: str = Field(..., description="Target stock code (e.g., '600519.SH')")
    stock_name: str = Field(..., description="Stock name")
    direction: Direction = Field(..., description="Predicted direction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Success probability")
    target_range: str = Field(..., description="Expected price range")
    time_horizon: TimeHorizon = Field(..., description="Prediction timeframe")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    reasoning_chain: ReasoningChain = Field(..., description="Complete reasoning chain")
    risk_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="Risk factors",
    )
    similar_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Similar historical events",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Prediction timestamp",
    )
    status: str = Field(default="pending", description="pending/confirmed/refuted")

    @property
    def is_bullish(self) -> bool:
        """Check if prediction is bullish."""
        return self.direction == Direction.BULLISH

    @property
    def is_bearish(self) -> bool:
        """Check if prediction is bearish."""
        return self.direction == Direction.BEARISH

    @property
    def is_neutral(self) -> bool:
        """Check if prediction is neutral."""
        return self.direction == Direction.NEUTRAL


class PredictionOutcome(BaseModel):
    """Actual outcome of a prediction.

    Used for backtesting and accuracy tracking.

    Attributes:
        prediction_id: Source prediction ID
        actual_direction: Actual price direction
        actual_change_pct: Actual price change percentage
        outcome_date: Date outcome was determined
        is_correct: Whether prediction was correct
        magnitude_error: Error in magnitude prediction
        notes: Additional notes
    """

    prediction_id: str = Field(..., description="Source prediction ID")
    actual_direction: Direction = Field(..., description="Actual price direction")
    actual_change_pct: float = Field(..., description="Actual price change %")
    outcome_date: datetime = Field(..., description="Outcome determination date")
    is_correct: bool = Field(..., description="Whether prediction was correct")
    magnitude_error: float | None = Field(
        default=None,
        description="Error in magnitude prediction",
    )
    notes: str | None = Field(default=None, description="Additional notes")
