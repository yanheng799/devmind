"""Agent state definition for LangGraph using latest API."""

from typing import Annotated, Any

from langgraph.graph import add_messages
from typing_extensions import TypedDict

from devmind.models import (
    Direction,
    ExtractedEvent,
    NewsArticle,
    PredictionResult,
    SentimentAnalysis,
    TimeHorizon,
)


class AgentState(TypedDict):
    """State for the stock prediction agent.

    The state flows through the LangGraph nodes and accumulates
    information at each step.

    Using latest LangGraph API with operator annotations for state updates.
    """

    # Input
    news_article: NewsArticle | None
    stock_code: str | None

    # Intermediate results
    extracted_events: Annotated[list[ExtractedEvent], "Extractor events list"]
    selected_event: ExtractedEvent | None
    sentiment: SentimentAnalysis | None
    similar_events: Annotated[list[dict[str, Any]], "Similar historical events"]
    market_data: dict[str, Any] | None

    # Reasoning
    reasoning_steps: Annotated[list[dict[str, Any]], "Reasoning steps list"]
    current_step: Annotated[int, "Current reasoning step number"]

    # Final output
    prediction: PredictionResult | None

    # Error handling
    error: str | None
    completed: bool

    # Metadata
    iteration_count: Annotated[int, "Number of iterations"]


def create_initial_state(
    article: NewsArticle,
    stock_code: str | None = None,
) -> dict[str, Any]:
    """Create initial agent state.

    Args:
        article: News article to analyze
        stock_code: Target stock code (optional, extracted from article if None)

    Returns:
        Initial agent state as dict
    """
    return {
        "news_article": article,
        "stock_code": stock_code or (article.related_stocks[0] if article.related_stocks else None),
        "extracted_events": [],
        "selected_event": None,
        "sentiment": None,
        "similar_events": [],
        "market_data": None,
        "reasoning_steps": [],
        "current_step": 0,
        "prediction": None,
        "error": None,
        "completed": False,
        "iteration_count": 0,
    }


def update_state_step(
    state: dict[str, Any],
    description: str,
    conclusion: str,
    evidence: list[str] | None = None,
    tool_calls: list[str] | None = None,
) -> dict[str, Any]:
    """Add a reasoning step to the state.

    Args:
        state: Current agent state
        description: Step description
        conclusion: Step conclusion
        evidence: Supporting evidence
        tool_calls: Tools used in this step

    Returns:
        Updated state with new reasoning step
    """
    step = {
        "step_number": state.get("current_step", 0) + 1,
        "description": description,
        "evidence": evidence or [],
        "conclusion": conclusion,
        "tool_calls": tool_calls or [],
    }

    # Create new state with updated values
    new_state = state.copy()
    new_state["reasoning_steps"] = state.get("reasoning_steps", []) + [step]
    new_state["current_step"] = state.get("current_step", 0) + 1

    return new_state
