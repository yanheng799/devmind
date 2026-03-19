"""LangGraph prediction graph for stock price prediction using latest API."""

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from devmind.agents.graph.state import AgentState, create_initial_state, update_state_step
from devmind.data.database.database import PredictionDatabase
from devmind.data.processors.event_extractor import MockEventExtractor
from devmind.data.processors.sentiment_analyzer import MockSentimentAnalyzer
from devmind.agents.tools import market_tools, vector_tools
from devmind.models import (
    Direction,
    PredictionResult,
    ReasoningChain,
    ReasoningStep,
    RiskFactor,
    TimeHorizon,
)


def extract_event_node(state: AgentState) -> dict[str, Any]:
    """Extract events from news article.

    Node function that returns partial state update.

    Args:
        state: Current agent state

    Returns:
        Partial state update dict
    """
    article = state.get("news_article")
    if not article:
        return {
            "error": "No news article provided",
            "completed": True,
        }

    try:
        # Use mock extractor for now
        extractor = MockEventExtractor()
        events = extractor.extract_events(article)

        updates: dict[str, Any] = {
            "extracted_events": events,
        }

        # Select most confident event
        if events:
            events.sort(key=lambda e: e.confidence, reverse=True)
            updates["selected_event"] = events[0]

            # Add reasoning step
            new_state = update_state_step(
                state,
                description=f"从新闻中提取了{len(events)}个事件",
                conclusion=f"选择置信度最高的事件: {events[0].event_type.value}",
                evidence=[events[0].raw_evidence],
            )
            updates["reasoning_steps"] = new_state["reasoning_steps"]
            updates["current_step"] = new_state["current_step"]
        else:
            updates["error"] = "No events extracted from article"
            updates["completed"] = True

        return updates

    except Exception as e:
        return {
            "error": f"Event extraction failed: {e}",
            "completed": True,
        }


def analyze_sentiment_node(state: AgentState) -> dict[str, Any]:
    """Analyze sentiment of news article.

    Args:
        state: Current agent state

    Returns:
        Partial state update dict
    """
    article = state.get("news_article")
    if not article:
        return {"error": "No news article for sentiment analysis"}

    try:
        # Use mock analyzer for now
        analyzer = MockSentimentAnalyzer()
        sentiment = analyzer.analyze(article)

        # Add reasoning step
        new_state = update_state_step(
            state,
            description="分析新闻情感倾向",
            conclusion=f"情感: {sentiment.sentiment.value}, 得分: {sentiment.score:.2f}",
            evidence=[f"置信度: {sentiment.confidence:.2f}"],
        )

        return {
            "sentiment": sentiment,
            "reasoning_steps": new_state["reasoning_steps"],
            "current_step": new_state["current_step"],
        }

    except Exception as e:
        return {"error": f"Sentiment analysis failed: {e}"}


def retrieve_history_node(state: AgentState) -> dict[str, Any]:
    """Retrieve similar historical events.

    Args:
        state: Current agent state

    Returns:
        Partial state update dict
    """
    event = state.get("selected_event")
    stock_code = state.get("stock_code")

    if not event or not stock_code:
        return {}

    try:
        # Query for similar events
        result = vector_tools.query_historical_events(
            query_text=event.raw_evidence,
            event_type=event.event_type.value,
            stock_code=stock_code,
            top_k=5,
            use_mock=True,  # Use mock for now
        )

        updates: dict[str, Any] = {}

        if result.get("success"):
            similar_events = result.get("events", [])
            updates["similar_events"] = similar_events

            # Analyze similar events and add reasoning step
            if similar_events:
                bullish_count = sum(1 for e in similar_events if e.get("actual_direction") == "bullish")
                bearish_count = sum(1 for e in similar_events if e.get("actual_direction") == "bearish")

                new_state = update_state_step(
                    state,
                    description=f"检索到{len(similar_events)}个相似历史事件",
                    conclusion=f"历史事件中: {bullish_count}个看涨, {bearish_count}个看跌",
                    evidence=[f"最相似事件: {similar_events[0].get('description', 'N/A')[:50]}..."],
                    tool_calls=["query_historical_events"],
                )
            else:
                new_state = update_state_step(
                    state,
                    description="检索历史相似事件",
                    conclusion="未找到相似的历史事件",
                    tool_calls=["query_historical_events"],
                )

            updates["reasoning_steps"] = new_state["reasoning_steps"]
            updates["current_step"] = new_state["current_step"]

        return updates

    except Exception as e:
        return {"error": f"History retrieval failed: {e}"}


def reasoning_node(state: AgentState) -> dict[str, Any]:
    """Perform reasoning and gather market data.

    Args:
        state: Current agent state

    Returns:
        Partial state update with prediction
    """
    stock_code = state.get("stock_code")
    event = state.get("selected_event")
    sentiment = state.get("sentiment")

    if not stock_code:
        return {
            "error": "No stock code for reasoning",
            "completed": True,
        }

    try:
        # Get current stock price
        price_result = market_tools.query_stock_price(stock_code, use_mock=True)

        if not price_result.get("success"):
            return {
                "error": f"Failed to get market data: {price_result.get('error')}",
                "completed": True,
            }

        current_price = price_result.get("close", 0)

        # Get stock info
        info_result = market_tools.query_stock_info(stock_code, use_mock=True)

        stock_name = stock_code
        if info_result.get("success"):
            stock_name = info_result.get("stock_name", stock_code)

        # Combine evidence for reasoning
        evidence_parts = []

        # Price evidence
        if current_price:
            evidence_parts.append(f"当前股价: {current_price:.2f}")

        # Sentiment evidence
        if sentiment:
            evidence_parts.append(f"新闻情感: {sentiment.sentiment.value} (得分: {sentiment.score:.2f})")

        # Event evidence
        if event:
            evidence_parts.append(f"事件类型: {event.event_type.value}")
            evidence_parts.append(f"影响程度: {event.magnitude}")

        # Historical evidence
        similar_events = state.get("similar_events", [])
        if similar_events:
            bullish = sum(1 for e in similar_events if e.get("actual_direction") == "bullish")
            bearish = sum(1 for e in similar_events if e.get("actual_direction") == "bearish")
            evidence_parts.append(f"历史相似事件: {len(similar_events)}个 (看涨: {bullish}, 看跌: {bearish})")

        # Make prediction based on combined evidence
        direction, probability, target_range = _make_prediction(
            sentiment, event, similar_events, current_price,
        )

        # Add final reasoning step
        new_state = update_state_step(
            state,
            description="综合分析市场数据、情感和历史案例",
            conclusion=f"预测: {direction.value}, 概率: {probability:.2f}",
            evidence=evidence_parts,
            tool_calls=["query_stock_price", "query_stock_info"],
        )

        # Create prediction
        prediction = _create_prediction(
            stock_code=stock_code,
            stock_name=stock_name,
            direction=direction,
            probability=probability,
            target_range=target_range,
            confidence=0.65,  # Base confidence
            reasoning_steps=new_state["reasoning_steps"],
            similar_events=similar_events,
        )

        return {
            "market_data": price_result,
            "prediction": prediction,
            "reasoning_steps": new_state["reasoning_steps"],
            "current_step": new_state["current_step"],
            "completed": True,
        }

    except Exception as e:
        return {
            "error": f"Reasoning failed: {e}",
            "completed": True,
        }


def _make_prediction(
    sentiment: Any | None,
    event: Any | None,
    similar_events: list[Any],
    current_price: float,
) -> tuple[Direction, float, str]:
    """Make prediction based on available evidence.

    Args:
        sentiment: Sentiment analysis result
        event: Selected event
        similar_events: Similar historical events
        current_price: Current stock price

    Returns:
        Tuple of (direction, probability, target_range)
    """
    score = 0.0

    # Sentiment factor (weight: 0.3)
    if sentiment:
        sentiment_score = sentiment.score * 0.3
        score += sentiment_score

    # Event magnitude factor (weight: 0.4)
    if event:
        magnitude_score = 0.0
        if event.magnitude == "high":
            magnitude_score = 0.4 if sentiment and sentiment.score > 0 else -0.4
        elif event.magnitude == "medium":
            magnitude_score = 0.2 if sentiment and sentiment.score > 0 else -0.2
        else:  # low
            magnitude_score = 0.1 if sentiment and sentiment.score > 0 else -0.1
        score += magnitude_score

    # Historical events factor (weight: 0.3)
    if similar_events:
        bullish_count = sum(1 for e in similar_events if e.get("actual_direction") == "bullish")
        total = len(similar_events)
        if total > 0:
            historical_score = ((bullish_count / total) - 0.5) * 2 * 0.3  # -0.3 to 0.3
            score += historical_score

    # Determine direction
    if score > 0.2:
        direction = Direction.BULLISH
    elif score < -0.2:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    # Calculate probability (0.5 to 0.9)
    probability = min(0.9, 0.5 + abs(score))

    # Calculate target range (simplified)
    if current_price > 0:
        change_pct = abs(score) * 0.05  # Up to 5% change
        if direction == Direction.BULLISH:
            low = current_price * (1 - change_pct * 0.5)
            high = current_price * (1 + change_pct)
        elif direction == Direction.BEARISH:
            low = current_price * (1 - change_pct)
            high = current_price * (1 + change_pct * 0.5)
        else:
            low = current_price * 0.99
            high = current_price * 1.01

        target_range = f"{low:.2f} - {high:.2f}"
    else:
        target_range = "未知"

    return direction, probability, target_range


def _create_prediction(
    stock_code: str,
    stock_name: str,
    direction: Direction,
    probability: float,
    target_range: str,
    confidence: float,
    reasoning_steps: list[dict[str, Any]],
    similar_events: list[Any],
) -> PredictionResult:
    """Create a prediction result.

    Args:
        stock_code: Stock code
        stock_name: Stock name
        direction: Predicted direction
        probability: Success probability
        target_range: Target price range
        confidence: Overall confidence
        reasoning_steps: Reasoning steps
        similar_events: Similar historical events

    Returns:
        PredictionResult object
    """
    import uuid
    from datetime import datetime

    # Create reasoning chain
    steps = [
        ReasoningStep(
            step_number=i + 1,
            description=step["description"],
            evidence=step.get("evidence", []),
            conclusion=step["conclusion"],
            tool_calls=step.get("tool_calls", []),
        )
        for i, step in enumerate(reasoning_steps)
    ]

    final_conclusion = f"基于{len(reasoning_steps)}步推理，预测{stock_name}({stock_code})将{direction.value}"

    reasoning_chain = ReasoningChain(
        steps=steps,
        final_conclusion=final_conclusion,
        confidence_factors={
            "sentiment_analysis": "新闻情感分析",
            "event_extraction": "事件提取与影响评估",
            "historical_similarity": "历史相似案例",
        },
        contradictory_evidence=[],
    )

    # Create risk factors
    risk_factors: list[RiskFactor] = []
    if probability < 0.6:
        risk_factors.append(RiskFactor(
            category="置信度风险",
            description="预测置信度较低，建议谨慎决策",
            severity="medium",
            mitigation="结合其他分析方法和市场信息",
        ))

    if direction == Direction.BULLISH:
        risk_factors.append(RiskFactor(
            category="市场风险",
            description="若市场整体下跌，可能影响预测结果",
            severity="low",
            mitigation="关注大盘走势和市场情绪",
        ))

    # Format similar events for output
    similar_events_formatted = [
        {
            "event_id": e.get("event_id"),
            "description": e.get("description"),
            "actual_direction": e.get("actual_direction"),
            "similarity": e.get("similarity_score", 0),
        }
        for e in similar_events[:3]
    ]

    return PredictionResult(
        prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
        stock_code=stock_code,
        stock_name=stock_name,
        direction=direction,
        probability=probability,
        target_range=target_range,
        time_horizon=TimeHorizon.SHORT,
        confidence=confidence,
        reasoning_chain=reasoning_chain,
        risk_factors=risk_factors,
        similar_events=similar_events_formatted,
        created_at=datetime.now(),
    )


def create_prediction_graph() -> StateGraph:
    """Create the LangGraph prediction graph using latest API.

    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create graph with state schema
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extract_event", extract_event_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("retrieve_history", retrieve_history_node)
    workflow.add_node("reasoning", reasoning_node)

    # Set entry point
    workflow.set_entry_point("extract_event")

    # Add edges (sequential flow)
    workflow.add_edge("extract_event", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "retrieve_history")
    workflow.add_edge("retrieve_history", "reasoning")
    workflow.add_edge("reasoning", END)

    # Compile graph
    return workflow.compile()


def run_prediction(
    article: Any,
    stock_code: str | None = None,
) -> dict[str, Any]:
    """Run prediction on a news article.

    Args:
        article: News article
        stock_code: Target stock code (optional)

    Returns:
        Dict with prediction result or error
    """
    try:
        # Create initial state
        initial_state = create_initial_state(article, stock_code)

        # Create and run graph
        graph = create_prediction_graph()
        result_state = graph.invoke(initial_state)

        if result_state.get("error"):
            return {
                "success": False,
                "error": result_state["error"],
            }

        prediction = result_state.get("prediction")
        if prediction:
            # Save to database
            db = PredictionDatabase()
            db.insert_prediction(prediction.model_dump())

            return {
                "success": True,
                "prediction": prediction.model_dump(),
            }
        else:
            return {
                "success": False,
                "error": "No prediction generated",
            }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }
