"""Unit tests for agent components."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from devmind.agents.graph.prediction_graph import (
    extract_event_node,
    analyze_sentiment_node,
    retrieve_history_node,
    reasoning_node,
    create_initial_state,
    AgentState,
)
from devmind.agents.stock_agent import StockPredictionAgent
from devmind.models import (
    NewsArticle,
    SourceType,
    Direction,
)


class TestAgentState:
    """Tests for AgentState."""

    def test_create_initial_state(self) -> None:
        """Test creating initial agent state."""
        article = NewsArticle(
            article_id="test_001",
            title="Test Article",
            content="Test content about 600519.SH stock",
            source="test",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com",
            related_stocks=["600519.SH"],
        )

        state = create_initial_state(article, "600519.SH")

        assert state["news_article"] == article
        assert state["stock_code"] == "600519.SH"
        assert state["extracted_events"] == []
        assert state["completed"] is False
        assert state["error"] is None

    def test_create_state_without_stock_code(self) -> None:
        """Test creating state with stock from article."""
        article = NewsArticle(
            article_id="test_002",
            title="Test",
            content="Content",
            source="test",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com",
            related_stocks=["000858.SZ"],
        )

        state = create_initial_state(article)

        assert state["stock_code"] == "000858.SZ"


class TestExtractEventNode:
    """Tests for extract_event_node."""

    def test_extract_events_from_article(self) -> None:
        """Test extracting events from article."""
        article = NewsArticle(
            article_id="test_001",
            title="央行宣布降准",
            content="中国人民银行宣布下调存款准备金率0.5个百分点",
            source="pbc",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com",
        )

        state = create_initial_state(article)
        result_state = extract_event_node(state)

        assert len(result_state["extracted_events"]) > 0
        assert result_state["selected_event"] is not None
        assert result_state["current_step"] > 0

    def test_no_article_error(self) -> None:
        """Test handling of missing article."""
        state: AgentState = {
            "news_article": None,
            "stock_code": "600519.SH",
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

        result_state = extract_event_node(state)

        assert result_state["error"] is not None
        assert result_state["completed"] is True


class TestAnalyzeSentimentNode:
    """Tests for analyze_sentiment_node."""

    def test_analyze_sentiment(self) -> None:
        """Test sentiment analysis."""
        article = NewsArticle(
            article_id="test_001",
            title="利好消息",
            content="央行降准，银行股上涨",
            source="test",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com",
        )

        state: AgentState = {
            "news_article": article,
            "stock_code": "601398.SH",
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

        result_state = analyze_sentiment_node(state)

        assert result_state["sentiment"] is not None
        assert result_state["sentiment"].sentiment is not None


class TestRetrieveHistoryNode:
    """Tests for retrieve_history_node."""

    def test_retrieve_similar_events(self) -> None:
        """Test retrieving similar historical events."""
        from devmind.models import ExtractedEvent, EventType, TimeHorizon

        event = ExtractedEvent(
            event_id="evt_001",
            article_id="test_001",
            event_type=EventType.MONETARY_POLICY,
            entities=["央行"],
            magnitude="high",
            timeframe=TimeHorizon.SHORT,
            transmission_chain=["降准"],
            confidence=0.9,
            raw_evidence="央行降准",
        )

        state: AgentState = {
            "news_article": None,
            "stock_code": "601398.SH",
            "extracted_events": [],
            "selected_event": event,
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

        result_state = retrieve_history_node(state)

        # With mock, should complete without error
        assert result_state["error"] is None


class TestReasoningNode:
    """Tests for reasoning_node."""

    def test_reasoning_and_prediction(self) -> None:
        """Test reasoning and prediction generation."""
        from devmind.models import ExtractedEvent, SentimentAnalysis, Sentiment, EventType, TimeHorizon

        event = ExtractedEvent(
            event_id="evt_001",
            article_id="test_001",
            event_type=EventType.MONETARY_POLICY,
            entities=["央行"],
            magnitude="high",
            timeframe=TimeHorizon.SHORT,
            transmission_chain=["降准"],
            confidence=0.9,
            raw_evidence="央行降准",
        )

        sentiment = SentimentAnalysis(
            article_id="test_001",
            sentiment=Sentiment.POSITIVE,
            score=0.7,
            confidence=0.8,
        )

        state: AgentState = {
            "news_article": None,
            "stock_code": "601398.SH",
            "extracted_events": [],
            "selected_event": event,
            "sentiment": sentiment,
            "similar_events": [],
            "market_data": None,
            "reasoning_steps": [],
            "current_step": 0,
            "prediction": None,
            "error": None,
            "completed": False,
            "iteration_count": 0,
        }

        result_state = reasoning_node(state)

        assert result_state["prediction"] is not None
        assert result_state["completed"] is True
        assert result_state["prediction"].direction in [Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL]


class TestStockPredictionAgent:
    """Tests for StockPredictionAgent."""

    def test_init_agent(self) -> None:
        """Test initializing the agent."""
        agent = StockPredictionAgent(use_mock=True)

        assert agent.use_mock is True
        assert agent.news_collector is not None
        assert agent.market_collector is not None

        agent.close()

    def test_predict_from_article(self) -> None:
        """Test prediction from article."""
        article = NewsArticle(
            article_id="test_001",
            title="央行降准",
            content="央行宣布降准0.5个百分点",
            source="pbc",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com",
            related_stocks=["601398.SH"],
        )

        agent = StockPredictionAgent(use_mock=True)
        result = agent.predict_from_article(article, "601398.SH")

        assert result["success"] is True
        assert result["prediction"] is not None

        agent.close()

    def test_fetch_and_predict(self) -> None:
        """Test fetching and predicting from latest news."""
        agent = StockPredictionAgent(use_mock=True)
        results = agent.fetch_and_predict_latest(limit=2)

        assert len(results) > 0

        agent.close()

    def test_query_stock(self) -> None:
        """Test querying stock information."""
        agent = StockPredictionAgent(use_mock=True)
        result = agent.query_stock("600519.SH")

        assert result["success"] is True
        assert result["info"] is not None
        assert result["info"]["stock_name"] == "贵州茅台"

        agent.close()

    def test_context_manager(self) -> None:
        """Test agent as context manager."""
        with StockPredictionAgent(use_mock=True) as agent:
            assert agent is not None
        # Should close automatically
