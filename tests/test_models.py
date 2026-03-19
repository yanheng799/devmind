"""Unit tests for data models."""

import pytest
from datetime import datetime
from decimal import Decimal

from devmind.models import (
    Direction,
    EventType,
    ExtractedEvent,
    HistoricalEvent,
    NewsArticle,
    PredictionResult,
    ReasoningChain,
    ReasoningStep,
    RiskFactor,
    Sentiment,
    SentimentAnalysis,
    SourceType,
    StockPrice,
    TimeHorizon,
)


class TestNewsArticle:
    """Tests for NewsArticle model."""

    def test_create_valid_article(self) -> None:
        """Test creating a valid news article."""
        article = NewsArticle(
            article_id="test_001",
            title="Test Article",
            content="This is test content.",
            source="test_source",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com/article",
        )
        assert article.article_id == "test_001"
        assert article.title == "Test Article"
        assert len(article.related_stocks) == 0

    def test_article_with_related_stocks(self) -> None:
        """Test article with related stock codes."""
        article = NewsArticle(
            article_id="test_002",
            title="Stock News",
            content="News about stocks.",
            source="eastmoney",
            source_type=SourceType.PRIMARY,
            publish_time=datetime.now(),
            url="https://example.com/article",
            related_stocks=["600519.SH", "000858.SZ"],
        )
        assert len(article.related_stocks) == 2
        assert "600519.SH" in article.related_stocks

    def test_invalid_url_raises_error(self) -> None:
        """Test that invalid URL raises validation error."""
        with pytest.raises(ValueError):
            NewsArticle(
                article_id="test_003",
                title="Test",
                content="Content",
                source="test",
                source_type=SourceType.PRIMARY,
                publish_time=datetime.now(),
                url="not-a-url",
            )

    def test_invalid_stock_code_raises_error(self) -> None:
        """Test that invalid stock code raises validation error."""
        with pytest.raises(ValueError):
            NewsArticle(
                article_id="test_004",
                title="Test",
                content="Content",
                source="test",
                source_type=SourceType.PRIMARY,
                publish_time=datetime.now(),
                url="https://example.com",
                related_stocks=["INVALID"],
            )


class TestExtractedEvent:
    """Tests for ExtractedEvent model."""

    def test_create_valid_event(self) -> None:
        """Test creating a valid extracted event."""
        event = ExtractedEvent(
            event_id="evt_001",
            article_id="test_001",
            event_type=EventType.MONETARY_POLICY,
            entities=["央行", "银行"],
            magnitude="high",
            timeframe=TimeHorizon.SHORT,
            transmission_chain=["降准", "银行利润增加", "股价上涨"],
            confidence=0.9,
            raw_evidence="央行宣布降准",
        )
        assert event.event_type == EventType.MONETARY_POLICY
        assert event.confidence == 0.9
        assert len(event.transmission_chain) == 3

    def test_magnitude_validation(self) -> None:
        """Test magnitude value validation."""
        event = ExtractedEvent(
            event_id="evt_002",
            article_id="test_001",
            event_type=EventType.EARNINGS,
            entities=["公司"],
            magnitude="High",  # Should be lowercased
            timeframe=TimeHorizon.IMMEDIATE,
            transmission_chain=["业绩发布"],
            confidence=0.8,
            raw_evidence="业绩增长",
        )
        assert event.magnitude == "high"


class TestSentimentAnalysis:
    """Tests for SentimentAnalysis model."""

    def test_create_sentiment(self) -> None:
        """Test creating sentiment analysis."""
        sentiment = SentimentAnalysis(
            article_id="test_001",
            sentiment=Sentiment.POSITIVE,
            score=0.75,
            confidence=0.85,
            aspects={"业绩": "positive", "政策": "neutral"},
        )
        assert sentiment.sentiment == Sentiment.POSITIVE
        assert sentiment.score == 0.75
        assert "业绩" in sentiment.aspects

    def test_score_range_validation(self) -> None:
        """Test score range validation."""
        with pytest.raises(ValueError):
            SentimentAnalysis(
                article_id="test_001",
                sentiment=Sentiment.POSITIVE,
                score=1.5,  # Invalid, must be -1 to 1
                confidence=0.8,
            )


class TestStockPrice:
    """Tests for StockPrice model."""

    def test_create_stock_price(self) -> None:
        """Test creating stock price data."""
        price = StockPrice(
            stock_code="600519.SH",
            date=datetime.now(),
            open=Decimal("1750.0"),
            high=Decimal("1780.0"),
            low=Decimal("1745.0"),
            close=Decimal("1770.0"),
            volume=2000000,
            change_pct=1.2,
        )
        assert price.stock_code == "600519.SH"
        assert price.close == Decimal("1770.0")
        assert price.volume == 2000000

    def test_stock_code_validation(self) -> None:
        """Test stock code format validation."""
        with pytest.raises(ValueError):
            StockPrice(
                stock_code="INVALID",
                date=datetime.now(),
                open=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("90"),
                close=Decimal("105"),
            )

    def test_is_trading_day(self) -> None:
        """Test is_trading_day property."""
        price_with_volume = StockPrice(
            stock_code="600519.SH",
            date=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=1000000,
        )
        assert price_with_volume.is_trading_day is True

        price_no_volume = StockPrice(
            stock_code="600519.SH",
            date=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
        )
        assert price_no_volume.is_trading_day is False


class TestPredictionResult:
    """Tests for PredictionResult model."""

    def test_create_prediction(self) -> None:
        """Test creating a prediction result."""
        reasoning = ReasoningChain(
            steps=[
                ReasoningStep(
                    step_number=1,
                    description="分析新闻",
                    conclusion="看涨",
                    evidence=["利好消息"],
                )
            ],
            final_conclusion="预测股价上涨",
        )

        prediction = PredictionResult(
            prediction_id="pred_001",
            stock_code="600519.SH",
            stock_name="贵州茅台",
            direction=Direction.BULLISH,
            probability=0.7,
            target_range="1780-1820",
            time_horizon=TimeHorizon.SHORT,
            confidence=0.65,
            reasoning_chain=reasoning,
        )
        assert prediction.is_bullish is True
        assert prediction.is_bearish is False
        assert prediction.direction == Direction.BULLISH

    def test_risk_factors(self) -> None:
        """Test prediction with risk factors."""
        reasoning = ReasoningChain(
            steps=[
                ReasoningStep(
                    step_number=1,
                    description="分析",
                    conclusion="中性",
                    evidence=[],
                )
            ],
            final_conclusion="预测",
        )

        risk = RiskFactor(
            category="市场风险",
            description="市场波动风险",
            severity="medium",
            mitigation="设置止损",
        )

        prediction = PredictionResult(
            prediction_id="pred_002",
            stock_code="601398.SH",
            stock_name="工商银行",
            direction=Direction.NEUTRAL,
            probability=0.55,
            target_range="5.2-5.4",
            time_horizon=TimeHorizon.SHORT,
            confidence=0.5,
            reasoning_chain=reasoning,
            risk_factors=[risk],
        )
        assert len(prediction.risk_factors) == 1
        assert prediction.risk_factors[0].severity == "medium"
