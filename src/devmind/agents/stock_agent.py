"""Main stock prediction agent class."""

import logging
from typing import Any

from devmind.agents.graph.prediction_graph import run_prediction
from devmind.agents.tools import market_tools, vector_tools
from devmind.config import get_settings
from devmind.data.collectors.market_collector import MockMarketCollector
from devmind.data.collectors.news_collector import EastMoneyNewsCollector, MockNewsCollector
from devmind.data.database.database import PredictionDatabase
from devmind.data.processors.event_extractor import MockEventExtractor
from devmind.data.processors.sentiment_analyzer import MockSentimentAnalyzer
from devmind.data.vectorstore.milvus_client import MockEmbeddingModel, MockVectorStore
from devmind.models import NewsArticle

logger = logging.getLogger(__name__)


class StockPredictionAgent:
    """Main agent for stock price prediction based on news.

    Orchestrates:
    - News collection
    - Event extraction
    - Sentiment analysis
    - Historical similarity search
    - Prediction generation
    """

    def __init__(
        self,
        use_mock: bool = True,
    ) -> None:
        """Initialize the agent.

        Args:
            use_mock: Use mock collectors and models for testing
        """
        self.settings = get_settings()
        self.use_mock = use_mock

        # Initialize components
        self._init_components()

        # Initialize database
        self.db = PredictionDatabase()

        logger.info(f"StockPredictionAgent initialized (mock={use_mock})")

    def _init_components(self) -> None:
        """Initialize agent components."""
        # News collector
        if self.use_mock:
            self.news_collector = MockNewsCollector()
            self.market_collector = MockMarketCollector()
            self.event_extractor = MockEventExtractor()
            self.sentiment_analyzer = MockSentimentAnalyzer()
            self.vector_store = MockVectorStore()
            self.embedding_model = MockEmbeddingModel()
        else:
            self.news_collector = EastMoneyNewsCollector()
            self.market_collector = MockMarketCollector()  # Still use mock for AKShare
            self.event_extractor = MockEventExtractor()  # TODO: Implement real
            self.sentiment_analyzer = MockSentimentAnalyzer()  # TODO: Implement real
            self.vector_store = MockVectorStore()  # TODO: Implement real
            self.embedding_model = MockEmbeddingModel()  # TODO: Implement real

    def predict_from_article(
        self,
        article: NewsArticle,
        stock_code: str | None = None,
    ) -> dict[str, Any]:
        """Make prediction from a news article.

        Args:
            article: News article
            stock_code: Target stock code (optional, extracted from article)

        Returns:
            Dict with prediction result
        """
        logger.info(f"Processing article: {article.title[:50]}...")

        try:
            result = run_prediction(article, stock_code)
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def predict_from_url(
        self,
        url: str,
        stock_code: str | None = None,
    ) -> dict[str, Any]:
        """Make prediction from a news article URL.

        Args:
            url: Article URL
            stock_code: Target stock code (optional)

        Returns:
            Dict with prediction result
        """
        logger.info(f"Fetching article from URL: {url}")

        try:
            # Fetch article
            article = self.news_collector.fetch_article(url)

            # Make prediction
            return self.predict_from_article(article, stock_code)

        except Exception as e:
            logger.error(f"Failed to process URL: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def fetch_and_predict_latest(
        self,
        limit: int = 5,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch latest news and make predictions.

        Args:
            limit: Maximum number of articles to process
            source: Filter by source name

        Returns:
            List of prediction results
        """
        logger.info(f"Fetching latest news (limit={limit})")

        try:
            # Fetch latest news
            articles = self.news_collector.fetch_latest_news(limit=limit)

            results: list[dict[str, Any]] = []

            for article in articles:
                # Skip if no related stocks
                if not article.related_stocks:
                    logger.debug(f"Skipping article with no related stocks: {article.title}")
                    continue

                # Predict for each related stock
                for stock_code in article.related_stocks[:1]:  # Limit to first stock for now
                    result = self.predict_from_article(article, stock_code)
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to fetch and predict: {e}")
            return [{
                "success": False,
                "error": str(e),
            }]

    def query_stock(self, stock_code: str) -> dict[str, Any]:
        """Query stock information and price.

        Args:
            stock_code: Stock code

        Returns:
            Dict with stock data
        """
        logger.info(f"Querying stock: {stock_code}")

        try:
            # Get stock info
            info = self.market_collector.get_stock_info(stock_code)

            # Get latest price
            price = self.market_collector.get_stock_price(stock_code)

            return {
                "success": True,
                "info": {
                    "stock_code": info.stock_code,
                    "stock_name": info.stock_name,
                    "industry": info.industry,
                    "market": info.market,
                },
                "price": {
                    "date": price.date.isoformat() if price else None,
                    "open": float(price.open) if price else None,
                    "high": float(price.high) if price else None,
                    "low": float(price.low) if price else None,
                    "close": float(price.close) if price else None,
                    "change_pct": price.change_pct if price else None,
                } if price else None,
            }

        except Exception as e:
            logger.error(f"Failed to query stock: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_prediction_history(
        self,
        stock_code: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get prediction history.

        Args:
            stock_code: Filter by stock code
            limit: Maximum number of predictions

        Returns:
            Dict with prediction history
        """
        logger.info(f"Getting prediction history (stock={stock_code}, limit={limit})")

        try:
            if stock_code:
                predictions = self.db.get_predictions_by_stock(stock_code, limit=limit)
            else:
                # Get all recent predictions would require a new method
                predictions = []

            return {
                "success": True,
                "count": len(predictions),
                "predictions": predictions,
            }

        except Exception as e:
            logger.error(f"Failed to get prediction history: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_accuracy_stats(self) -> dict[str, Any]:
        """Get prediction accuracy statistics.

        Returns:
            Dict with accuracy statistics
        """
        logger.info("Getting accuracy statistics")

        try:
            # For now, return mock data
            # In production, this would analyze actual outcomes
            return {
                "success": True,
                "total_predictions": 0,
                "confirmed_predictions": 0,
                "accuracy": 0.0,
                "by_direction": {},
                "by_horizon": {},
            }

        except Exception as e:
            logger.error(f"Failed to get accuracy stats: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def close(self) -> None:
        """Close the agent and release resources."""
        if hasattr(self, "news_collector"):
            self.news_collector.close()
        self.db.close()

        logger.info("StockPredictionAgent closed")

    def __enter__(self) -> "StockPredictionAgent":
        """Context manager entry."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Context manager exit."""
        self.close()
