"""Unit tests for data collectors."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from devmind.data.collectors.base_collector import (
    BaseCollector,
    extract_stock_codes,
    clean_text,
    RateLimitError,
)
from devmind.data.collectors.news_collector import (
    EastMoneyNewsCollector,
    MockNewsCollector,
)
from devmind.data.collectors.market_collector import (
    MockMarketCollector,
    AkshareMarketCollector,
)
from devmind.models import SourceType


class TestBaseCollector:
    """Tests for BaseCollector."""

    def test_rate_limit_check(self) -> None:
        """Test rate limiting."""
        collector = BaseCollector(rate_limit_per_minute=5)

        # Should allow first few requests
        for _ in range(5):
            collector._check_rate_limit()

        # Should raise on next request
        with pytest.raises(RateLimitError):
            collector._check_rate_limit()

    def test_burst_tokens_refill(self) -> None:
        """Test burst token refill."""
        import time

        collector = BaseCollector(rate_limit_per_minute=10)

        # Use burst tokens
        for _ in range(5):
            collector._check_rate_limit()

        # Wait for refill
        time.sleep(2)

        # Should have more tokens now
        collector._check_rate_limit()


class TestCleanText:
    """Tests for clean_text utility."""

    def test_remove_extra_whitespace(self) -> None:
        """Test removing extra whitespace."""
        text = "Hello    world\n\n  test"
        result = clean_text(text)
        assert result == "Hello world test"

    def test_remove_control_chars(self) -> None:
        """Test removing control characters."""
        text = "Hello\x00world\x1ftest"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x1f" not in result


class TestExtractStockCodes:
    """Tests for extract_stock_codes utility."""

    def test_extract_with_exchange(self) -> None:
        """Test extracting codes with exchange suffix."""
        text = "关注600519.SH和000858.SZ的表现"
        codes = extract_stock_codes(text)
        assert "600519.SH" in codes
        assert "000858.SZ" in codes

    def test_extract_without_exchange(self) -> None:
        """Test extracting codes without exchange suffix."""
        text = "600519和000858的股价"
        codes = extract_stock_codes(text)
        assert "600519.SH" in codes
        assert "000858.SZ" in codes

    def test_shanghai_codes(self) -> None:
        """Test Shanghai stock codes."""
        text = "关注600000和601398"
        codes = extract_stock_codes(text)
        assert "600000.SH" in codes
        assert "601398.SH" in codes

    def test_shenzhen_codes(self) -> None:
        """Test Shenzhen stock codes."""
        text = "关注000001和300750"
        codes = extract_stock_codes(text)
        assert "000001.SZ" in codes
        assert "300750.SZ" in codes

    def test_star_market_codes(self) -> None:
        """Test STAR market codes."""
        text = "688981是科创板股票"
        codes = extract_stock_codes(text)
        assert "688981.SH" in codes


class TestMockNewsCollector:
    """Tests for MockNewsCollector."""

    def test_fetch_latest_news(self) -> None:
        """Test fetching mock news."""
        collector = MockNewsCollector()
        articles = collector.fetch_latest_news(limit=3)

        assert len(articles) <= 3
        assert all(a.article_id.startswith("mock_") for a in articles)
        assert all(a.source in ["eastmoney", "pbc"] for a in articles)

    def test_article_has_required_fields(self) -> None:
        """Test articles have all required fields."""
        collector = MockNewsCollector()
        articles = collector.fetch_latest_news(limit=1)

        if articles:
            article = articles[0]
            assert article.title
            assert article.content
            assert article.url
            assert article.publish_time
            assert article.source_type == SourceType.PRIMARY


class TestMockMarketCollector:
    """Tests for MockMarketCollector."""

    def test_get_stock_info(self) -> None:
        """Test getting stock info."""
        collector = MockMarketCollector()
        info = collector.get_stock_info("600519.SH")

        assert info.stock_code == "600519.SH"
        assert info.stock_name == "贵州茅台"
        assert info.industry == "白酒"
        assert info.status == "active"

    def test_get_stock_price(self) -> None:
        """Test getting stock price."""
        collector = MockMarketCollector()
        price = collector.get_stock_price("600519.SH")

        assert price is not None
        assert price.stock_code == "600519.SH"
        assert price.close > 0
        assert price.volume is not None

    def test_get_stock_prices(self) -> None:
        """Test getting historical prices."""
        from datetime import timedelta

        collector = MockMarketCollector()
        start_date = datetime.now() - timedelta(days=7)
        prices = collector.get_stock_prices("600519.SH", start_date)

        assert len(prices) > 0
        assert all(p.stock_code == "600519.SH" for p in prices)

    def test_unknown_stock(self) -> None:
        """Test getting info for unknown stock."""
        collector = MockMarketCollector()
        info = collector.get_stock_info("999999.SH")

        assert info.stock_code == "999999.SH"
        assert info.stock_name == "测试股票"
        assert info.status == "active"

    def test_price_for_unknown_stock(self) -> None:
        """Test getting price for unknown stock."""
        collector = MockMarketCollector()
        price = collector.get_stock_price("999999.SH")

        assert price is None
