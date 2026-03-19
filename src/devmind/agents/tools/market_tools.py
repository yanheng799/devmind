"""Agent tools for querying market data."""

from datetime import datetime
from typing import Any

from devmind.data.collectors.market_collector import (
    AkshareMarketCollector,
    MockMarketCollector,
)
from devmind.data.database.database import PredictionDatabase
from devmind.models import StockPrice


def query_stock_price(
    stock_code: str,
    date: datetime | None = None,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Query stock price for a given stock and date.

    Args:
        stock_code: Stock code (e.g., "600519.SH")
        date: Target date (None for latest)
        use_mock: Use mock collector for testing

    Returns:
        Dict with price data or error message
    """
    try:
        collector = MockMarketCollector() if use_mock else AkshareMarketCollector()
        price = collector.get_stock_price(stock_code, date)

        if price is None:
            return {
                "success": False,
                "error": f"No price data found for {stock_code}",
            }

        return {
            "success": True,
            "stock_code": price.stock_code,
            "date": price.date.isoformat(),
            "open": float(price.open),
            "high": float(price.high),
            "low": float(price.low),
            "close": float(price.close),
            "volume": price.volume,
            "change_pct": price.change_pct,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def query_stock_info(
    stock_code: str,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Query stock information.

    Args:
        stock_code: Stock code (e.g., "600519.SH")
        use_mock: Use mock collector for testing

    Returns:
        Dict with stock info or error message
    """
    try:
        collector = MockMarketCollector() if use_mock else AkshareMarketCollector()
        info = collector.get_stock_info(stock_code)

        return {
            "success": True,
            "stock_code": info.stock_code,
            "stock_name": info.stock_name,
            "industry": info.industry,
            "sector": info.sector,
            "market": info.market,
            "status": info.status,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def query_price_history(
    stock_code: str,
    days: int = 30,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Query historical price data.

    Args:
        stock_code: Stock code
        days: Number of days of history
        use_mock: Use mock collector for testing

    Returns:
        Dict with historical prices or error message
    """
    try:
        from datetime import timedelta

        start_date = datetime.now() - timedelta(days=days)
        collector = MockMarketCollector() if use_mock else AkshareMarketCollector()
        prices = collector.get_stock_prices(stock_code, start_date)

        return {
            "success": True,
            "stock_code": stock_code,
            "prices": [
                {
                    "date": p.date.isoformat(),
                    "open": float(p.open),
                    "high": float(p.high),
                    "low": float(p.low),
                    "close": float(p.close),
                    "volume": p.volume,
                    "change_pct": p.change_pct,
                }
                for p in prices
            ],
            "count": len(prices),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def query_financial_report(
    stock_code: str,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Query latest financial report for a stock.

    Args:
        stock_code: Stock code
        use_mock: Use mock collector for testing

    Returns:
        Dict with financial data or error message
    """
    try:
        import akshare as ak

        # AKShare uses code without exchange suffix
        code = stock_code.split(".")[0]

        # Get financial indicators
        df = ak.stock_financial_analysis_indicator(symbol=code)

        if df.empty:
            return {
                "success": False,
                "error": f"No financial data found for {stock_code}",
            }

        # Get latest report
        latest = df.iloc[-1]

        return {
            "success": True,
            "stock_code": stock_code,
            "report_date": latest.get("日期"),
            "roe": float(latest.get("净资产收益率", 0)) if latest.get("净资产收益率") else None,
            "roa": float(latest.get("总资产净利率", 0)) if latest.get("总资产净利率") else None,
            "gross_margin": float(latest.get("销售毛利率", 0)) if latest.get("销售毛利率") else None,
            "net_margin": float(latest.get("销售净利率", 0)) if latest.get("销售净利率") else None,
            "debt_ratio": float(latest.get("资产负债率", 0)) if latest.get("资产负债率") else None,
            "eps": float(latest.get("每股收益", 0)) if latest.get("每股收益") else None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def query_industry_relation(
    stock_code: str,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Query industry relations for a stock.

    Args:
        stock_code: Stock code
        use_mock: Use mock collector for testing

    Returns:
        Dict with industry relation data or error message
    """
    try:
        # For now, return mock data
        # In production, this would query a database or API
        industry_relations = {
            "600519.SH": {
                "industry": "白酒",
                "peers": ["000858.SZ", "000568.SZ", "600809.SH"],
                "upstream": ["粮食种植"],
                "downstream": ["经销商", "零售终端"],
            },
            "601398.SH": {
                "industry": "银行",
                "peers": ["601939.SH", "601288.SH", "000001.SZ"],
                "upstream": [],
                "downstream": ["企业贷款", "个人贷款"],
            },
        }

        data = industry_relations.get(stock_code, {
            "industry": "未知",
            "peers": [],
            "upstream": [],
            "downstream": [],
        })

        return {
            "success": True,
            "stock_code": stock_code,
            **data,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_prediction_accuracy(
    stock_code: str | None = None,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Get prediction accuracy statistics.

    Args:
        stock_code: Filter by stock code
        event_type: Filter by event type

    Returns:
        Dict with accuracy statistics
    """
    try:
        db = PredictionDatabase()

        # Get predictions with outcomes
        # This would require joining predictions and outcomes tables
        # For now, return mock data
        return {
            "success": True,
            "total_predictions": 100,
            "correct_predictions": 58,
            "accuracy": 0.58,
            "by_direction": {
                "bullish": {"total": 40, "correct": 25, "accuracy": 0.625},
                "bearish": {"total": 35, "correct": 18, "accuracy": 0.514},
                "neutral": {"total": 25, "correct": 15, "accuracy": 0.6},
            },
            "by_horizon": {
                "immediate": {"total": 30, "correct": 14, "accuracy": 0.467},
                "short": {"total": 40, "correct": 24, "accuracy": 0.6},
                "medium": {"total": 30, "correct": 20, "accuracy": 0.667},
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
