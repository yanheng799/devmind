"""Market data collector using AKShare."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import akshare as ak

from devmind.models import StockInfo, StockPrice


class MarketCollectorError(Exception):
    """Exception raised by market collector."""

    pass


class AkshareMarketCollector:
    """Market data collector using AKShare.

    AKShare is a Python library for fetching Chinese financial data.
    """

    def __init__(self) -> None:
        """Initialize the collector."""
        self._cache: dict[str, Any] = {}

    def get_stock_info(self, stock_code: str) -> StockInfo:
        """Get basic stock information.

        Args:
            stock_code: Stock code (e.g., "600519.SH" or "600519")

        Returns:
            StockInfo object
        """
        # AKShare uses code without exchange suffix
        code = stock_code.split(".")[0]

        try:
            # Get stock info
            info = ak.stock_individual_info_em(symbol=code)

            # Parse response
            stock_name = info.get("股票简称", "")
            industry = info.get("行业", "")
            list_date_str = info.get("上市日期", "")

            list_date = None
            if list_date_str:
                try:
                    list_date = datetime.strptime(list_date_str, "%Y-%m-%d")
                except ValueError:
                    pass

            # Determine market
            market = "unknown"
            if code.startswith("688"):
                market = "STAR"
            elif code.startswith("300"):
                market = "ChiNext"
            elif code.startswith("002", "003"):
                market = "SME"
            elif code.startswith("60"):
                market = "main"

            return StockInfo(
                stock_code=stock_code,
                stock_name=stock_name,
                industry=industry,
                list_date=list_date,
                market=market,
                status="active",
            )

        except Exception as e:
            raise MarketCollectorError(f"Failed to get stock info for {stock_code}: {e}") from e

    def get_stock_price(
        self,
        stock_code: str,
        date: datetime | None = None,
    ) -> StockPrice | None:
        """Get stock price for a specific date.

        Args:
            stock_code: Stock code (e.g., "600519.SH" or "600519")
            date: Target date, None for latest

        Returns:
            StockPrice object or None if data not found
        """
        # AKShare uses code without exchange suffix
        code = stock_code.split(".")[0]

        # Determine exchange for AKShare
        # Shanghai: 60xxxx, 688xxxx
        # Shenzhen: 000xxx, 001xxx, 002xxx, 003xxx, 300xxx, 301xxx
        if code.startswith(("60", "68")):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        try:
            if date is None:
                # Get latest daily data
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                if df.empty:
                    return None
                row = df.iloc[0]
            else:
                # Get historical data for date range
                end_date = date.strftime("%Y%m%d")
                start_date = (date - timedelta(days=10)).strftime("%Y%m%d")
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )
                if df.empty:
                    return None
                row = df.iloc[0]

            # Parse row
            return self._parse_stock_price_row(stock_code, row)

        except Exception as e:
            raise MarketCollectorError(f"Failed to get stock price for {stock_code}: {e}") from e

    def get_stock_prices(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[StockPrice]:
        """Get historical stock prices.

        Args:
            stock_code: Stock code
            start_date: Start date
            end_date: End date (None for latest)
            limit: Maximum number of records

        Returns:
            List of StockPrice objects
        """
        # AKShare uses code without exchange suffix
        code = stock_code.split(".")[0]

        # Determine exchange
        if code.startswith(("60", "68")):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d") if end_date else datetime.now().strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="qfq",
            )

            if df.empty:
                return []

            prices: list[StockPrice] = []
            for _, row in df.tail(limit).iterrows():
                price = self._parse_stock_price_row(stock_code, row)
                if price:
                    prices.append(price)

            return prices

        except Exception as e:
            raise MarketCollectorError(f"Failed to get stock prices for {stock_code}: {e}") from e

    def _parse_stock_price_row(self, stock_code: str, row: Any) -> StockPrice | None:
        """Parse a row from AKShare data.

        Args:
            stock_code: Stock code
            row: DataFrame row

        Returns:
            StockPrice object or None
        """
        try:
            # AKShare columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
            date_str = row.get("日期", "")
            if not date_str:
                return None

            date = datetime.strptime(date_str, "%Y-%m-%d")

            return StockPrice(
                stock_code=stock_code,
                date=date,
                open=Decimal(str(row.get("开盘", 0))),
                high=Decimal(str(row.get("最高", 0))),
                low=Decimal(str(row.get("最低", 0))),
                close=Decimal(str(row.get("收盘", 0))),
                volume=int(row.get("成交量", 0)) if row.get("成交量") else None,
                amount=Decimal(str(row.get("成交额", 0))) if row.get("成交额") else None,
                change_pct=float(row.get("涨跌幅", 0)) if row.get("涨跌幅") else None,
                turnover_rate=float(row.get("换手率", 0)) if row.get("换手率") else None,
            )

        except (ValueError, TypeError) as e:
            return None

    def get_market_index(
        self,
        index_code: str = "000001",
    ) -> dict[str, Any] | None:
        """Get market index data.

        Args:
            index_code: Index code (default: Shanghai Composite 000001)

        Returns:
            Index data dict or None
        """
        try:
            # Get index data
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")

            if df.empty:
                return None

            latest = df.iloc[-1]

            return {
                "index_code": index_code,
                "date": datetime.strptime(latest.name, "%Y-%m-%d"),
                "open": float(latest.get("open", 0)),
                "high": float(latest.get("high", 0)),
                "low": float(latest.get("low", 0)),
                "close": float(latest.get("close", 0)),
                "volume": int(latest.get("volume", 0)) if latest.get("volume") else None,
            }

        except Exception as e:
            raise MarketCollectorError(f"Failed to get index data: {e}") from e

    def search_stocks(self, keyword: str, limit: int = 20) -> list[StockInfo]:
        """Search stocks by keyword.

        Args:
            keyword: Search keyword
            limit: Maximum results

        Returns:
            List of StockInfo objects
        """
        try:
            # Get all stocks
            df = ak.stock_zh_a_spot_em()

            if df.empty:
                return []

            # Filter by keyword (name or code)
            mask = (
                df["名称"].str.contains(keyword, case=False, na=False) |
                df["代码"].str.contains(keyword, na=False)
            )
            filtered = df[mask].head(limit)

            stocks: list[StockInfo] = []
            for _, row in filtered.iterrows():
                code = row.get("代码", "")
                name = row.get("名称", "")

                # Add exchange suffix
                if code.startswith(("60", "68")):
                    stock_code = f"{code}.SH"
                else:
                    stock_code = f"{code}.SZ"

                stocks.append(StockInfo(
                    stock_code=stock_code,
                    stock_name=name,
                    status="active",
                ))

            return stocks

        except Exception as e:
            raise MarketCollectorError(f"Failed to search stocks: {e}") from e


class MockMarketCollector:
    """Mock market collector for testing."""

    def __init__(self) -> None:
        """Initialize mock collector with predefined data."""
        self._mock_data = {
            "600519.SH": {
                "stock_name": "贵州茅台",
                "industry": "白酒",
                "prices": [
                    {
                        "date": datetime.now() - timedelta(days=2),
                        "open": 1750.0,
                        "high": 1780.0,
                        "low": 1745.0,
                        "close": 1770.0,
                        "volume": 2000000,
                    },
                    {
                        "date": datetime.now() - timedelta(days=1),
                        "open": 1765.0,
                        "high": 1790.0,
                        "low": 1760.0,
                        "close": 1785.0,
                        "volume": 2500000,
                    },
                    {
                        "date": datetime.now(),
                        "open": 1780.0,
                        "high": 1810.0,
                        "low": 1775.0,
                        "close": 1800.0,
                        "volume": 3000000,
                        "change_pct": 0.84,
                    },
                ],
            },
            "601398.SH": {
                "stock_name": "工商银行",
                "industry": "银行",
                "prices": [
                    {
                        "date": datetime.now(),
                        "open": 5.2,
                        "high": 5.3,
                        "low": 5.18,
                        "close": 5.25,
                        "volume": 50000000,
                        "change_pct": 0.96,
                    },
                ],
            },
        }

    def get_stock_info(self, stock_code: str) -> StockInfo:
        """Get mock stock info."""
        if stock_code not in self._mock_data:
            return StockInfo(
                stock_code=stock_code,
                stock_name="测试股票",
                industry="测试行业",
                status="active",
            )

        data = self._mock_data[stock_code]
        return StockInfo(
            stock_code=stock_code,
            stock_name=data["stock_name"],
            industry=data["industry"],
            status="active",
        )

    def get_stock_price(
        self,
        stock_code: str,
        date: datetime | None = None,
    ) -> StockPrice | None:
        """Get mock stock price."""
        if stock_code not in self._mock_data:
            return None

        prices = self._mock_data[stock_code]["prices"]
        if date is None:
            price_data = prices[-1]
        else:
            for p in prices:
                if p["date"].date() == date.date():
                    price_data = p
                    break
            else:
                return None

        return StockPrice(
            stock_code=stock_code,
            date=price_data["date"],
            open=Decimal(str(price_data["open"])),
            high=Decimal(str(price_data["high"])),
            low=Decimal(str(price_data["low"])),
            close=Decimal(str(price_data["close"])),
            volume=price_data.get("volume"),
            change_pct=price_data.get("change_pct"),
        )

    def get_stock_prices(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[StockPrice]:
        """Get mock stock prices."""
        if stock_code not in self._mock_data:
            return []

        prices: list[StockPrice] = []
        for price_data in self._mock_data[stock_code]["prices"]:
            if price_data["date"] >= start_date:
                prices.append(StockPrice(
                    stock_code=stock_code,
                    date=price_data["date"],
                    open=Decimal(str(price_data["open"])),
                    high=Decimal(str(price_data["high"])),
                    low=Decimal(str(price_data["low"])),
                    close=Decimal(str(price_data["close"])),
                    volume=price_data.get("volume"),
                    change_pct=price_data.get("change_pct"),
                ))
        return prices
