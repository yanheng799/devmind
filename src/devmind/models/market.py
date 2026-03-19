"""Market data models."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StockPrice(BaseModel):
    """Stock price data.

    Attributes:
        stock_code: Stock code (e.g., "600519.SH")
        date: Price date
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        amount: Trading amount
        change_pct: Daily change percentage
        turnover_rate: Turnover rate
        pe_ratio: P/E ratio
        pb_ratio: P/B ratio
        total_market_cap: Total market capitalization
        circulating_market_cap: Circulating market capitalization
    """

    stock_code: str = Field(..., description="Stock code")
    date: datetime = Field(..., description="Price date")
    open: Decimal = Field(..., ge=0, description="Opening price")
    high: Decimal = Field(..., ge=0, description="Highest price")
    low: Decimal = Field(..., ge=0, description="Lowest price")
    close: Decimal = Field(..., ge=0, description="Closing price")
    volume: int | None = Field(default=None, ge=0, description="Trading volume")
    amount: Decimal | None = Field(default=None, ge=0, description="Trading amount")
    change_pct: float | None = Field(default=None, description="Daily change %")
    turnover_rate: float | None = Field(default=None, ge=0, description="Turnover rate")
    pe_ratio: float | None = Field(default=None, description="P/E ratio")
    pb_ratio: float | None = Field(default=None, description="P/B ratio")
    total_market_cap: float | None = Field(default=None, ge=0, description="Total market cap")
    circulating_market_cap: float | None = Field(
        default=None,
        ge=0,
        description="Circulating market cap",
    )

    @field_validator("stock_code")
    @classmethod
    def validate_stock_code(cls, value: str) -> str:
        """Validate stock code format."""
        # Format: 6-digit code + .SH or .SZ
        parts = value.split(".")
        if len(parts) == 2:
            code, exchange = parts
            if len(code) != 6 or not code.isdigit():
                raise ValueError(f"Invalid stock code format: {value}")
            if exchange.upper() not in ("SH", "SZ"):
                raise ValueError(f"Invalid exchange: {exchange}")
        elif len(parts) == 1:
            if len(value) != 6 or not value.isdigit():
                raise ValueError(f"Invalid stock code format: {value}")
        else:
            raise ValueError(f"Invalid stock code format: {value}")
        return value

    @property
    def is_trading_day(self) -> bool:
        """Check if this is a trading day (has volume)."""
        return self.volume is not None and self.volume > 0


class StockInfo(BaseModel):
    """Basic stock information.

    Attributes:
        stock_code: Stock code
        stock_name: Stock name
        industry: Industry sector
        sector: Economic sector
        list_date: Listing date
        market: Main board, SME board, ChiNext, STAR
        status: Active, suspended, delisted
    """

    stock_code: str = Field(..., description="Stock code")
    stock_name: str = Field(..., description="Stock name")
    industry: str | None = Field(default=None, description="Industry sector")
    sector: str | None = Field(default=None, description="Economic sector")
    list_date: datetime | None = Field(default=None, description="Listing date")
    market: str | None = Field(default=None, description="Market type")
    status: str = Field(default="active", description="Stock status")


class FinancialReport(BaseModel):
    """Financial report data.

    Attributes:
        stock_code: Stock code
        report_date: Report date (quarter end)
        report_type: Q1, Q2, Q3, or annual
        revenue: Total revenue
        revenue_growth: Revenue growth rate
        net_profit: Net profit
        net_profit_growth: Net profit growth rate
        gross_margin: Gross profit margin
        net_margin: Net profit margin
        roe: Return on equity
        roa: Return on assets
        debt_ratio: Debt to asset ratio
        eps: Earnings per share
        bps: Book value per share
        cash_flow: Operating cash flow
    """

    stock_code: str = Field(..., description="Stock code")
    report_date: datetime = Field(..., description="Report period end date")
    report_type: str = Field(..., description="Q1/Q2/Q3/annual")
    revenue: Decimal | None = Field(default=None, ge=0, description="Total revenue")
    revenue_growth: float | None = Field(default=None, description="Revenue growth %")
    net_profit: Decimal | None = Field(default=None, description="Net profit")
    net_profit_growth: float | None = Field(default=None, description="Net profit growth %")
    gross_margin: float | None = Field(default=None, description="Gross margin %")
    net_margin: float | None = Field(default=None, description="Net margin %")
    roe: float | None = Field(default=None, description="Return on equity %")
    roa: float | None = Field(default=None, description="Return on assets %")
    debt_ratio: float | None = Field(default=None, ge=0, le=1, description="Debt ratio")
    eps: float | None = Field(default=None, description="Earnings per share")
    bps: float | None = Field(default=None, description="Book value per share")
    cash_flow: Decimal | None = Field(default=None, description="Operating cash flow")

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, value: str) -> str:
        """Validate report type."""
        valid_types = {"Q1", "Q2", "Q3", "annual", "Q1-Q3"}
        value_upper = value.upper()
        if value_upper not in valid_types:
            raise ValueError(f"Report type must be one of {valid_types}, got: {value}")
        return value_upper


class IndustryRelation(BaseModel):
    """Industry relationship data.

    Attributes:
        stock_code: Stock code
        industry: Industry name
        upstream_stocks: Upstream related stocks
        downstream_stocks: Downstream related stocks
        peer_stocks: Peer company stocks
        correlation_matrix: Correlation data
    """

    stock_code: str = Field(..., description="Stock code")
    industry: str = Field(..., description="Industry name")
    upstream_stocks: list[str] = Field(
        default_factory=list,
        description="Upstream related stocks",
    )
    downstream_stocks: list[str] = Field(
        default_factory=list,
        description="Downstream related stocks",
    )
    peer_stocks: list[str] = Field(default_factory=list, description="Peer stocks")
    correlation_matrix: dict[str, float] = Field(
        default_factory=dict,
        description="Correlation with other stocks",
    )


class MarketIndex(BaseModel):
    """Market index data.

    Attributes:
        index_code: Index code (e.g., "000001.SH" for Shanghai Composite)
        index_name: Index name
        date: Date
        open: Opening value
        high: Highest value
        low: Lowest value
        close: Closing value
        volume: Trading volume
        change_pct: Daily change percentage
    """

    index_code: str = Field(..., description="Index code")
    index_name: str = Field(..., description="Index name")
    date: datetime = Field(..., description="Date")
    open: float = Field(..., ge=0, description="Opening value")
    high: float = Field(..., ge=0, description="Highest value")
    low: float = Field(..., ge=0, description="Lowest value")
    close: float = Field(..., ge=0, description="Closing value")
    volume: int | None = Field(default=None, ge=0, description="Trading volume")
    change_pct: float | None = Field(default=None, description="Daily change %")
