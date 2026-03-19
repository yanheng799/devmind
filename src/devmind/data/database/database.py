"""Database layer for DEVMIND using SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from devmind.config import get_settings


class PredictionDatabase:
    """SQLite database for storing predictions and historical data.

    Provides methods for storing and retrieving:
    - News articles
    - Extracted events
    - Predictions and outcomes
    - Historical events for backtesting
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database.

        Args:
            db_path: Path to SQLite database file. If None, uses settings.
        """
        if db_path is None:
            db_path = get_settings().db_path

        self.db_path = db_path
        self._connection: sqlite3.Connection | None = None
        self._init_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "PredictionDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Context manager exit."""
        self.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self.connection
        cursor = conn.cursor()

        # News articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                article_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                publish_time TIMESTAMP NOT NULL,
                url TEXT NOT NULL,
                related_stocks TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Extracted events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_events (
                event_id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entities TEXT NOT NULL,
                magnitude TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                transmission_chain TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_evidence TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles(article_id)
            )
        """)

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                probability REAL NOT NULL,
                target_range TEXT NOT NULL,
                time_horizon TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning_chain TEXT NOT NULL,
                risk_factors TEXT,
                similar_events TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Prediction outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                outcome_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                actual_direction TEXT NOT NULL,
                actual_change_pct REAL NOT NULL,
                outcome_date TIMESTAMP NOT NULL,
                is_correct BOOLEAN NOT NULL,
                magnitude_error REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            )
        """)

        # Historical events table (for similarity search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                entities TEXT NOT NULL,
                magnitude TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                actual_direction TEXT NOT NULL,
                actual_change_pct REAL NOT NULL,
                event_date TIMESTAMP NOT NULL,
                vector_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Stock prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                date TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER,
                amount REAL,
                change_pct REAL,
                turnover_rate REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                total_market_cap REAL,
                circulating_market_cap REAL,
                UNIQUE(stock_code, date)
            )
        """)

        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_articles_publish_time
            ON news_articles(publish_time DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_articles_source
            ON news_articles(source)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extracted_events_article_id
            ON extracted_events(article_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extracted_events_event_type
            ON extracted_events(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_stock_code
            ON predictions(stock_code)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions(created_at DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_status
            ON predictions(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_prices_stock_date
            ON stock_prices(stock_code, date DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_events_stock_code
            ON historical_events(stock_code)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_events_event_type
            ON historical_events(event_type)
        """)

        conn.commit()

    # News article operations
    def insert_news_article(self, article: dict[str, Any]) -> str:
        """Insert a news article.

        Args:
            article: Article data dict

        Returns:
            The article_id
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO news_articles
            (article_id, title, content, source, source_type, publish_time, url, related_stocks, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article["article_id"],
            article["title"],
            article["content"],
            article["source"],
            article["source_type"],
            article["publish_time"],
            article["url"],
            json.dumps(article.get("related_stocks", [])),
            json.dumps(article.get("metadata", {})),
        ))
        self.connection.commit()
        return article["article_id"]

    def get_news_article(self, article_id: str) -> dict[str, Any] | None:
        """Get a news article by ID.

        Args:
            article_id: Article ID

        Returns:
            Article dict or None if not found
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM news_articles WHERE article_id = ?",
            (article_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    # Event operations
    def insert_event(self, event: dict[str, Any]) -> str:
        """Insert an extracted event.

        Args:
            event: Event data dict

        Returns:
            The event_id
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO extracted_events
            (event_id, article_id, event_type, entities, magnitude, timeframe,
             transmission_chain, confidence, raw_evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event["event_id"],
            event["article_id"],
            event["event_type"],
            json.dumps(event["entities"]),
            event["magnitude"],
            event["timeframe"],
            json.dumps(event["transmission_chain"]),
            event["confidence"],
            event["raw_evidence"],
        ))
        self.connection.commit()
        return event["event_id"]

    def get_events_by_article(self, article_id: str) -> list[dict[str, Any]]:
        """Get all events for an article.

        Args:
            article_id: Article ID

        Returns:
            List of event dicts
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM extracted_events WHERE article_id = ?",
            (article_id,),
        )
        rows = cursor.fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["entities"] = json.loads(event["entities"])
            event["transmission_chain"] = json.loads(event["transmission_chain"])
            events.append(event)
        return events

    # Prediction operations
    def insert_prediction(self, prediction: dict[str, Any]) -> str:
        """Insert a prediction.

        Args:
            prediction: Prediction data dict

        Returns:
            The prediction_id
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO predictions
            (prediction_id, stock_code, stock_name, direction, probability,
             target_range, time_horizon, confidence, reasoning_chain,
             risk_factors, similar_events, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction["prediction_id"],
            prediction["stock_code"],
            prediction["stock_name"],
            prediction["direction"],
            prediction["probability"],
            prediction["target_range"],
            prediction["time_horizon"],
            prediction["confidence"],
            json.dumps(prediction["reasoning_chain"]),
            json.dumps(prediction.get("risk_factors", [])),
            json.dumps(prediction.get("similar_events", [])),
            prediction.get("status", "pending"),
            prediction.get("created_at", datetime.now()),
        ))
        self.connection.commit()
        return prediction["prediction_id"]

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        """Get a prediction by ID.

        Args:
            prediction_id: Prediction ID

        Returns:
            Prediction dict or None if not found
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        prediction = dict(row)
        prediction["reasoning_chain"] = json.loads(prediction["reasoning_chain"])
        prediction["risk_factors"] = json.loads(prediction["risk_factors"])
        prediction["similar_events"] = json.loads(prediction["similar_events"])
        return prediction

    def get_predictions_by_stock(
        self,
        stock_code: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get predictions for a stock.

        Args:
            stock_code: Stock code
            limit: Maximum number of predictions

        Returns:
            List of prediction dicts
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE stock_code = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (stock_code, limit))
        rows = cursor.fetchall()
        predictions = []
        for row in rows:
            pred = dict(row)
            pred["reasoning_chain"] = json.loads(pred["reasoning_chain"])
            pred["risk_factors"] = json.loads(pred["risk_factors"])
            pred["similar_events"] = json.loads(pred["similar_events"])
            predictions.append(pred)
        return predictions

    def get_pending_predictions(self) -> list[dict[str, Any]]:
        """Get all pending predictions.

        Returns:
            List of prediction dicts
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE status = 'pending'
            ORDER BY created_at ASC
        """)
        rows = cursor.fetchall()
        predictions = []
        for row in rows:
            pred = dict(row)
            pred["reasoning_chain"] = json.loads(pred["reasoning_chain"])
            pred["risk_factors"] = json.loads(pred["risk_factors"])
            pred["similar_events"] = json.loads(pred["similar_events"])
            predictions.append(pred)
        return predictions

    def update_prediction_status(
        self,
        prediction_id: str,
        status: str,
    ) -> None:
        """Update prediction status.

        Args:
            prediction_id: Prediction ID
            status: New status (confirmed/refuted/pending)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE predictions
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE prediction_id = ?
        """, (status, prediction_id))
        self.connection.commit()

    # Outcome operations
    def insert_outcome(self, outcome: dict[str, Any]) -> str:
        """Insert a prediction outcome.

        Args:
            outcome: Outcome data dict

        Returns:
            The outcome_id
        """
        import uuid

        outcome_id = outcome.get("outcome_id", str(uuid.uuid4()))
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO prediction_outcomes
            (outcome_id, prediction_id, actual_direction, actual_change_pct,
             outcome_date, is_correct, magnitude_error, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome_id,
            outcome["prediction_id"],
            outcome["actual_direction"],
            outcome["actual_change_pct"],
            outcome["outcome_date"],
            outcome["is_correct"],
            outcome.get("magnitude_error"),
            outcome.get("notes"),
        ))
        self.connection.commit()

        # Also update prediction status
        self.update_prediction_status(
            outcome["prediction_id"],
            "confirmed" if outcome["is_correct"] else "refuted",
        )
        return outcome_id

    # Historical event operations
    def insert_historical_event(self, event: dict[str, Any]) -> str:
        """Insert a historical event.

        Args:
            event: Historical event dict

        Returns:
            The event_id
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO historical_events
            (event_id, event_type, description, entities, magnitude, timeframe,
             stock_code, actual_direction, actual_change_pct, event_date, vector_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event["event_id"],
            event["event_type"],
            event["description"],
            json.dumps(event["entities"]),
            event["magnitude"],
            event["timeframe"],
            event["stock_code"],
            event["actual_direction"],
            event["actual_change_pct"],
            event["event_date"],
            event.get("vector_id"),
        ))
        self.connection.commit()
        return event["event_id"]

    def get_historical_events(
        self,
        stock_code: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get historical events.

        Args:
            stock_code: Filter by stock code
            event_type: Filter by event type
            limit: Maximum number of events

        Returns:
            List of historical event dicts
        """
        import json

        query = "SELECT * FROM historical_events WHERE 1=1"
        params: list[Any] = []

        if stock_code:
            query += " AND stock_code = ?"
            params.append(stock_code)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY event_date DESC LIMIT ?"
        params.append(limit)

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["entities"] = json.loads(event["entities"])
            events.append(event)
        return events

    # Stock price operations
    def insert_stock_price(self, price: dict[str, Any]) -> None:
        """Insert a stock price record.

        Args:
            price: Stock price dict
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO stock_prices
            (stock_code, date, open, high, low, close, volume, amount,
             change_pct, turnover_rate, pe_ratio, pb_ratio,
             total_market_cap, circulating_market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            price["stock_code"],
            price["date"],
            price["open"],
            price["high"],
            price["low"],
            price["close"],
            price.get("volume"),
            price.get("amount"),
            price.get("change_pct"),
            price.get("turnover_rate"),
            price.get("pe_ratio"),
            price.get("pb_ratio"),
            price.get("total_market_cap"),
            price.get("circulating_market_cap"),
        ))
        self.connection.commit()

    def get_stock_prices(
        self,
        stock_code: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get stock prices.

        Args:
            stock_code: Stock code
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records

        Returns:
            List of price dicts
        """
        query = "SELECT * FROM stock_prices WHERE stock_code = ?"
        params: list[Any] = [stock_code]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_price(self, stock_code: str) -> dict[str, Any] | None:
        """Get latest stock price.

        Args:
            stock_code: Stock code

        Returns:
            Latest price dict or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM stock_prices
            WHERE stock_code = ?
            ORDER BY date DESC
            LIMIT 1
        """, (stock_code,))
        row = cursor.fetchone()
        return dict(row) if row else None
