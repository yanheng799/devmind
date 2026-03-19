"""Agent tools for querying vector store and historical events."""

from typing import Any

from devmind.data.database.database import PredictionDatabase
from devmind.data.vectorstore.milvus_client import (
    EmbeddingModel,
    MockEmbeddingModel,
    MockVectorStore,
    MilvusVectorStore,
)


def query_historical_events(
    query_text: str,
    event_type: str | None = None,
    stock_code: str | None = None,
    top_k: int = 5,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Query historical events similar to the query.

    Args:
        query_text: Query text describing the event
        event_type: Filter by event type
        stock_code: Filter by stock code
        top_k: Number of results to return
        use_mock: Use mock store for testing

    Returns:
        Dict with similar events or error message
    """
    try:
        # Initialize store and model
        store = MockVectorStore() if use_mock else MilvusVectorStore()
        model = MockEmbeddingModel() if use_mock else EmbeddingModel()

        # Generate query embedding
        query_embedding = model.embed_single(query_text)

        # Search for similar events
        results = store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            event_type=event_type,
            stock_code=stock_code,
        )

        # Get full event details from database
        db = PredictionDatabase()
        events_with_outcomes = []

        for result in results:
            # Get historical event details
            hist_events = db.get_historical_events(
                stock_code=result.get("stock_code"),
                event_type=result.get("event_type"),
                limit=1,
            )

            if hist_events:
                event = hist_events[0]
                events_with_outcomes.append({
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "description": event["description"],
                    "stock_code": event["stock_code"],
                    "actual_direction": event["actual_direction"],
                    "actual_change_pct": event["actual_change_pct"],
                    "similarity_score": result["score"],
                })

        return {
            "success": True,
            "query": query_text,
            "count": len(events_with_outcomes),
            "events": events_with_outcomes,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def store_historical_event(
    event_id: str,
    description: str,
    event_type: str,
    stock_code: str,
    actual_direction: str,
    actual_change_pct: float,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Store a historical event in vector store and database.

    Args:
        event_id: Event ID
        description: Event description
        event_type: Event type
        stock_code: Related stock code
        actual_direction: Actual price direction
        actual_change_pct: Actual price change percentage
        use_mock: Use mock store for testing

    Returns:
        Dict with result or error message
    """
    try:
        from datetime import datetime
        import uuid

        # Initialize store and model
        store = MockVectorStore() if use_mock else MilvusVectorStore()
        model = MockEmbeddingModel() if use_mock else EmbeddingModel()

        # Generate embedding
        embedding = model.embed_single(description)

        # Generate vector ID
        vector_id = f"vec_{uuid.uuid4().hex[:12]}"

        # Store in vector store
        store.insert_event(
            vector_id=vector_id,
            event_id=event_id,
            embedding=embedding,
            event_type=event_type,
            description=description,
            stock_code=stock_code,
        )

        # Store in database
        db = PredictionDatabase()
        db.insert_historical_event({
            "event_id": event_id,
            "event_type": event_type,
            "description": description,
            "entities": [stock_code],
            "magnitude": "medium",  # Default
            "timeframe": "short",  # Default
            "stock_code": stock_code,
            "actual_direction": actual_direction,
            "actual_change_pct": actual_change_pct,
            "event_date": datetime.now(),
            "vector_id": vector_id,
        })

        return {
            "success": True,
            "event_id": event_id,
            "vector_id": vector_id,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def find_similar_stocks(
    stock_code: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Find stocks with similar price movements.

    Args:
        stock_code: Reference stock code
        top_k: Number of similar stocks to return

    Returns:
        Dict with similar stocks or error message
    """
    try:
        db = PredictionDatabase()

        # Get price history for reference stock
        prices = db.get_stock_prices(stock_code, limit=30)
        if not prices:
            return {
                "success": False,
                "error": f"No price data found for {stock_code}",
            }

        # Calculate recent trend
        recent_changes = [p.get("change_pct", 0) for p in prices[:10]]
        avg_change = sum(recent_changes) / len(recent_changes) if recent_changes else 0

        # Find stocks with similar trends
        # This is a simplified version - production would use correlation analysis
        similar_stocks = []

        # Mock similar stocks based on industry
        industry_peers = {
            "600519.SH": ["000858.SZ", "000568.SZ", "600809.SH"],
            "601398.SH": ["601939.SH", "601288.SH", "000001.SZ"],
            "000002.SZ": ["600048.SH", "001979.SZ"],
        }

        peers = industry_peers.get(stock_code, [])

        for peer in peers[:top_k]:
            peer_prices = db.get_stock_prices(peer, limit=30)
            if peer_prices:
                peer_changes = [p.get("change_pct", 0) for p in peer_prices[:10]]
                peer_avg = sum(peer_changes) / len(peer_changes) if peer_changes else 0

                # Calculate similarity
                similarity = 1.0 - abs(avg_change - peer_avg) / 10.0
                similarity = max(0, min(1, similarity))

                similar_stocks.append({
                    "stock_code": peer,
                    "similarity": similarity,
                    "avg_change_pct": peer_avg,
                })

        similar_stocks.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "success": True,
            "stock_code": stock_code,
            "avg_change_pct": avg_change,
            "similar_stocks": similar_stocks,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def analyze_price_impact(
    stock_code: str,
    event_type: str,
) -> dict[str, Any]:
    """Analyze historical price impact for similar events.

    Args:
        stock_code: Stock code
        event_type: Event type to analyze

    Returns:
        Dict with impact analysis or error message
    """
    try:
        db = PredictionDatabase()

        # Get historical events of same type
        hist_events = db.get_historical_events(
            stock_code=stock_code,
            event_type=event_type,
            limit=50,
        )

        if not hist_events:
            return {
                "success": False,
                "error": f"No historical events found for {stock_code} and {event_type}",
            }

        # Analyze outcomes
        bullish_count = sum(1 for e in hist_events if e["actual_direction"] == "bullish")
        bearish_count = sum(1 for e in hist_events if e["actual_direction"] == "bearish")
        neutral_count = sum(1 for e in hist_events if e["actual_direction"] == "neutral")

        total = len(hist_events)

        # Calculate average change
        avg_change = sum(e["actual_change_pct"] for e in hist_events) / total

        # Calculate standard deviation
        import statistics
        changes = [e["actual_change_pct"] for e in hist_events]
        std_dev = statistics.stdev(changes) if len(changes) > 1 else 0

        # Calculate confidence interval
        margin_of_error = 1.96 * std_dev / (total**0.5)  # 95% confidence

        return {
            "success": True,
            "stock_code": stock_code,
            "event_type": event_type,
            "total_events": total,
            "direction_distribution": {
                "bullish": bullish_count,
                "bearish": bearish_count,
                "neutral": neutral_count,
            },
            "probability_bullish": bullish_count / total,
            "probability_bearish": bearish_count / total,
            "avg_change_pct": avg_change,
            "std_dev": std_dev,
            "confidence_interval": {
                "lower": avg_change - margin_of_error,
                "upper": avg_change + margin_of_error,
            },
            "recommendation": (
                "bullish" if bullish_count > total * 0.5 else
                "bearish" if bearish_count > total * 0.5 else
                "neutral"
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_recent_predictions(
    stock_code: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Get recent predictions.

    Args:
        stock_code: Filter by stock code
        limit: Maximum number of predictions

    Returns:
        Dict with recent predictions or error message
    """
    try:
        db = PredictionDatabase()

        if stock_code:
            predictions = db.get_predictions_by_stock(stock_code, limit=limit)
        else:
            # Get all recent predictions
            # This would require a new method in database
            predictions = []

        return {
            "success": True,
            "count": len(predictions),
            "predictions": [
                {
                    "prediction_id": p["prediction_id"],
                    "stock_code": p["stock_code"],
                    "stock_name": p["stock_name"],
                    "direction": p["direction"],
                    "probability": p["probability"],
                    "target_range": p["target_range"],
                    "confidence": p["confidence"],
                    "created_at": p["created_at"],
                    "status": p["status"],
                }
                for p in predictions
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
