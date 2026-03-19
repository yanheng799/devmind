"""Command-line interface for DEVMIND."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from devmind.agents.stock_agent import StockPredictionAgent
from devmind.config import get_settings
from devmind.models import NewsArticle, SourceType


def json_serialize(obj: object) -> object:
    """Custom JSON serializer for datetime and other types.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_predict(args: argparse.Namespace) -> int:
    """Handle predict command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        with StockPredictionAgent(use_mock=args.mock) as agent:
            if args.url:
                # Predict from URL
                result = agent.predict_from_url(args.url, args.stock)
                print(json.dumps(result, ensure_ascii=False, indent=2, default=json_serialize))

            elif args.article:
                # Predict from article text
                article = NewsArticle(
                    article_id=f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    title=args.title or "手动输入的新闻",
                    content=args.article,
                    source="manual",
                    source_type=SourceType.SECONDARY,
                    publish_time=datetime.now(),
                    url="manual://input",
                    related_stocks=[args.stock] if args.stock else [],
                )
                result = agent.predict_from_article(article, args.stock)
                print(json.dumps(result, ensure_ascii=False, indent=2, default=json_serialize))

            else:
                # Fetch and predict latest news
                results = agent.fetch_and_predict_latest(limit=args.limit)
                for i, result in enumerate(results):
                    print(f"\n=== Prediction {i + 1} ===")
                    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_serialize))

        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def cmd_fetch(args: argparse.Namespace) -> int:
    """Handle fetch command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        with StockPredictionAgent(use_mock=args.mock) as agent:
            articles = agent.news_collector.fetch_latest_news(limit=args.limit)

            print(f"Fetched {len(articles)} articles:\n")
            for article in articles:
                print(f"ID: {article.article_id}")
                print(f"Title: {article.title}")
                print(f"Source: {article.source}")
                print(f"URL: {article.url}")
                print(f"Related stocks: {', '.join(article.related_stocks)}")
                print(f"Content: {article.content[:100]}...")
                print("-" * 60)

        return 0

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return 1


def cmd_query(args: argparse.Namespace) -> int:
    """Handle query command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        with StockPredictionAgent(use_mock=args.mock) as agent:
            result = agent.query_stock(args.stock)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        return 0

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return 1


def cmd_history(args: argparse.Namespace) -> int:
    """Handle history command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        with StockPredictionAgent(use_mock=True) as agent:
            result = agent.get_prediction_history(
                stock_code=args.stock,
                limit=args.limit,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        return 0

    except Exception as e:
        logger.error(f"History query failed: {e}")
        return 1


def cmd_accuracy(args: argparse.Namespace) -> int:
    """Handle accuracy command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        with StockPredictionAgent(use_mock=True) as agent:
            result = agent.get_accuracy_stats()
            print(json.dumps(result, ensure_ascii=False, indent=2))

        return 0

    except Exception as e:
        logger.error(f"Accuracy query failed: {e}")
        return 1


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="DEVMIND - News-based Stock Price Prediction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch and predict from latest news
  devmind predict

  # Predict from article URL
  devmind predict --url https://finance.eastmoney.com/news/12345.html

  # Predict with manual input
  devmind predict --article "央行宣布降准" --title "货币政策新闻" --stock 601398.SH

  # Query stock information
  devmind query 600519.SH

  # Get prediction history
  devmind history 600519.SH

  # Get accuracy statistics
  devmind accuracy
        """,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock collectors and models (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Make stock price prediction from news",
    )
    predict_parser.add_argument(
        "--url",
        help="Article URL to fetch and analyze",
    )
    predict_parser.add_argument(
        "--article",
        help="Article text content (manual input)",
    )
    predict_parser.add_argument(
        "--title",
        help="Article title (for manual input)",
    )
    predict_parser.add_argument(
        "--stock",
        help="Target stock code (e.g., 600519.SH)",
    )
    predict_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of latest articles to fetch (default: 5)",
    )

    # Fetch command
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch latest news articles",
    )
    fetch_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of articles to fetch (default: 10)",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query stock information and price",
    )
    query_parser.add_argument(
        "stock",
        help="Stock code to query",
    )

    # History command
    history_parser = subparsers.add_parser(
        "history",
        help="Get prediction history",
    )
    history_parser.add_argument(
        "stock",
        nargs="?",
        help="Filter by stock code",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of predictions (default: 20)",
    )

    # Accuracy command
    accuracy_parser = subparsers.add_parser(
        "accuracy",
        help="Get prediction accuracy statistics",
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0

    # Validate settings
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure DEVMIND_LLM_API_KEY is set")
        return 1

    # Dispatch command
    if args.command == "predict":
        return cmd_predict(args)
    elif args.command == "fetch":
        return cmd_fetch(args)
    elif args.command == "query":
        return cmd_query(args)
    elif args.command == "history":
        return cmd_history(args)
    elif args.command == "accuracy":
        return cmd_accuracy(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
