"""EastMoney news collector implementation."""

import hashlib
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from devmind.data.collectors.base_collector import (
    BaseCollector,
    clean_text,
    extract_stock_codes,
    ParseError,
)
from devmind.models import NewsArticle, SourceType


class EastMoneyNewsCollector(BaseCollector):
    """Collector for EastMoney (东方财富网) news.

    EastMoney is a primary source for A-share financial news.
    """

    def __init__(
        self,
        base_url: str = "https://finance.eastmoney.com",
        **kwargs: Any,
    ) -> None:
        """Initialize the collector.

        Args:
            base_url: Base URL for EastMoney
            **kwargs: Additional arguments for BaseCollector
        """
        super().__init__(rate_limit_per_minute=30, **kwargs)
        self.base_url = base_url

    def fetch_news_list(
        self,
        page: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch list of news articles.

        Args:
            page: Page number (starts from 1)
            limit: Maximum articles to fetch

        Returns:
            List of article metadata dicts
        """
        url = f"{self.base_url}/a/cjkx_{page}.html"

        try:
            response = self.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            articles: list[dict[str, Any]] = []

            # EastMoney news list structure
            # Usually in div.list-item or similar
            for item in soup.select(".list-item, .news-item, article")[:limit]:
                title_elem = item.select_one("a.title, h3 a, .title a")
                if not title_elem:
                    continue

                title = clean_text(title_elem.get_text())
                article_url = title_elem.get("href", "")

                if not article_url or not title:
                    continue

                # Make URL absolute
                if not article_url.startswith("http"):
                    article_url = urljoin(self.base_url, article_url)

                # Extract publish time
                time_elem = item.select_one(".time, .date, .pub-time")
                publish_time = None
                if time_elem:
                    time_text = clean_text(time_elem.get_text())
                    publish_time = self._parse_time(time_text)

                # Extract summary
                summary_elem = item.select_one(".summary, .desc, p")
                summary = ""
                if summary_elem:
                    summary = clean_text(summary_elem.get_text())

                articles.append({
                    "title": title,
                    "url": article_url,
                    "publish_time": publish_time,
                    "summary": summary,
                    "source": "eastmoney",
                })

            return articles

        except Exception as e:
            raise ParseError(f"Failed to parse news list: {e}") from e

    def fetch_article(self, url: str) -> NewsArticle:
        """Fetch full article content.

        Args:
            url: Article URL

        Returns:
            NewsArticle object
        """
        response = self.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_elem = soup.select_one("h1, .title, .article-title")
        if not title_elem:
            raise ParseError(f"Could not find title in article: {url}")
        title = clean_text(title_elem.get_text())

        # Extract content
        content_elem = soup.select_one(".article-content, .Body, #Content, article, .content")
        if not content_elem:
            raise ParseError(f"Could not find content in article: {url}")

        # Remove unwanted elements
        for elem in content_elem.select("script, style, .ad, .advertisement"):
            elem.decompose()

        content = clean_text(content_elem.get_text())

        if not content:
            raise ParseError(f"Empty content in article: {url}")

        # Extract publish time
        publish_time = None
        time_elem = soup.select_one(".time, .date, .pub-time, .article-time")
        if time_elem:
            time_text = clean_text(time_elem.get_text())
            publish_time = self._parse_time(time_text)

        if publish_time is None:
            publish_time = datetime.now()

        # Generate article ID from URL
        article_id = self._generate_article_id(url)

        # Extract related stock codes
        related_stocks = extract_stock_codes(title + " " + content)

        return NewsArticle(
            article_id=article_id,
            title=title,
            content=content,
            source="eastmoney",
            source_type=SourceType.PRIMARY,
            publish_time=publish_time,
            url=url,
            related_stocks=related_stocks,
        )

    def _parse_time(self, time_text: str) -> datetime | None:
        """Parse time text to datetime.

        Args:
            time_text: Time string like "2024-03-19 14:30", "14:30", etc.

        Returns:
            datetime or None if parsing fails
        """
        from datetime import datetime

        time_text = time_text.strip()

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y年%m月%d日 %H:%M",
            "%Y年%m月%d日",
            "%H:%M",
            "%M分钟前",
        ]

        for fmt in formats:
            try:
                if fmt == "%M分钟前":
                    # Handle relative time
                    import re
                    match = re.match(r"(\d+)分钟前", time_text)
                    if match:
                        minutes = int(match.group(1))
                        return datetime.now().replace(second=0, microsecond=0) - __import__("datetime").timedelta(minutes=minutes)
                else:
                    return datetime.strptime(time_text, fmt)
            except ValueError:
                continue

        return None

    def _generate_article_id(self, url: str) -> str:
        """Generate unique article ID from URL.

        Args:
            url: Article URL

        Returns:
            Article ID
        """
        # Use hash of URL for uniqueness
        hash_obj = hashlib.md5(url.encode())
        return f"em_{hash_obj.hexdigest()[:16]}"

    def fetch_latest_news(self, limit: int = 50) -> list[NewsArticle]:
        """Fetch latest news articles.

        Args:
            limit: Maximum number of articles to fetch

        Returns:
            List of NewsArticle objects
        """
        articles: list[NewsArticle] = []
        page = 1

        while len(articles) < limit:
            news_list = self.fetch_news_list(page=page)

            if not news_list:
                break

            for item in news_list:
                if len(articles) >= limit:
                    break

                try:
                    article = self.fetch_article(item["url"])
                    articles.append(article)
                except ParseError as e:
                    # Skip articles that fail to parse
                    continue

            page += 1

            if page > 3:  # Limit to 3 pages for now
                break

        return articles


class MockNewsCollector(BaseCollector):
    """Mock news collector for testing.

    Returns predefined news articles without making actual HTTP requests.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock collector."""
        super().__init__(**kwargs)
        self._mock_articles = self._create_mock_articles()

    def _create_mock_articles(self) -> list[dict[str, Any]]:
        """Create mock article data.

        Returns:
            List of mock article dicts
        """
        return [
            {
                "title": "央行宣布下调存款准备金率0.5个百分点",
                "content": "中国人民银行宣布决定于近期下调金融机构存款准备金率0.5个百分点，"
                          "此次降准将释放长期资金约1万亿元，保持流动性合理充裕。",
                "source": "pbc",
                "url": "https://example.com/news/1",
                "related_stocks": ["601398.SH", "601939.SH"],
            },
            {
                "title": "贵州茅台发布2024年半年报，净利润增长15%",
                "content": "贵州茅台发布半年报，上半年实现营业收入约700亿元，"
                          "同比增长约18%；净利润约360亿元，同比增长约15%。",
                "source": "eastmoney",
                "url": "https://example.com/news/2",
                "related_stocks": ["600519.SH"],
            },
            {
                "title": "新能源汽车产销两旺，产业链公司受益",
                "content": "据中汽协数据，7月新能源汽车产销分别完成80万辆和78万辆，"
                          "同比分别增长30%和31%。比亚迪、宁德时代等产业链公司业绩向好。",
                "source": "eastmoney",
                "url": "https://example.com/news/3",
                "related_stocks": ["002594.SZ", "300750.SZ"],
            },
            {
                "title": "半导体板块持续调整，国产替代进程加速",
                "content": "受国际贸易摩擦影响，半导体板块近期持续调整。"
                          "但国产替代进程加速，中芯国际、北方华创等公司获得大额订单。",
                "source": "eastmoney",
                "url": "https://example.com/news/4",
                "related_stocks": ["688981.SH", "002371.SZ"],
            },
            {
                "title": "房地产行业政策持续放松，多地取消限购",
                "content": "近期多个城市取消房地产限购政策，降低首付比例和贷款利率。"
                          "万科、保利发展等房企销售出现回暖迹象。",
                "source": "eastmoney",
                "url": "https://example.com/news/5",
                "related_stocks": ["000002.SZ", "600048.SH"],
            },
        ]

    def fetch_latest_news(self, limit: int = 10) -> list[NewsArticle]:
        """Fetch mock news articles.

        Args:
            limit: Maximum number of articles to return

        Returns:
            List of NewsArticle objects
        """
        import uuid

        articles: list[NewsArticle] = []

        for i, item in enumerate(self._mock_articles[:limit]):
            article = NewsArticle(
                article_id=f"mock_{uuid.uuid4().hex[:12]}",
                title=item["title"],
                content=item["content"],
                source=item["source"],
                source_type=SourceType.PRIMARY,
                publish_time=datetime.now(),
                url=item["url"],
                related_stocks=item.get("related_stocks", []),
            )
            articles.append(article)

        return articles
