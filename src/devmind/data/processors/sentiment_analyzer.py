"""Sentiment analysis for news articles."""

import uuid
from typing import Any

from openai import OpenAI

from devmind.config import get_settings
from devmind.models import NewsArticle, Sentiment, SentimentAnalysis


class SentimentAnalyzer:
    """Analyze sentiment of news articles using LLM.

    Determines:
    - Overall sentiment (POSITIVE, NEGATIVE, NEUTRAL)
    - Sentiment score (-1 to 1)
    - Confidence level
    - Aspect-based sentiment (optional)
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the sentiment analyzer.

        Args:
            api_key: LLM API key (uses settings if None)
        """
        settings = get_settings()
        llm_config = settings.get_llm_config()

        if api_key:
            llm_config["api_key"] = api_key

        self.client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config.get("base_url"),
        )
        self.model = llm_config["model"]
        self.temperature = llm_config.get("temperature", 0.3)

    def analyze(self, article: NewsArticle) -> SentimentAnalysis:
        """Analyze sentiment of an article.

        Args:
            article: News article

        Returns:
            SentimentAnalysis result
        """
        prompt = self._build_analysis_prompt(article)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content or "{}")

            return self._parse_result(result, article.article_id)

        except Exception as e:
            raise RuntimeError(f"Sentiment analysis failed: {e}") from e

    def _get_system_prompt(self) -> str:
        """Get system prompt for sentiment analysis."""
        return """你是一个专业的金融情感分析专家。你的任务是分析新闻文章的情感倾向，特别是对A股市场的影响。

分析维度：
1. sentiment: 情感分类（positive: 利好, negative: 利空, neutral: 中性）
2. score: 情感得分（-1到1之间，负值表示利空，正值表示利好，0表示中性）
3. confidence: 分析置信度（0-1之间的浮点数）
4. aspects: 各方面的情感分析（可选）

请基于文章内容对股市的影响进行客观分析。注意：
- 政策利好、业绩增长、行业景气等通常是positive
- 政策收紧、业绩下滑、风险事件等通常是negative
- 客观事实陈述、无明显倾向的是neutral"""

    def _build_analysis_prompt(self, article: NewsArticle) -> str:
        """Build analysis prompt for article.

        Args:
            article: News article

        Returns:
            Analysis prompt
        """
        return f"""请分析以下新闻文章的情感倾向：

标题：{article.title}

来源：{article.source}

相关股票：{", ".join(article.related_stocks) if article.related_stocks else "未指定"}

正文内容：
{article.content[:2000]}

请以JSON格式返回分析结果：
{{
  "sentiment": "positive",
  "score": 0.7,
  "confidence": 0.85,
  "aspects": {{
    "政策影响": "positive",
    "市场情绪": "positive"
  }}
}}

注意：
- sentiment必须是positive、negative或neutral之一
- score应该是-1到1之间的数值
- confidence应该是0到1之间的数值
- aspects是可选的，可以包含对不同方面的分析"""

    def _parse_result(self, result: dict[str, Any], article_id: str) -> SentimentAnalysis:
        """Parse analysis result from LLM.

        Args:
            result: Analysis result from LLM
            article_id: Source article ID

        Returns:
            SentimentAnalysis object
        """
        # Parse sentiment
        sentiment_str = result.get("sentiment", "neutral")
        try:
            sentiment = Sentiment(sentiment_str.lower())
        except ValueError:
            sentiment = Sentiment.NEUTRAL

        return SentimentAnalysis(
            article_id=article_id,
            sentiment=sentiment,
            score=float(result.get("score", 0.0)),
            confidence=float(result.get("confidence", 0.5)),
            aspects=result.get("aspects", {}),
        )


class MockSentimentAnalyzer(SentimentAnalyzer):
    """Mock sentiment analyzer for testing.

    Returns predefined sentiment without calling LLM.
    """

    def __init__(self) -> None:
        """Initialize mock analyzer."""
        pass

    def analyze(self, article: NewsArticle) -> SentimentAnalysis:
        """Analyze sentiment using keyword matching.

        Args:
            article: News article

        Returns:
            SentimentAnalysis result
        """
        import re

        content = article.title + " " + article.content
        content_lower = content.lower()

        # Positive keywords
        positive_keywords = [
            "增长", "上涨", "利好", "突破", "创新高", "盈利", "收益",
            "降准", "降息", "宽松", "刺激", "支持", "促进",
        ]

        # Negative keywords
        negative_keywords = [
            "下降", "下跌", "利空", "跌破", "创新低", "亏损", "下滑",
            "加息", "收紧", "遏制", "限制", "风险", "危机",
        ]

        # Count matches
        positive_count = sum(1 for kw in positive_keywords if kw in content)
        negative_count = sum(1 for kw in negative_keywords if kw in content)

        # Determine sentiment
        if positive_count > negative_count:
            sentiment = Sentiment.POSITIVE
            score = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = Sentiment.NEGATIVE
            score = max(-0.9, -0.5 - (negative_count - positive_count) * 0.1)
        else:
            sentiment = Sentiment.NEUTRAL
            score = 0.0

        # Extract aspects based on keywords
        aspects: dict[str, str] = {}
        if "业绩" in content or "财报" in content:
            aspects["业绩"] = sentiment.value
        if "政策" in content:
            aspects["政策"] = sentiment.value
        if "行业" in content:
            aspects["行业"] = sentiment.value

        # Confidence based on keyword count
        total_matches = positive_count + negative_count
        confidence = min(0.95, 0.5 + total_matches * 0.1)

        return SentimentAnalysis(
            article_id=article.article_id,
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            aspects=aspects,
        )
