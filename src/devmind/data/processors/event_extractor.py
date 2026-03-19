"""Event extraction from news articles using LLM."""

import json
import uuid
from typing import Any

from openai import OpenAI

from devmind.config import get_settings
from devmind.models import ExtractedEvent, EventType, NewsArticle, TimeHorizon


class EventExtractor:
    """Extract structured events from news articles using LLM.

    Uses few-shot prompting to extract:
    - Event type (monetary policy, earnings, etc.)
    - Related entities (companies, sectors)
    - Impact magnitude
    - Time horizon
    - Transmission chain (causal reasoning)
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the event extractor.

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
        self.temperature = llm_config.get("temperature", 0.3)  # Lower for extraction

    def extract_events(self, article: NewsArticle) -> list[ExtractedEvent]:
        """Extract events from a news article.

        Args:
            article: News article

        Returns:
            List of extracted events
        """
        prompt = self._build_extraction_prompt(article)

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
            events_data = result.get("events", [])

            events: list[ExtractedEvent] = []
            for event_data in events_data:
                try:
                    event = self._parse_event(event_data, article.article_id)
                    events.append(event)
                except Exception as e:
                    # Skip invalid events
                    continue

            return events

        except Exception as e:
            raise RuntimeError(f"Event extraction failed: {e}") from e

    def _get_system_prompt(self) -> str:
        """Get system prompt for event extraction."""
        return """你是一个专业的金融事件提取专家。你的任务是从新闻文章中提取可能影响A股股价的结构化事件。

提取的事件应包含：
1. event_type: 事件类型（monetary_policy, fiscal_policy, earnings, geopolitical, regulatory, industry, market_sentiment, technology, supply_chain, commodity, other）
2. entities: 相关实体（公司名称、股票代码、行业等）
3. magnitude: 影响程度（high, medium, low）
4. timeframe: 影响时间跨度（immediate: <1天, short: 1-5天, medium: 5-20天, long: >20天）
5. transmission_chain: 传导链条，解释事件如何影响股价的因果链条
6. confidence: 提取置信度（0-1之间的浮点数）
7. raw_evidence: 支持该事件提取的原文证据

请仔细分析文章内容，提取所有可能影响股价的事件。如果文章没有明确的事件，返回空数组。"""

    def _build_extraction_prompt(self, article: NewsArticle) -> str:
        """Build extraction prompt for article.

        Args:
            article: News article

        Returns:
            Extraction prompt
        """
        return f"""请从以下新闻文章中提取可能影响A股股价的事件：

标题：{article.title}

来源：{article.source}

发布时间：{article.publish_time}

相关股票：{", ".join(article.related_stocks) if article.related_stocks else "未指定"}

正文内容：
{article.content[:2000]}

请以JSON格式返回事件列表，格式如下：
{{
  "events": [
    {{
      "event_type": "monetary_policy",
      "entities": ["央行", "银行股"],
      "magnitude": "high",
      "timeframe": "short",
      "transmission_chain": ["央行降准", "银行资金成本降低", "银行利润增加", "银行股上涨"],
      "confidence": 0.9,
      "raw_evidence": "中国人民银行宣布下调存款准备金率0.5个百分点"
    }}
  ]
}}

注意：
- event_type必须是指定类型之一
- magnitude必须是high、medium、low之一
- timeframe必须是immediate、short、medium、long之一
- transmission_chain应该是完整的因果推理链
- confidence应该是0到1之间的数值
- raw_evidence必须是文章原文的摘录"""

    def _parse_event(self, event_data: dict[str, Any], article_id: str) -> ExtractedEvent:
        """Parse event data from LLM response.

        Args:
            event_data: Event data from LLM
            article_id: Source article ID

        Returns:
            ExtractedEvent object

        Raises:
            ValueError: If event data is invalid
        """
        # Parse event type
        event_type_str = event_data.get("event_type", "other")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.OTHER

        # Parse timeframe
        timeframe_str = event_data.get("timeframe", "short")
        try:
            timeframe = TimeHorizon(timeframe_str)
        except ValueError:
            timeframe = TimeHorizon.SHORT

        return ExtractedEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            article_id=article_id,
            event_type=event_type,
            entities=event_data.get("entities", []),
            magnitude=event_data.get("magnitude", "medium"),
            timeframe=timeframe,
            transmission_chain=event_data.get("transmission_chain", []),
            confidence=float(event_data.get("confidence", 0.5)),
            raw_evidence=event_data.get("raw_evidence", ""),
        )


class MockEventExtractor(EventExtractor):
    """Mock event extractor for testing.

    Returns predefined events without calling LLM.
    """

    def __init__(self) -> None:
        """Initialize mock extractor."""
        pass

    def extract_events(self, article: NewsArticle) -> list[ExtractedEvent]:
        """Extract mock events from article.

        Args:
            article: News article

        Returns:
            List of mock events
        """
        import uuid

        # Simple keyword-based mock extraction
        events: list[ExtractedEvent] = []
        content = article.title + " " + article.content

        # Check for monetary policy
        if any(kw in content for kw in ["降准", "降息", "MLF", "LPR", "存款准备金", "货币政策"]):
            events.append(ExtractedEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                article_id=article.article_id,
                event_type=EventType.MONETARY_POLICY,
                entities=["央行", "银行股"],
                magnitude="high",
                timeframe=TimeHorizon.SHORT,
                transmission_chain=["央行降准", "银行资金成本降低", "银行利润增加", "银行股上涨"],
                confidence=0.9,
                raw_evidence="货币政策相关新闻",
            ))

        # Check for earnings
        if any(kw in content for kw in ["财报", "业绩", "净利润", "营收", "季报", "年报"]):
            events.append(ExtractedEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                article_id=article.article_id,
                event_type=EventType.EARNINGS,
                entities=article.related_stocks if article.related_stocks else ["相关公司"],
                magnitude="medium",
                timeframe=TimeHorizon.IMMEDIATE,
                transmission_chain=["业绩发布", "市场反应", "股价变动"],
                confidence=0.8,
                raw_evidence="财报相关新闻",
            ))

        # Check for industry news
        if any(kw in content for kw in ["行业", "板块", "产业链"]):
            events.append(ExtractedEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                article_id=article.article_id,
                event_type=EventType.INDUSTRY,
                entities=article.related_stocks if article.related_stocks else ["相关行业"],
                magnitude="medium",
                timeframe=TimeHorizon.MEDIUM,
                transmission_chain=["行业变化", "公司业绩预期", "股价调整"],
                confidence=0.7,
                raw_evidence="行业相关新闻",
            ))

        # If no specific events, create a generic one
        if not events:
            events.append(ExtractedEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                article_id=article.article_id,
                event_type=EventType.MARKET_SENTIMENT,
                entities=article.related_stocks if article.related_stocks else ["市场"],
                magnitude="low",
                timeframe=TimeHorizon.SHORT,
                transmission_chain=["新闻发布", "市场情绪变化", "短期波动"],
                confidence=0.5,
                raw_evidence="一般市场新闻",
            ))

        return events
