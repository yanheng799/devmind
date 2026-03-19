# DevMind

News-based Stock Price Prediction Agent

## Overview

DevMind is an intelligent agent that analyzes international and domestic news to predict stock price movements. It combines:

- **News Collection**: RSS feeds from Reuters, Bloomberg, and Chinese financial media
- **Event Extraction**: LLM-powered structured event extraction
- **Historical Pattern Matching**: Vector-based retrieval of similar historical events
- **Multi-factor Reasoning**: ReAct-based agent for decision making

## Architecture

![Architecture](news_stock_agent_architecture.svg)

## Project Structure

```
devmind/
├── src/devmind/
│   ├── agents/      # Agent orchestration with LangGraph
│   ├── data/        # News and stock data collection
│   ├── models/      # Event extraction and prediction models
│   └── utils/       # Utility functions
├── tests/           # Unit and integration tests
├── config/          # Configuration files
└── data/            # Data storage
```

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
TUSHARE_TOKEN=your_token_here  # Optional
```

## Usage

```python
from devmind.agents import StockPredictionAgent

agent = StockPredictionAgent()
result = agent.predict(news_text="美联储宣布降息25个基点")
print(result)
```

## License

MIT
