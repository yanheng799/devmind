# DevMind 知识问答系统 — 开发任务计划

## 任务依赖关系

```
Phase 0   #1 代码清理
  ↓
Phase 1a  #2 依赖/配置更新
  ↓            ↘
Phase 1b  #3 DB迁移   Phase 1c #4 领域层
  ↓     ↘             ↓          ↘
Phase 1d #5 适配器   Phase 2a #6 策略   Phase 2b #7 新基础设施
  ↓          ↘         ↓          ↘
Phase 2c  #8 领域服务
  ↓
Phase 3   #9 应用服务
  ↓         ↘
Phase 4   #10 CLI+WebUI   Phase 5 #11 测试
  ↓         ↘
Phase 6   #12 集成验证
```

**关键路径**: #1 → #2 → #3 → #5 → #8 → #9 → #10 → #12
**可并行**: #3 和 #4；#6 和 #7；#10 和 #11

---

## Phase 0 — 代码清理

### #1 删除死代码和清理项目

**状态**: pending
**前置**: 无
**产出**: 干净的项目代码库，无死代码，无 import 错误

**步骤**:

| # | 操作 | 文件/目录 |
|---|------|----------|
| 1.1 | 删除整个目录 | `src/agents/` |
| 1.2 | 删除整个目录 | `src/split/` |
| 1.3 | 删除整个目录 | `src/rag/` |
| 1.4 | 删除整个目录 | `src/utils/` |
| 1.5 | 删除测试脚本 | `tests/split_document.py` |
| 1.6 | 删除测试脚本 | `tests/split_document_v2.py` |
| 1.7 | 删除测试脚本 | `tests/embedding.py` |
| 1.8 | 删除测试脚本 | `tests/milvus_study.py` |
| 1.9 | 修复 import | `src/models/__init__.py`: `from models.document` → `from devmind.models.document` |
| 1.10 | 修复 import | `src/models/base.py`: `from models.types` → `from devmind.models.types` |
| 1.11 | 删除无效 import | `src/cli.py`: 删除 `from devmind.agents.stock_agent import StockPredictionAgent` 及相关引用 |
| 1.12 | 删除无效字段 | `src/config/settings.py`: 删除 `news_sources_config` 字段 |
| 1.13 | 验证 | `pytest tests/test_models.py -v` 确保现有测试通过 |

---

## Phase 1 — 领域层 + 基础设施 + 数据库

### #2 更新依赖、配置和环境变量

**状态**: pending
**前置**: #1
**产出**: pyproject.toml 新依赖就绪，Settings 新配置可用，.env 更新

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 2.1 | 添加依赖 | `pyproject.toml`: `gradio`, `minio`, `pdfplumber`, `Pillow`, `jieba`, `sqlalchemy>=2.0`, `psycopg[binary]`, `openai` |
| 2.2 | 确认版本 | `pyproject.toml`: `pymilvus>=2.4.0` (支持稀疏向量) |
| 2.3 | 移除字段 | `src/config/settings.py`: 删除 `db_path` |
| 2.4 | 新增 PG 配置 | `src/config/settings.py`: `postgres_host`, `postgres_port`, `postgres_username`, `postgres_password`, `postgres_database`, `postgres_pool_size`, `postgres_url` property |
| 2.5 | 新增 QA 配置 | `src/config/settings.py`: `qa_context_turns`, `qa_top_k`, `qa_min_score`, `qa_collection_name`, `qa_enable_query_rewrite`, `qa_enable_hybrid_search`, `qa_bm25_tokenizer` 等 |
| 2.6 | 新增 MinIO 配置 | `src/config/settings.py`: `minio_endpoint`, `minio_access_key`, `minio_secret_key`, `minio_bucket`, `minio_secure` |
| 2.7 | 新增图片配置 | `src/config/settings.py`: `qa_image_max_size`, `qa_image_max_dimensions`, `qa_image_formats`, `qa_generate_image_description` |
| 2.8 | 新增方法 | `src/config/settings.py`: `get_qa_config()`, `get_minio_config()` |
| 2.9 | 创建 docker-compose | `docker-compose.yml`: PostgreSQL 16 + MinIO + Milvus 2.4 服务定义 |
| 2.10 | 更新 .env | `.env`: 写入实际配置值 (PostgreSQL, MinIO, LLM, Embedding, Milvus) |
| 2.11 | 验证 | `docker compose up -d` → `pip install -e ".[dev]"` → `python -c "from devmind.config import get_settings; s = get_settings(); print(s.postgres_url)"` |

---

### #3 数据库从 SQLite 迁移到 PostgreSQL

**状态**: pending
**前置**: #2
**产出**: SQLAlchemy 2.0 ORM 模型，PostgreSQL 连接池，所有 CRUD 方法迁移完成

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 3.1 | 创建 ORM 模型 | `src/knowledgeqa/infrastructure/database/models.py`: Base, NewsArticleModel, ExtractedEventModel, PredictionModel, PredictionOutcomeModel, HistoricalEventModel, StockPriceModel, DocumentModel, DocumentChunkModel, ConversationModel, MessageModel |
| 3.2 | 创建会话管理 | `src/knowledgeqa/infrastructure/database/session.py`: DatabaseSessionManager (engine, sessionmaker, session() as context manager, create_tables, close) |
| 3.3 | 迁移 news CRUD | `src/data/database/database.py`: insert_news_article, get_news_article → SQLAlchemy |
| 3.4 | 迁移 event CRUD | `src/data/database/database.py`: insert_event, get_events_by_article → SQLAlchemy |
| 3.5 | 迁移 prediction CRUD | `src/data/database/database.py`: insert_prediction, get_prediction, get_predictions_by_stock, get_pending_predictions, update_prediction_status, insert_outcome → SQLAlchemy |
| 3.6 | 迁移 historical CRUD | `src/data/database/database.py`: insert_historical_event, get_historical_events → SQLAlchemy |
| 3.7 | 迁移 stock CRUD | `src/data/database/database.py`: insert_stock_price, get_stock_prices, get_latest_price → SQLAlchemy |
| 3.8 | 迁移 document CRUD | `src/data/database/database.py`: insert_document, get_document, get_document_by_checksum, list_documents, update_document_status, delete_document → SQLAlchemy |
| 3.9 | 迁移 chunk CRUD | `src/data/database/database.py`: insert_chunk, get_chunks_by_document, get_chunk_by_embedding_id, update_chunk_embedding_id, delete_chunks_by_document → SQLAlchemy |
| 3.10 | 更新构造函数 | `src/data/database/database.py`: `__init__` 改为接收 DatabaseSessionManager 而非 db_path |
| 3.11 | 更新调用方 | `src/cli.py` 及其他使用 PredictionDatabase 的地方，适配新构造方式 |
| 3.12 | 验证 | `pytest tests/test_models.py tests/test_collectors.py -v` 确保现有测试通过 |

---

### #4 创建领域层

**状态**: pending
**前置**: #2
**产出**: knowledgeqa 模块骨架，领域模型、Protocol 接口、异常类

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 4.1 | 创建包 | `src/knowledgeqa/__init__.py` |
| 4.2 | 创建枚举 | `src/knowledgeqa/domain/models.py`: MessageRole, SourceType, ConversationStatus |
| 4.3 | 创建值对象 | `src/knowledgeqa/domain/models.py`: SourceReference, Citation, RetrievedChunk, ImageInfo, QueryResult |
| 4.4 | 创建实体 | `src/knowledgeqa/domain/models.py`: Message, Conversation (含 add_message, get_recent_messages, get_context_window, to_llm_messages), IngestedDocument |
| 4.5 | 创建 Protocol | `src/knowledgeqa/domain/protocols.py`: EmbeddingProvider, SparseEmbeddingProvider, VectorStore, DocumentPersistence, ConversationPersistence, LlmClient, WebContentLoader, ImageStorage |
| 4.6 | 创建异常 | `src/knowledgeqa/domain/exceptions.py`: KnowledgeQAError → DocumentValidationError, DocumentIngestionError, DocumentDuplicateError, WebLoadingError, RetrievalError, AnswerGenerationError, ConversationNotFoundError, EmbeddingError |
| 4.7 | 验证 | `python -c "from devmind.knowledgeqa.domain.models import Conversation, QueryResult"` 无报错 |

---

### #5 创建基础设施适配器、Mock 和 Repository

**状态**: pending
**前置**: #3, #4
**产出**: 所有 Protocol 的生产实现和 Mock，数据库 Repository

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 5.1 | Embedding 适配器 | `infrastructure/adapters.py`: DashScopeEmbeddingAdapter (embed_single, embed_batch) |
| 5.2 | VectorStore 适配器 | `infrastructure/adapters.py`: HybridVectorStoreAdapter — 创建新 Milvus collection (含 SPARSE_FLOAT_VECTOR), insert_chunk 同时写入稠密+稀疏向量, search_chunks 使用 HybridSearchRequest + RRFRanker |
| 5.3 | Document 适配器 | `infrastructure/adapters.py`: DocumentProcessorAdapter (validate_file, calculate_checksum, parse_file, chunk_text) |
| 5.4 | Mock 实现 | `infrastructure/mock_adapters.py`: MockEmbeddingProvider, MockSparseEmbeddingProvider, MockVectorStore, MockDocumentPersistence, MockConversationPersistence, MockLlmClient, MockWebContentLoader, MockImageStorage |
| 5.5 | Document Repository | `infrastructure/database/repositories.py`: DocumentRepository 实现 DocumentPersistence Protocol |
| 5.6 | Conversation Repository | `infrastructure/database/repositories.py`: ConversationRepository 实现 ConversationPersistence Protocol |
| 5.7 | 验证 | `pytest tests/test_models.py -v` + 适配器单元测试 |

---

## Phase 2 — 策略 + 新基础设施 + 领域服务

### #6 实现文档处理策略

**状态**: pending
**前置**: #4
**产出**: 4 种文档策略 + 注册表，统一分块格式

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 6.1 | 策略基类 | `infrastructure/strategies/base.py`: ParseResult, TableInfo, DocumentStrategy Protocol, DocumentStrategyRegistry, create_default_registry() |
| 6.2 | PDF 策略 | `infrastructure/strategies/pdf_strategy.py`: pypdf 文本提取 + 图片提取 + pdfplumber 表格 (文本+截图→存 MinIO URL) + 按页分块 |
| 6.3 | DOCX 策略 | `infrastructure/strategies/docx_strategy.py`: python-docx 段落+标题+图片+表格 + 按标题层级分块（不支持旧 .doc 格式，需提示转换） |
| 6.4 | Markdown 策略 | `infrastructure/strategies/markdown_strategy.py`: 正则解析 + 代码块保持完整 + 按 # 标题分块 |
| 6.5 | Web 策略 | `infrastructure/strategies/web_strategy.py`: BS4 正文提取 + 图片下载 + HTML 结构分块 |
| 6.6 | Mock 策略 | `infrastructure/strategies/mock_strategy.py`: MockDocumentStrategy |
| 6.7 | 验证 | 各策略单元测试：输入样例文件 → 验证 parse 输出格式 → 验证 chunk 输出格式 |

---

### #7 实现稀疏嵌入、LLM 客户端、图片存储、网页加载

**状态**: pending
**前置**: #4
**产出**: 4 个基础设施组件 + 对应 Mock

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 7.1 | BM25 稀疏嵌入 | `infrastructure/sparse_embedding.py`: JiebaBM25Embedding (jieba 分词 → BM25 权重 → dict[str, float]) |
| 7.2 | LLM 客户端 | `infrastructure/llm_client.py`: DashScopeLlmClient (OpenAI 兼容, qwen3-max) |
| 7.3 | MinIO 图片存储 | `infrastructure/image_storage.py`: MinIOImageStorage (upload, download, delete, get_url, exists) + PIL 缩略图 |
| 7.4 | 网页加载器 | `infrastructure/web_loader.py`: WebContentLoader (requests + BS4, 正文提取, 图片列表, 超时) |
| 7.5 | Mock 实现 | 各文件内 Mock 类 |
| 7.6 | 验证 | 稀疏嵌入单元测试 + LLM 客户端 mock 测试 + MinIO 连通测试 |

---

### #8 实现领域服务

**状态**: pending
**前置**: #5, #6, #7
**产出**: 4 个领域服务，完整查询流水线可跑通

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 8.1 | QueryRewriter | `domain/services.py`: 用 LLM 改写查询，无上下文时直通 |
| 8.2 | RetrievalOrchestrator | `domain/services.py`: 编排稠密+稀疏嵌入 → VectorStore 混合检索 → 阈值过滤 |
| 8.3 | AnswerGenerator | `domain/services.py`: 组装 prompt + 生成答案 + 提取 citations + 汇总 images |
| 8.4 | ImageProcessor | `domain/services.py`: 校验图片 → 上传 MinIO → 生成描述 (可选) → 返回 ImageInfo |
| 8.5 | 验证 | 全 Mock 端到端：查询 → 改写 → 检索 → 生成答案，验证 QueryResult 结构 |

---

## Phase 3 — 应用服务

### #9 实现应用服务

**状态**: pending
**前置**: #8
**产出**: 4 个应用服务，完整业务用例可编排

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 9.1 | ConversationService | `application/conversation_service.py`: create/get/list/add_message/archive/delete/get_history |
| 9.2 | IngestService | `application/ingest_service.py`: ingest_file (策略解析→去重→图片→分块→嵌入→存储), ingest_url |
| 9.3 | QueryService | `application/query_service.py`: ask (会话→改写→检索→生成→持久化) |
| 9.4 | KnowledgeService | `application/knowledge_service.py`: list_documents, delete_document, get_document_stats |
| 9.5 | 验证 | 全 Mock 集成测试：ingest_file + ask 完整链路 |

---

## Phase 4 — 接口层

### #10 实现 CLI 命令和 Gradio Web UI

**状态**: pending
**前置**: #9
**产出**: 可用的 CLI 命令和 Web 界面

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 10.1 | CLI 注册 | `interface/cli.py`: add_knowledgeqa_commands(subparsers) 函数 |
| 10.2 | qa 命令 | `interface/cli.py`: cmd_qa — 单轮/多轮问答，--session, --top-k 参数 |
| 10.3 | upload-url 命令 | `interface/cli.py`: cmd_upload_url — 网页入库 |
| 10.4 | qa-history 命令 | `interface/cli.py`: cmd_qa_history — 列出/查看会话 |
| 10.5 | qa-delete 命令 | `interface/cli.py`: cmd_qa_delete — 删除会话 |
| 10.6 | 集成到主 CLI | `src/cli.py`: main() 中调用 add_knowledgeqa_commands() |
| 10.7 | Gradio Web UI | `interface/web_ui.py`: 左栏(文件上传+URL+文档列表) + 右栏(聊天界面) |
| 10.8 | Mock 模式 | 所有 CLI 命令支持 --mock 全局标志 |
| 10.9 | 验证 | `devmind --mock qa "测试"` → 返回 mock 回答 |

---

## Phase 5 — 测试

### #11 编写单元测试

**状态**: pending
**前置**: #9
**产出**: tests/knowledgeqa/ 全部通过

**步骤**:

| # | 操作 | 文件 |
|---|------|------|
| 11.1 | 领域模型测试 | `tests/knowledgeqa/test_domain_models.py`: 边界值、约束验证 |
| 11.2 | 领域服务测试 | `tests/knowledgeqa/test_domain_services.py`: QueryRewriter, RetrievalOrchestrator, AnswerGenerator |
| 11.3 | 策略测试 | `tests/knowledgeqa/test_strategies.py`: 各策略 parse/chunk 格式验证 |
| 11.4 | 稀疏嵌入测试 | `tests/knowledgeqa/test_sparse_embedding.py`: 分词、权重、batch |
| 11.5 | 对话服务测试 | `tests/knowledgeqa/test_conversation_service.py`: 完整生命周期 |
| 11.6 | 摄入服务测试 | `tests/knowledgeqa/test_ingest_service.py`: 正常/重复/格式错误 |
| 11.7 | 查询服务测试 | `tests/knowledgeqa/test_query_service.py`: 端到端 Mock 链路 |
| 11.8 | 适配器测试 | `tests/knowledgeqa/test_adapters.py`: 方法映射正确性 |
| 11.9 | 图片存储测试 | `tests/knowledgeqa/test_image_storage.py`: MockImageStorage |
| 11.10 | 运行全部 | `pytest tests/knowledgeqa/ -v` 全绿 |

---

## Phase 6 — 集成验证

### #12 端到端集成验证

**状态**: pending
**前置**: #10, #11
**产出**: 系统完整可用

**步骤**:

| # | 操作 | 命令 |
|---|------|------|
| 12.1 | 安装依赖 | `pip install -e ".[dev]"` |
| 12.2 | 全部测试 | `pytest tests/ -v` |
| 12.3 | Lint | `ruff check src tests` |
| 12.4 | 格式 | `black --check src tests` |
| 12.5 | 文件摄入 | `devmind upload-doc data/milvus_docs/en/faq/milvus_faq.md` |
| 12.6 | 单轮问答 | `devmind qa "什么是Milvus?"` |
| 12.7 | 多轮对话 | `devmind qa --session new "介绍向量数据库"` → `devmind qa --session <id> "索引类型"` |
| 12.8 | 网页摄入 | `devmind upload-url https://milvus.io/docs/overview.md` |
| 12.9 | 会话管理 | `devmind qa-history` → `devmind qa-history --session <id>` → `devmind qa-delete --session <id>` |
| 12.10 | Web UI | `devmind qa-web` → 浏览器访问，验证文件上传和聊天 |

---

## 补充说明

### .doc 旧格式
`2.杆塔明细表.doc` 为旧 Word 格式，python-docx 不支持。CLI 收到 .doc 文件时提示用户先转换为 .docx。

### 大图片处理
`公式.png` 

### Docker Compose
需在项目根目录新建 `docker-compose.yml`（PostgreSQL 16 + MinIO + Milvus 2.4），参见 `docs/knowledgeqa-design.md §16.3`。

### VectorStore Adapter 复杂度
Adapter 不是简单方法名翻译，需要创建新 Milvus collection（含 `SPARSE_FLOAT_VECTOR`），实现 `HybridSearchRequest` + `RRFRanker`。Mock 实现也需支持双路检索和 RRF 融合。

### ORM 模型位置统一
所有 SQLAlchemy 模型统一放在 `src/knowledgeqa/infrastructure/database/models.py`。

### 数据库 Session 生命周期
使用 context manager 模式：`with db_manager.session() as session:`，自动 commit/rollback。
