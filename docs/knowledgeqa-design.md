# DevMind 企业级知识问答系统 — DDD 设计文档

## Context

基于 `src/agents/chat_model.py` 原型（LangChain RAG pipeline），在 DevMind 项目内新建 `src/knowledgeqa/` 模块，实现企业级知识问答系统。采用领域驱动设计（DDD），复用现有基础设施（向量存储、嵌入模型、文档处理、配置），新增多轮对话、网页抓取、图片存储/召回（MinIO）、混合检索（稠密+稀疏向量）、统一知识管理能力。同时清理项目中不用的死代码和修复已知 bug。

---

## 0. 代码清理 (实施前先做)

### 删除死代码

| 路径 | 原因 |
|------|------|
| `src/agents/` (整个目录) | chat_model.py 是 LangChain 临时脚本，从未被引用；`__init__.py` 仅空文档串 |
| `src/split/` (整个目录) | 空模块，无任何引用 |
| `src/rag/` (整个目录) | FaqIngestor/FaqRetriever 从未被外部使用，将被 knowledgeqa 替代 |
| `src/utils/` (整个目录) | 空模块，无任何引用 |
| `tests/split_document.py` | PDF 拆分实验脚本，非 pytest 测试 |
| `tests/split_document_v2.py` | 同上 v2 |
| `tests/embedding.py` | API 调试脚本，含硬编码 API key |
| `tests/milvus_study.py` | 已损坏的原型，引用不存在的模块 |

### 修复 Bug

| 文件 | 问题 | 修复 |
|------|------|------|
| `src/models/__init__.py:9` | `from models.document` → 缺少包前缀 | 改为 `from devmind.models.document` |
| `src/models/base.py:8` | `from models.types` → 缺少包前缀 | 改为 `from devmind.models.types` |
| `src/cli.py:13` | `from devmind.agents.stock_agent import StockPredictionAgent` → 文件不存在 | 删除此行及相关引用（StockPredictionAgent 未实现） |
| `src/config/settings.py:66` | `news_sources_config` 引用不存在的 `config/sources.yaml` | 删除此字段 |

---

## 1. 文档处理策略 (策略模式)

### 1.1 设计思路

现有 `DocumentProcessor` 对所有文档类型使用相同的解析和分块逻辑，这不够灵活。新设计采用 **策略模式 + 注册表**，每种文档类型有独立的处理策略，负责解析、图片提取、分块。

### 1.2 策略接口

```python
# infrastructure/strategies/base.py

class ParseResult(BaseModel):
    """文档解析结果。"""
    text: str
    title: str
    author: str | None = None
    images: list[ImageInfo] = Field(default_factory=list)
    tables: list[TableInfo] = Field(default_factory=list)
    metadata: dict[str, str | int | bool] = Field(default_factory=dict)

class TableInfo(BaseModel):
    """表格信息。"""
    table_id: str
    content: str              # Markdown 格式表格文本
    image_url: str | None     # 表格截图 URL (存储在 MinIO)
    row_count: int
    col_count: int
    page_number: int | None = None

class DocumentStrategy(Protocol):
    """文档处理策略接口，每种文档类型一个实现。"""

    @property
    def supported_extensions(self) -> list[str]:
        """支持的文件扩展名，如 ['pdf'], ['docx']。"""
        ...

    def parse(self, file_path: str) -> ParseResult:
        """解析文档，提取文本、图片、表格。"""
        ...

    def chunk(self, result: ParseResult, chunk_size: int, chunk_overlap: int) -> list[dict]:
        """将解析结果分块。

        每个块返回:
        {
            "content": str,           # 块文本
            "chunk_type": str,        # "text" | "table" | "image_description"
            "images": list[ImageInfo], # 关联图片
            "tables": list[TableInfo], # 关联表格
            "page_number": int | None,
            "metadata": dict,
        }
        """
        ...
```

### 1.3 四种策略实现

#### PdfStrategy

```
PDF 文件
  ├── 解析: pypdf 提取文本 (按页)
  ├── 图片提取: pypdf 提取内嵌图片 → PIL 校验/缩放 → MinIO 上传
  ├── 表格处理:
  │     ├── 尝试用 pdfplumber 提取表格结构 → 转为 Markdown 表格文本
  │     ├── 用 pdfplumber 截取表格图片 → MinIO 上传
  │     └── 同时保留文本和图片两种形式
  ├── 分块策略:
  │     ├── 优先按页分块 (page-aware)
  │     ├── 页内按段落分割
  │     └── 表格作为独立块 (不拆分表格)
  └── 元数据: page_count, has_images, has_tables
```

#### DocxStrategy

```
DOCX 文件
  ├── 解析: python-docx 提取段落 + 标题样式
  ├── 图片提取: 提取 DOCX 内嵌图片 (relationships) → MinIO 上传
  ├── 表格处理:
  │     ├── 遍历 doc.tables 提取单元格 → 转 Markdown 表格
  │     └── 表格截图暂不支持 (可后续通过渲染引擎添加)
  ├── 分块策略:
  │     ├── 按标题层级分块 (Heading 1/2 作为块边界)
  │     ├── 标题作为块的前缀上下文
  │     ├── 表格作为独立块
  │     └── 段落按语义长度分割
  └── 元数据: paragraph_count, heading_count, table_count, image_count
```

#### MarkdownStrategy

```
Markdown 文件
  ├── 解析: 正则/解析器提取标题、代码块、图片链接
  ├── 图片提取: 下载 `![alt](url)` 引用的图片 → MinIO 上传
  ├── 表格处理: 直接保留 Markdown 表格语法
  ├── 分块策略:
  │     ├── 按 `#` 标题层级分块 (Heading 1/2/3 作为块边界)
  │     ├── 代码块保持完整 (不跨块拆分)
  │     ├── 表格保持完整
  │     └── 块头部附加标题路径 (如 "## 1.2 > ### 1.2.3")
  └── 元数据: heading_count, code_block_count, image_count
```

#### WebStrategy

```
网页 URL
  ├── 解析: requests + BeautifulSoup 提取正文
  │     ├── 移除 nav/header/footer/script/style 等非正文标签
  │     ├── 提取 <article> 或 <main> 内容
  │     └── 提取 title, author, publish_time
  ├── 图片提取: 下载 <img> 标签图片 (过滤 icon/logo 等小图) → MinIO 上传
  ├── 表格处理: 提取 <table> → 转 Markdown 表格
  ├── 分块策略:
  │     ├── 按 HTML 标签结构分块 (h1-h6 作为块边界)
  │     ├── 块头部附加标题上下文
  │     └── 长段落按句子边界分割
  └── 元数据: url, domain, content_type, image_count
```

### 1.4 策略注册表

```python
class DocumentStrategyRegistry:
    """策略注册表，根据文件类型选择处理策略。"""

    def __init__(self) -> None:
        self._strategies: dict[str, DocumentStrategy] = {}

    def register(self, strategy: DocumentStrategy) -> None:
        for ext in strategy.supported_extensions:
            self._strategies[ext] = strategy

    def get_strategy(self, file_extension: str) -> DocumentStrategy:
        ext = file_extension.lstrip(".").lower()
        if ext not in self._strategies:
            raise ValueError(
                f"No strategy for file type: {ext}. "
                f"Supported: {', '.join(self._strategies)}"
            )
        return self._strategies[ext]

    @property
    def supported_extensions(self) -> list[str]:
        return list(self._strategies.keys())


def create_default_registry() -> DocumentStrategyRegistry:
    """创建默认策略注册表。"""
    registry = DocumentStrategyRegistry()
    registry.register(PdfStrategy())
    registry.register(DocxStrategy())
    registry.register(MarkdownStrategy())
    registry.register(WebStrategy())  # "html", "web"
    return registry
```

### 1.5 分块结果统一格式

所有策略的 `chunk()` 方法返回统一的 `list[dict]`，每项包含:

```python
{
    "content": str,             # 块文本内容 (用于嵌入和检索)
    "chunk_type": str,          # "text" | "table" | "code" | "image_description"
    "images": list[ImageInfo],  # 关联的图片信息
    "tables": list[TableInfo],  # 关联的表格信息
    "heading_path": str | None, # 标题路径 (如 "Ch1 > Sec1.2")
    "page_number": int | None,  # PDF 页码
    "metadata": dict,           # 策略特定的额外信息
}
```

这种设计使得下游的 `IngestService` 不需要知道文档类型，统一处理所有块：嵌入文本 → 存入 Milvus → 图片 URL 放入 metadata。

---

## 2. 限界上下文

```
DevMind
├── Stock Prediction Context  (现有，保留)
└── Knowledge Q&A Context     (新增) ← 本次实现

共享内核 (Shared Kernel): Settings, DocumentProcessor, DashScopeEmbeddingModel,
                         DocumentVectorStore, DirectoryWatcher

替换基础设施: PredictionDatabase(SQLite) → PostgreSQL (SQLAlchemy 2.0)

新增基础设施: PostgreSQL (关系数据库), MinIO (图片对象存储), Gradio (Web UI)
```

新模块通过 **Protocol 接口** 包装现有组件，领域层不直接依赖具体实现。

---

## 3. 模块结构

```
src/knowledgeqa/
├── __init__.py
├── domain/                        # 领域层
│   ├── __init__.py
│   ├── models.py                  # 实体: Conversation, Message, IngestedDocument
│   │                              # 值对象: Citation, RetrievedChunk, QueryResult, SourceReference, ImageInfo
│   ├── protocols.py               # Protocol 接口 (EmbeddingProvider, VectorStore, ImageStorage, ...)
│   ├── exceptions.py              # 领域异常
│   └── services.py                # 领域服务 (QueryRewriter, RetrievalOrchestrator, AnswerGenerator)
├── application/                   # 应用层 (用例编排)
│   ├── __init__.py
│   ├── ingest_service.py          # 文档/网页/图片摄入用例
│   ├── query_service.py           # 问答查询用例 (核心)
│   ├── conversation_service.py    # 对话管理用例
│   └── knowledge_service.py       # 知识库管理用例
├── infrastructure/                # 基础设施层
│   ├── __init__.py
│   ├── database/                  # PostgreSQL 数据库
│   │   ├── __init__.py
│   │   ├── models.py              # SQLAlchemy ORM 模型 (Document, Chunk, Conversation, Message)
│   │   ├── session.py             # 会话管理器 (连接池、create_tables)
│   │   └── repositories.py        # Repository 实现 (DocumentRepo, ConversationRepo)
│   ├── adapters.py                # 包装现有组件: DashScopeEmbeddingAdapter, DocumentVectorStoreAdapter, JiebaSparseEmbeddingAdapter
│   ├── strategies/                # 文档处理策略
│   │   ├── __init__.py
│   │   ├── base.py                # ParseResult, TableInfo, DocumentStrategy(Protocol), DocumentStrategyRegistry
│   │   ├── pdf_strategy.py        # PdfStrategy
│   │   ├── docx_strategy.py       # DocxStrategy
│   │   ├── markdown_strategy.py   # MarkdownStrategy
│   │   ├── web_strategy.py        # WebStrategy
│   │   └── mock_strategy.py       # MockDocumentStrategy
│   ├── sparse_embedding.py        # JiebaBM25Embedding (jieba 分词 → BM25 稀疏向量) + Mock
│   ├── llm_client.py              # LlmClient 实现 (OpenAI兼容) + MockLlmClient
│   ├── web_loader.py              # WebContentLoader (requests+BS4) + MockWebContentLoader
│   ├── image_storage.py           # MinIO 图片存储 (minio SDK) + MockImageStorage
│   └── mock_adapters.py           # MockEmbeddingProvider, MockVectorStore, MockDocumentPersistence
└── interface/                     # 接口层
    ├── __init__.py
    ├── cli.py                     # CLI 命令 (qa, upload-url, qa-history, qa-delete)
    └── web_ui.py                  # Gradio Web UI
```

---

## 4. 领域模型

### 实体

| 实体 | 标识 | 关键属性 |
|------|------|----------|
| `Conversation` | `conversation_id` | title, status, messages[], created_at, updated_at |
| `Message` | `message_id` | conversation_id, role(user/assistant/system), content, citations[], created_at |
| `IngestedDocument` | `document_id` | source_type(document/web/milvus), title, url, file_path, checksum, chunk_count, status |

### 值对象

| 值对象 | 用途 |
|--------|------|
| `SourceReference` | 知识来源引用: source_type, source_id, title, url |
| `Citation` | 答案引用: chunk_id, document_id, content_snippet, score, source |
| `RetrievedChunk` | 检索结果: chunk_id, document_id, content, score, doc_type, metadata |
| `ImageInfo` | 图片信息: image_id, object_key, url, filename, mime_type, size, width, height, description |
| `QueryResult` | 查询结果: query, answer, citations[], retrieved_chunks[], images[], confidence |

### 枚举

- `MessageRole`: user, assistant, system
- `SourceType`: document, web, milvus
- `ConversationStatus`: active, archived, deleted

---

## 5. 图片存储与召回 (MinIO)

### 5.1 存储架构

```
文档摄入
  → 解析文档/网页
  → 提取文本内容 (chunking → embedding → Milvus)
  → 提取图片
      → 上传到 MinIO (bucket: devmind-images)
      → 图片元数据存入 Milvus (同一 collection 的 metadata JSON 字段中)
      → 图片描述文本用于嵌入检索
```

### 5.2 图片处理流程

```
[PDF/DOCX 解析]                    [网页抓取]
     │                                  │
     ▼                                  ▼
提取内嵌图片                      下载 <img> 资源
     │                                  │
     ▼                                  ▼
生成缩略图 (可选)                  生成缩略图 (可选)
     │                                  │
     └──────────┬───────────────────────┘
                ▼
        上传到 MinIO
        获取 object_key → 生成访问 URL
                │
                ▼
        图片元数据 (ImageInfo) 关联到 DocumentChunk.metadata
                │
                ▼
        用 LLM 生成图片描述 → 作为 chunk 内容存入 Milvus
```

### 5.3 MinIO 存储结构

```
Bucket: devmind-images
├── documents/{document_id}/{image_id}.{ext}     # 原图
├── documents/{document_id}/thumb_{image_id}.{ext}  # 缩略图 (可选)
└── web/{url_hash}/{image_id}.{ext}              # 网页图片
```

### 5.4 图片召回方式

1. **文本关联召回**: 图片有 LLM 生成的描述文本，该描述作为 chunk 存入 Milvus。当用户查询匹配到该描述 chunk 时，从 metadata 中提取图片 URL 一并返回。
2. **直接检索**: `RetrievedChunk.metadata["images"]` 包含 `list[ImageInfo]`，在 `QueryResult` 中汇总为 `images` 字段。

### 5.5 ImageStorage Protocol

```python
class ImageStorage(Protocol):
    def upload(self, data: bytes, object_key: str, content_type: str) -> str: ...
    # 返回公开访问 URL

    def download(self, object_key: str) -> bytes: ...
    # 返回原始图片数据

    def delete(self, object_key: str) -> None: ...

    def get_url(self, object_key: str) -> str: ...
    # 获取已上传图片的访问 URL

    def exists(self, object_key: str) -> bool: ...

    def close(self) -> None: ...
```

### 5.6 MinIO 配置

```python
# 新增到 Settings
minio_endpoint: str = "localhost:9000"
minio_access_key: str = ""
minio_secret_key: str = ""
minio_bucket: str = "devmind-images"
minio_secure: bool = False
qa_image_max_size: int = 10 * 1024 * 1024   # 10MB
qa_image_max_dimensions: int = 4096           # 最大边长
qa_image_formats: list[str] = ["png", "jpg", "jpeg", "gif", "webp", "svg"]
qa_generate_image_description: bool = True    # 用 LLM 生成图片描述
```

---

## 6. Protocol 接口 (依赖注入)

```python
# domain/protocols.py

class EmbeddingProvider(Protocol):
    """稠密向量嵌入 (语义匹配)。"""
    def embed_single(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def close(self) -> None: ...

class SparseEmbeddingProvider(Protocol):
    """稀疏向量嵌入 (关键词/BM25 匹配)。

    使用 BM25 或 SPLADE 风格的稀疏表示，支持中文分词。
    返回 dict[str, float] 表示 {token: weight}。
    """
    def embed_single(self, text: str) -> dict[str, float]: ...
    def embed_batch(self, texts: list[str]) -> list[dict[str, float]]: ...
    def close(self) -> None: ...

class VectorStore(Protocol):
    """支持混合检索的向量存储 (稠密 + 稀疏)。"""

    def insert_chunk(
        self, chunk_id, document_id, embedding, content,
        doc_type, metadata=None, sparse_embedding=None,
    ) -> str: ...

    def search_chunks(
        self, query_embedding, top_k=5,
        doc_type=None, document_id=None,
        sparse_embedding=None,  # 混合检索时传入稀疏向量
    ) -> list[dict]: ...

    def delete_chunks(self, document_id: str) -> int: ...
    def close(self) -> None: ...

class DocumentParser(Protocol):
    def validate_file(self, file_path: str) -> tuple[bool, str | None]: ...
    def calculate_checksum(self, file_path: str) -> str: ...
    def parse_file(self, file_path: str) -> dict: ...
    def chunk_text(self, text: str) -> list[str]: ...

class DocumentPersistence(Protocol):
    def insert_document(self, document: dict) -> str: ...
    def get_document(self, document_id: str) -> dict | None: ...
    def get_document_by_checksum(self, checksum: str) -> dict | None: ...
    def list_documents(self, file_type=None, status=None, limit=100) -> list[dict]: ...
    def update_document_status(self, document_id, status, chunk_count=None) -> None: ...
    def delete_document(self, document_id: str) -> None: ...

class ConversationPersistence(Protocol):
    def save_conversation(self, conversation: Conversation) -> None: ...
    def get_conversation(self, conversation_id: str) -> Conversation | None: ...
    def list_conversations(self, status=None, limit=50) -> list[Conversation]: ...
    def delete_conversation(self, conversation_id: str) -> None: ...
    def close(self) -> None: ...

class LlmClient(Protocol):
    def generate(self, messages: list[dict[str, str]], temperature=0.7, max_tokens=4096) -> str: ...
    def close(self) -> None: ...

class WebContentLoader(Protocol):
    def load_url(self, url: str) -> dict: ...
    # 返回 {text, title, url, images: list[dict], metadata}

class ImageStorage(Protocol):
    def upload(self, data: bytes, object_key: str, content_type: str) -> str: ...
    def download(self, object_key: str) -> bytes: ...
    def delete(self, object_key: str) -> None: ...
    def get_url(self, object_key: str) -> str: ...
    def exists(self, object_key: str) -> bool: ...
    def close(self) -> None: ...
```

**设计要点**: Protocol 的 snake_case 方法名与现有组件的 camelCase 方法名不同。通过 Adapter 层做翻译，不改现有代码。

---

## 7. 领域服务

### QueryRewriter
- 用 LLM 将用户查询结合对话上下文改写为独立检索查询
- 无上下文时直接返回原查询
- 可通过 `qa_enable_query_rewrite` 配置开关

### RetrievalOrchestrator
- 编排 **混合检索**: 稠密向量 (语义) + 稀疏向量 (关键词/BM25) → RRF 融合
- 依赖 `EmbeddingProvider` + `SparseEmbeddingProvider` + `VectorStore`
- 返回 `list[RetrievedChunk]`（包含关联图片信息）
- 支持最低相似度阈值过滤 (`qa_min_score`)
- 混合检索可通过 `qa_enable_hybrid_search` 配置开关

### AnswerGenerator
- 组装 system prompt + 对话历史 + 检索上下文 + 用户问题
- 调用 LLM 生成带引用标记的答案
- 提取 citations 映射回 RetrievedChunks
- 汇总检索结果中的图片信息到 QueryResult.images
- 无相关内容时诚实回答

### ImageProcessor (新增)
- 从 PDF/DOCX 中提取内嵌图片
- 从网页中下载图片资源
- 图片校验 (格式、大小、尺寸)
- 调用 LLM 生成图片描述文本 (可选)
- 上传图片到 MinIO 并返回 ImageInfo

---

## 8. 查询流水线

```
用户提问
  → [1] 加载/创建会话 (ConversationService)
  → [2] 添加用户消息
  → [3] 查询改写 (QueryRewriter, 可选)
  → [4] 生成稠密嵌入 (EmbeddingProvider.embed_single)
  → [5] 生成稀疏嵌入 (SparseEmbeddingProvider.embed_single, BM25 分词)
  → [6] 混合检索 (VectorStore.search_chunks, 稠密+稀疏, RRF 融合)
  → [7] 相似度阈值过滤 (可选)
  → [8] 组装上下文 (文本 + 关联图片 URL) + 生成答案 (AnswerGenerator)
  → [9] 提取引用 + 汇总图片
  → [10] 添加助手消息
  → [11] 持久化会话
  → 返回 QueryResult (answer + citations + images)
```

### 混合检索架构

```
                     用户查询
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
    EmbeddingProvider    SparseEmbeddingProvider
    (text-embedding-v4)  (jieba 分词 → BM25)
              │                   │
              ▼                   ▼
         稠密向量            稀疏向量
         (1024维)          ({token: weight})
              │                   │
              └─────────┬─────────┘
                        ▼
               VectorStore.search_chunks
               (Milvus Hybrid Search)
               ANN 搜索 + BM25 搜索
                        │
                        ▼
                  RRF 融合排序
                        │
                        ▼
                  top-k 结果
```

### Milvus Collection Schema (devmind_knowledge)

```python
# 新 collection 需要同时包含稠密和稀疏向量字段
fields = [
    FieldSchema(name="vector_id", dtype=VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="chunk_id", dtype=VARCHAR, max_length=64),
    FieldSchema(name="document_id", dtype=VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=FLOAT_VECTOR, dim=1024),       # 稠密向量
    FieldSchema(name="sparse_embedding", dtype=SPARSE_FLOAT_VECTOR),    # 稀疏向量 (BM25)
    FieldSchema(name="content", dtype=VARCHAR, max_length=8000),
    FieldSchema(name="doc_type", dtype=VARCHAR, max_length=32),
    FieldSchema(name="metadata", dtype=JSON),
]

# 稠密索引: IVF_FLAT, COSINE
# 稀疏索引: SPARSE_INVERTED_INDEX (BM25 内置)
# 混合检索: HNSW + SPARSE, ranker=RRF
```

---

## 9. 数据库 (PostgreSQL)

### 9.1 概述

PostgreSQL 全面替换 SQLite，使用 **SQLAlchemy 2.0** (sync) 作为 ORM 层。

现有 `src/data/database/database.py` 基于 `sqlite3` + 原始 SQL，需重写为 SQLAlchemy 模型。

### 9.2 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 驱动 | `psycopg2` (或 `psycopg[binary]`) | 成熟稳定，SQLAlchemy 一等支持 |
| ORM | SQLAlchemy 2.0 (Declarative) | 类型安全，支持迁移，连接池 |
| 迁移 | Alembic (可选，Phase 2+) | 数据库版本管理 |

### 9.3 数据库模型 (SQLAlchemy)

```python
# infrastructure/database/models.py

from sqlalchemy import String, Text, Integer, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(255))
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    upload_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_modified: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="processing")
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    chunks: Mapped[list["DocumentChunkModel"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_id: Mapped[str | None] = mapped_column(String(64))
    page_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    document: Mapped["DocumentModel"] = relationship(back_populates="chunks")


class ConversationModel(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    messages: Mapped[list["MessageModel"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan",
        order_by="MessageModel.created_at"
    )


class MessageModel(Base):
    __tablename__ = "conversation_messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    conversation: Mapped["ConversationModel"] = relationship(back_populates="messages")


# 现有表也需要迁移: news_articles, extracted_events, predictions,
# prediction_outcomes, historical_events, stock_prices
```

### 9.4 会话管理

```python
# infrastructure/database/session.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from devmind.config import get_settings


class DatabaseSessionManager:
    """数据库会话管理器。"""

    def __init__(self) -> None:
        settings = get_settings()
        self._engine = create_engine(
            settings.postgres_url,
            pool_size=settings.postgres_pool_size,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)

    def get_session(self) -> Session:
        return self._session_factory()

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    def close(self) -> None:
        self._engine.dispose()
```

### 9.5 上下文窗口
- 默认保留最近 10 轮 (20 条消息)
- 可配置: `qa_context_turns`
- 更早的消息保留在数据库中供历史浏览，不发送给 LLM

---

## 10. 配置新增

### .env 实际配置

```env
# PostgreSQL
POSTGRESQL_USERNAME=yanheng
POSTGRESQL_PASSWORD=123456
POSTGRESQL_POSTGRES_PASSWORD=yanheng
POSTGRESQL_MAX_CONNECTIONS=100

# MinIO
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=miniosecret

# LLM
LLM_API_KEY=sk-901f743555d246408a9492dc96d57caa
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-max

# Embedding
EMBEDDING_API_KEY=sk-901f743555d246408a9492dc96d57caa
EMBEDDING_MODEL=text-embedding-v4

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### Settings 类新增字段

```python
# PostgreSQL (替换 SQLite)
postgres_host: str = "localhost"
postgres_port: int = 5432
postgres_username: str = ""
postgres_password: str = ""
postgres_database: str = "devmind"
postgres_pool_size: int = 10

@property
def postgres_url(self) -> str:
    return f"postgresql://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

# Knowledge QA
qa_context_turns: int = 10
qa_top_k: int = 5
qa_min_score: float = 0.3
qa_max_conversations: int = 100
qa_web_timeout: int = 30
qa_web_max_length: int = 100000
qa_collection_name: str = "devmind_knowledge"
qa_enable_query_rewrite: bool = True
qa_generate_image_description: bool = True
qa_enable_hybrid_search: bool = True
qa_bm25_tokenizer: str = "jieba"

# MinIO
minio_endpoint: str = "localhost:9000"
minio_access_key: str = "minio"
minio_secret_key: str = "miniosecret"
minio_bucket: str = "devmind-images"
minio_secure: bool = False

# 图片处理
qa_image_max_size: int = 10 * 1024 * 1024
qa_image_max_dimensions: int = 4096
qa_image_formats: list[str] = ["png", "jpg", "jpeg", "gif", "webp", "svg"]
```

新增 `get_qa_config()` 和 `get_minio_config()` 方法。
移除 `db_path` (SQLite) 字段。

---

## 11. 接口层

### CLI 命令

通过 `add_knowledgeqa_commands(subparsers)` 注册到现有 CLI:

```
devmind qa "问题"                          # 单轮问答
devmind qa --session <id> "追问"            # 多轮对话
devmind qa --top-k 10 "问题"               # 指定检索数量
devmind upload-url <url>                   # 抓取网页入库
devmind qa-history                         # 列出会话
devmind qa-history --session <id>          # 查看会话历史
devmind qa-delete --session <id>           # 删除会话
```

### Gradio Web UI

- 左侧: 知识库管理 (上传文件/URL、文档列表、删除)
- 右侧: 聊天界面 (会话选择器、消息流、输入框、图片展示)
- 使用 `gr.Chatbot` + `gr.State` 管理会话状态
- 图片通过 Markdown 格式嵌入聊天消息展示

---

## 12. 关键设计决策

| 决策 | 理由 |
|------|------|
| 独立 Milvus 集合 (`devmind_knowledge`) | 与现有数据隔离，互不影响 |
| Protocol 而非 ABC | 更 Pythonic，支持结构化子类型 |
| 图片描述文本存入 Milvus | 复用现有文本向量检索能力，无需多模态 embedding |
| 图片文件存 MinIO | 对象存储适合非结构化二进制数据，MinIO 兼容 S3 API |
| 策略模式处理文档 | 每种文档类型独立策略 (PDF/DOCX/MD/Web)，分块逻辑各异 |
| 混合检索 (稠密+稀疏) | 语义匹配 + 关键词匹配互补，中文专有名词/术语召回更准确 |
| Adapter 层翻译命名 | 新模块用 snake_case，现有代码保持 camelCase |
| 删除 rag/ 整个目录 | FaqIngestor/FaqRetriever 从未被外部使用，新模块完全替代 |
| 表格双重保留 | 同时保留 Markdown 文本 + 截图图片，兼顾文本检索和视觉展示 |
| 删除 agents/ 整个目录 | chat_model.py 是临时脚本，StockPredictionAgent 未实现 |
| v1 不做重排序 | Milvus COSINE + BM25 已够用，后续可插拔加入 |
| PostgreSQL 替换 SQLite | 更健壮的关系存储，支持连接池、JSON 字段、并发 |
| SQLAlchemy 2.0 ORM | 类型安全，连接池管理，与 DDD 领域模型清晰映射 |
| OCR 暂不引入 | 后续按需加入多模态 LLM (qwen-vl) 和/或 PaddleOCR |

---

## 13. 需修改/删除的现有文件

### 删除

| 文件 | 原因 |
|------|------|
| `src/agents/` (整个目录) | 死代码 |
| `src/split/` (整个目录) | 空模块 |
| `src/rag/` (整个目录) | 被 knowledgeqa 替代 |
| `src/utils/` (整个目录) | 空模块 |
| `tests/split_document.py` | 实验脚本 |
| `tests/split_document_v2.py` | 实验脚本 |
| `tests/embedding.py` | 调试脚本 |
| `tests/milvus_study.py` | 损坏原型 |

### 修改

| 文件 | 修改内容 |
|------|----------|
| `src/models/__init__.py` | 修复 import 路径 `from models.document` → `from devmind.models.document` |
| `src/models/base.py` | 修复 import 路径 `from models.types` → `from devmind.models.types` |
| `src/cli.py` | 删除 StockPredictionAgent import 及相关引用；注册 qa 子命令 |
| `src/config/settings.py` | 删除 `news_sources_config` 和 `db_path` 字段；添加 PostgreSQL + QA + MinIO 配置 |
| `src/data/database/database.py` | **重写**: sqlite3 → SQLAlchemy 2.0 ORM，所有表模型迁移到 PostgreSQL |
| `pyproject.toml` | 添加 `gradio`, `minio`, `pdfplumber`, `Pillow`, `jieba`, `sqlalchemy`, `psycopg[binary]`, `openai` 依赖；移除 sqlite3 相关；确认 `pymilvus>=2.4.0` |

---

## 14. 实施顺序

### Phase 0 — 代码清理
1. 删除死代码 (agents/, split/, rag/, utils/, 4个测试脚本)
2. 修复 import bug (models/__init__.py, models/base.py)
3. 修复 cli.py (删除 StockPredictionAgent import)
4. 清理 settings.py (删除 news_sources_config)

### Phase 1 — 领域层 + 基础设施适配 + 数据库迁移
5. `src/config/settings.py` — 新增 PostgreSQL + QA + MinIO 配置，移除 SQLite
6. `src/knowledgeqa/infrastructure/database/` — SQLAlchemy 模型 (Document, DocumentChunk, Conversation, Message) + 会话管理器
7. 重写 `src/data/database/database.py` — sqlite3 → SQLAlchemy (现有 news/prediction/market 表迁移)
8. `src/knowledgeqa/domain/` — models.py, protocols.py, exceptions.py
9. `src/knowledgeqa/infrastructure/adapters.py` — 包装现有组件
10. `src/knowledgeqa/infrastructure/mock_adapters.py` — Mock 实现

### Phase 2 — 领域服务 + 文档策略 + 新基础设施
11. `infrastructure/strategies/` — base.py, pdf_strategy.py, docx_strategy.py, markdown_strategy.py, web_strategy.py
12. `infrastructure/sparse_embedding.py` — JiebaBM25Embedding (jieba 分词 → BM25 稀疏向量) + Mock
13. `infrastructure/llm_client.py` — DashScopeLlmClient (OpenAI SDK) + Mock
14. `infrastructure/web_loader.py` — WebContentLoader + Mock (含图片提取)
15. `infrastructure/image_storage.py` — MinIO 客户端 + Mock
16. `domain/services.py` — QueryRewriter, RetrievalOrchestrator (混合检索), AnswerGenerator, ImageProcessor

### Phase 3 — 应用服务
17. `application/conversation_service.py`
18. `application/ingest_service.py` (含图片提取+上传)
19. `application/query_service.py` (含图片召回)
20. `application/knowledge_service.py`

### Phase 4 — 接口层
21. `interface/cli.py` — CLI 命令
22. `interface/web_ui.py` — Gradio Web UI
23. 修改 `src/cli.py` 注册子命令

### Phase 5 — 测试
24. `tests/knowledgeqa/` — 各层单元测试

---

## 15. 验证方式

```bash
# 前置条件: 确保 PostgreSQL, Milvus, MinIO 服务运行
docker compose up -d

# 单元测试
pytest tests/knowledgeqa/ -v

# CLI 问答
devmind qa "什么是Milvus?"

# CLI 多轮对话
devmind qa --session new "介绍一下向量数据库"
devmind qa --session <id> "它有哪些索引类型?"

# CLI 网页入库 (含图片)
devmind upload-url https://milvus.io/docs/overview.md

# CLI 文件上传 (含图片提取)
devmind upload-doc document_with_images.pdf

# Web UI
devmind qa-web   # 启动 Gradio 界面

# Mock 模式测试
devmind --mock qa "测试问题"

# MinIO 连通性
devmind qa-health  # 检查 Milvus + MinIO + LLM 连接状态
```

---

## 16. 补充说明

### 16.1 .doc 旧格式处理

测试文档包含 `2.杆塔明细表.doc`（旧 Word 二进制格式）。`python-docx` 仅支持 `.docx`。

**策略**: 不原生支持 `.doc`。CLI 收到 `.doc` 文件时提示用户先转换为 `.docx`（LibreOffice: `soffice --convert-to docx file.doc`）。未来可选集成 LibreOffice headless 自动转换。

### 16.2 大图片处理

测试文档 `公式.png` 为 50MB，超过 `qa_image_max_size` 默认值 10MB。

**策略**: 图片超限时，自动压缩/缩放后重试。若压缩后仍超限则跳过并记录警告。

### 16.3 Docker Compose

项目根目录需新建 `docker-compose.yml`，包含 PostgreSQL、MinIO、Milvus 三个服务。参考配置:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: yanheng
      POSTGRES_PASSWORD: "123456"
      POSTGRES_DB: devmind
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: miniosecret
    ports: ["9000:9000", "9001:9001"]
    volumes: [miniodata:/data]

  milvus:
    image: milvusdb/milvus:v2.4-latest
    command: milvus run standalone
    ports: ["19530:19530"]
    volumes: [milvusdata:/var/lib/milvus]

volumes:
  pgdata:
  miniodata:
  milvusdata:
```

### 16.4 VectorStore Adapter 说明

现有 `DocumentVectorStore` 不支持稀疏向量和混合检索。Adapter 不是简单的方法名翻译，需要:

1. 创建新的 Milvus collection（含 `SPARSE_FLOAT_VECTOR` 字段）
2. `insert_chunk` 同时写入稠密+稀疏向量
3. `search_chunks` 使用 `HybridSearchRequest` + `RRFRanker`
4. Mock 实现也需支持双路检索和 RRF 融合

### 16.5 数据库 ORM 模型位置

统一放在 `src/knowledgeqa/infrastructure/database/models.py`。现有股票预测相关的表模型（news_articles, predictions 等）也迁移到此处。`src/data/database/` 目录下的 `database.py` 重写为调用 SQLAlchemy Session 的薄封装层，保持 `PredictionDatabase` 类接口不变。

### 16.6 数据库 Session 生命周期

使用 **context manager** 模式:

```python
# 用法
with db_manager.session() as session:
    repo = DocumentRepository(session)
    repo.insert_document(doc)
    # 自动 commit/rollback
```

`DatabaseSessionManager` 提供 `session()` 方法返回 context manager，确保事务完整性。应用服务层通过构造函数注入 `DatabaseSessionManager`。
