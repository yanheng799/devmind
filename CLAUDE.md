# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DevMind 是基于新闻的 A 股股价预测 Agent + 企业级知识问答系统，自用研究辅助工具，本地运行。

- **知识问答**: 文档摄入（PDF/DOCX/MD/网页）→ 分块 → 向量化 → 混合检索 → LLM 生成答案（含图片）

## 常用命令

```bash
# 安装
pip install -e ".[dev]"

# 测试
pytest                              # 运行全部测试
pytest tests/knowledgeqa/           # 知识问答模块测试
pytest -k "test_clean_text"         # 运行单个测试

# 代码质量
ruff check src tests                # Lint
black src tests                     # 格式化
mypy src                            # 类型检查

# CLI — 知识问答
devmind qa "问题"                    # 单轮问答
devmind qa --session <id> "追问"      # 多轮对话
devmind qa --top-k 10 "问题"         # 指定检索数量
devmind upload-url <url>            # 抓取网页入库
devmind upload-doc <path>           # 上传文档
devmind qa-history                   # 列出会话
devmind qa-history --session <id>    # 查看会话历史
devmind qa-delete --session <id>     # 删除会话
devmind qa-web                       # 启动 Gradio Web UI

```

## 架构

### DDD 分层（knowledgeqa 模块）

```
src/knowledgeqa/
├── domain/              # 领域层: 实体、值对象、Protocol 接口、领域服务
├── application/         # 应用层: 用例编排 (IngestService, QueryService, ConversationService)
├── infrastructure/      # 基础设施层: 适配器、策略、外部服务实现
│   ├── strategies/      # 文档处理策略 (PDF/DOCX/MD/Web)
│   └── database/        # PostgreSQL ORM 模型 + Repository
└── interface/           # 接口层: CLI + Gradio Web UI
```

### 依赖注入

所有跨层依赖通过 `typing.Protocol` 接口，不直接依赖具体实现。Mock 模式通过替换实现完成。

### 混合检索

稠密向量（text-embedding-v4，语义匹配）+ 稀疏向量（jieba BM25，关键词匹配）→ RRF 融合排序。

### 文档处理策略

每种文档类型有独立的 Strategy：PDF（按页分块+表格截图）、DOCX（按标题分块+内嵌图片）、Markdown（按标题分块+代码块完整）、Web（正文提取+图片下载）。

### Mock 模式

每个核心组件都有对应的 Mock 类，CLI 的 `--mock` 标志一键切换，用于离线开发和测试。

## 编码规范

- 语言: Python ≥3.10，类型注解必须，禁止 `Any`（用 `Unknown` 替代）
- 风格: Black (line-length=100)，Ruff (E,F,I,N,W)
- 错误处理: 不吞异常，必须记录或重新抛出；检查空值和边界条件
- 输入验证: 使用 Pydantic，参数化查询
- 函数: 不超过 30 行，动词命名，单一职责
- 命名: 文件 snake_case，类 PascalCase，变量/函数 snake_case，常量 SCREAMING_SNAKE_CASE

## 任务执行流程

执行每个开发任务时，必须严格遵循以下流程：

### 1. 分析与计划

- 阅读任务描述，结合设计文档 `docs/knowledgeqa-design.md` 和任务列表 `docs/knowledgeqa-tasks.md` 理解完整上下文
- 读取相关现有代码，确认当前状态
- 输出本任务的**详细执行计划**：列出要创建/修改的文件、关键代码变更点、执行顺序
- 明确标注计划中每一步的预期产出

### 2. 按计划执行

- 严格按照输出的计划逐步执行
- 每完成一步，验证该步骤的产出（import 正确、类型正确、逻辑正确）
- 执行过程中发现问题及时记录

### 3. 文档同步（关键）

- 如果执行中发现设计文档有问题（接口定义不合理、遗漏字段、依赖缺失、技术方案不可行等），**必须立即更新以下文档**：
  - `docs/knowledgeqa-design.md` — 更新受影响的设计章节
  - `docs/knowledgeqa-tasks.md` — 更新受影响任务的步骤描述
- 文档更新时在修改处标注日期和原因，确保后续任务能根据文档了解最新设计状态
- 不得以口头说明代替文档更新

### 4. 及时提交

- 每完成一个**有意义的实现单元**（一个完整的类、一个策略、一个服务），立即 git commit
- 不要等整个 Phase 完成才提交，避免大量代码堆积在一次提交中
- 提交信息格式：`feat/fix/refactor: 简要描述`，例如 `feat(knowledgeqa): implement PdfStrategy with image extraction`
- 提交前确保该单元不破坏已有功能（测试通过、Lint 无警告）

### 5. 验证

- 每个任务完成后运行相关测试确认无回归
- 检查 Lint 和格式化是否通过

## 质量红线

提交前必须通过: Lint（零警告）→ 格式化 → 类型检查 → 全部测试

## 关键依赖

- **LLM**: Qwen3-Max via DashScope（OpenAI 兼容接口）
- **Embedding**: DashScope text-embedding-v4（稠密）+ jieba BM25（稀疏）
- **向量数据库**: Milvus 2.4+（Docker 本地，混合检索）
- **关系数据库**: PostgreSQL（SQLAlchemy 2.0 ORM）
- **图片存储**: MinIO（Docker 本地）
- **行情数据**: AKShare
- **Web UI**: Gradio
- **Agent 框架**: LangGraph

## 设计文档

- `docs/knowledgeqa-design.md` — 知识问答系统 DDD 设计文档
- `docs/knowledgeqa-tasks.md` — 开发任务计划
- `docs/test_docs/` — 测试用文档样本
