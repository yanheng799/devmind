"""FAQ retrieval and answer generation."""

import logging
from typing import Any

from openai import OpenAI

from devmind.config import get_settings
from devmind.data.vectorstore.dashscope_embedding import DashScopeEmbeddingModel
from devmind.data.vectorstore.document_store import DocumentVectorStore


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Human: You are an AI assistant. You are able to find answers "
    "to the questions from the contextual passage snippets provided."
)


class FaqRetriever:
    """Retrieve answers from the FAQ knowledge base using RAG.

    Supports multiple queries without re-vectorizing data.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        top_k: int | None = None,
    ) -> None:
        """Initialize the FAQ retriever.

        Args:
            collection_name: Milvus collection name (from settings if None)
            top_k: Number of chunks to retrieve (from settings if None)
        """
        settings = get_settings()
        rag_config = settings.get_rag_config()

        self.collection_name = collection_name or rag_config["faq_collection_name"]
        self.top_k = top_k or rag_config["faq_top_k"]

        self._embedding_model = DashScopeEmbeddingModel()
        self._store = DocumentVectorStore(collection_name=self.collection_name)

    def ask(self, question: str) -> str:
        """Ask a question and get a generated answer.

        Embeds the question, searches the store, then uses
        the LLM to generate an answer from retrieved context.

        Args:
            question: The question to ask.

        Returns:
            Generated answer string.
        """
        context_chunks = self.retrieveContext(question)
        if not context_chunks:
            return "No relevant information found in the knowledge base."

        system_prompt, user_prompt = self._buildPrompt(question, context_chunks)
        return self._generateAnswer(system_prompt, user_prompt)

    def retrieveContext(self, question: str) -> list[dict[str, Any]]:
        """Retrieve relevant context chunks for a question.

        Args:
            question: The question to search for.

        Returns:
            List of dicts with 'content' and 'score' keys.
        """
        query_embedding = self._embedding_model.embed_single(question)
        return self._store.searchChunks(
            query_embedding=query_embedding,
            top_k=self.top_k,
            doc_type="faq",
        )

    def _buildPrompt(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Build system and user prompts for the LLM.

        Args:
            question: The user question.
            context_chunks: Retrieved context chunks.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        context_text = "\n".join(chunk["content"] for chunk in context_chunks)

        user_prompt = (
            f"Use the following pieces of information enclosed in <context> tags "
            f"to provide an answer to the question enclosed in <question> tags.\n"
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{question}\n</question>"
        )

        return SYSTEM_PROMPT, user_prompt

    def _generateAnswer(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM to generate an answer.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.

        Returns:
            Generated answer string.
        """
        settings = get_settings()
        llm_config = settings.get_llm_config()

        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
        )

        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            timeout=llm_config["timeout"],
        )

        return response.choices[0].message.content

    def close(self) -> None:
        """Clean up resources."""
        self._embedding_model.close()
        self._store.close()
