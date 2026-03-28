import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever

from src.config.settings import settings
from src.custom_logger.logger import logger

_VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}


class RelevanceChecker:

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict relevance classifier.\n"
            "Return ONLY one of these exact words — nothing else:\n"
            "CAN_ANSWER  PARTIAL  NO_MATCH\n"
            "Do not add punctuation, explanation, or any other text."
        ),
        (
            "human",
            "Question: {question}\n\nPassages:\n{passages}\n\nLabel:"
        )
    ])

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.RELEVANCY_MODEL,
            temperature=1,              # reasoning models require temperature=1
            max_tokens=200,             # enough for reasoning budget + output token
            model_kwargs={
                "reasoning_effort": "low"   # minimal reasoning for simple classification
            },
            api_key=settings.OPENAI_API_KEY,
        )

    async def check(
        self,
        question: str,
        retriever: EnsembleRetriever,
        k: int = 3,
    ) -> str:
        """Retrieve top-k chunks and classify relevance.
        Returns: CAN_ANSWER | PARTIAL | NO_MATCH"""

        try:
            top_docs = await asyncio.to_thread(retriever.invoke, question)
        except Exception as e:
            logger.error(f"Retriever failed in relevance check: {e}")
            return "NO_MATCH"

        if not top_docs:
            logger.warning("No documents retrieved — defaulting to NO_MATCH")
            return "NO_MATCH"

        passages = "\n\n".join(
            doc.page_content[:500] for doc in top_docs[:k]
        )

        try:
            messages = self.PROMPT_TEMPLATE.format_messages(
                question=question,
                passages=passages,
            )
            raw_response = await self.llm.ainvoke(messages)

            text = raw_response.content

            if isinstance(text, list):
                text = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )

            logger.debug(f"RelevanceChecker content: {repr(text)}")

        except Exception as e:
            logger.warning(f"RelevanceChecker LLM call failed: {e}")
            return "NO_MATCH"

        return self._parse_label(text)

    @staticmethod
    def _parse_label(raw: str) -> str:
        if not raw:
            logger.warning("RelevanceChecker: model returned empty content")
            return "NO_MATCH"

        for token in raw.upper().split():
            cleaned = token.strip(".:,;\"'")
            if cleaned in _VALID_LABELS:
                return cleaned

        logger.warning(
            f"RelevanceChecker: unrecognized output '{raw}' — defaulting to NO_MATCH"
        )
        return "NO_MATCH"