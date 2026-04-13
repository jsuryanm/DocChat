from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from openai import LengthFinishReasonError
except ImportError:
    LengthFinishReasonError = None  

from src.config.settings import settings
from src.custom_logger.logger import logger

_VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}


class RelevanceResult(BaseModel):
    label: str = Field(description="Exactly one of: CAN_ANSWER, PARTIAL, NO_MATCH")


class RelevanceChecker:

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict relevance classifier.\n"
            "Given a question and some document passages, decide whether the passages "
            "contain enough information to answer the question.\n\n"
            "Rules:\n"
            "- CAN_ANSWER: passages directly address the question\n"
            "- PARTIAL: passages are related but incomplete\n"
            "- NO_MATCH: passages are unrelated or empty\n\n"
            "Return ONLY the label field. Do not explain."
        ),
        (
            "human",
            "Question: {question}\n\nPassages:\n{passages}"
        )
    ])

    def __init__(self):
        llm = ChatOpenAI(
            model=settings.RELEVANCY_MODEL,
            temperature=0,
            max_tokens=settings.RELEVANCE_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
            reasoning_effort="low"
        )

        self.chain = self.PROMPT_TEMPLATE | llm.with_structured_output(RelevanceResult)

    async def check(
        self,
        question: str,
        documents: List[Document],
        k: int = 3,
    ) -> str:
        """Classify relevance of already-retrieved documents against the question.

        Parameters
        ----------
        question : str
            The (rewritten) user question.
        documents : List[Document]
            The reranked documents produced by the rerank node.
        k : int
            How many of the top documents to inspect.

        Returns
        -------
        str
            One of: CAN_ANSWER | PARTIAL | NO_MATCH
        """
        if not documents:
            logger.warning("No documents passed to relevance check — defaulting to NO_MATCH")
            return "NO_MATCH"

        passages = "\n\n".join(
            doc.page_content[:500] for doc in documents[:k]
        )

        try:
            result: RelevanceResult = await self.chain.ainvoke({
                "question": question,
                "passages": passages,
            })
            label = result.label.strip().upper()
            logger.debug(f"RelevanceChecker structured output: {repr(label)}")

        except Exception as e:
            if LengthFinishReasonError is not None and isinstance(e, LengthFinishReasonError):
                logger.warning(
                    "RelevanceChecker: token limit reached before structured output "
                    "was produced — defaulting to PARTIAL to preserve pipeline progress. "
                    "Consider raising RELEVANCE_MAX_TOKENS in settings."
                )
                return "PARTIAL"

            logger.warning(f"RelevanceChecker LLM call failed: {e}")
            return "NO_MATCH"

        return self._parse_label(label)

    @staticmethod
    def _parse_label(raw: str) -> str:
        if not raw:
            logger.warning("RelevanceChecker: model returned empty label")
            return "NO_MATCH"

        for token in raw.upper().split():
            cleaned = token.strip(".:,;\"'")
            if cleaned in _VALID_LABELS:
                return cleaned

        logger.warning(
            f"RelevanceChecker: unrecognised label '{raw}' — defaulting to NO_MATCH"
        )
        return "NO_MATCH"