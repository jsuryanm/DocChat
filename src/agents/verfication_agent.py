from typing import List, Dict
from pydantic import BaseModel, Field
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
)

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# FIX: import LengthFinishReasonError so we can exclude it from the retry
# predicate.  Retrying a token-limit error is pointless — the same prompt
# will hit the same ceiling every time, burning three attempts and ~90 s of
# wall-clock time before surfacing a RetryError to the caller.
try:
    from openai import LengthFinishReasonError
except ImportError:
    LengthFinishReasonError = None  # type: ignore[assignment,misc]

from src.config.settings import settings
from src.custom_logger.logger import logger


class VerificationResult(BaseModel):
    supported: str = Field(description="YES or NO")
    unsupported_claims: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    relevant: str = Field(description="YES or NO")
    additional_details: str = Field(default="")


_SYSTEM_PROMPT = """You are a strict fact verification agent.

Verify whether the answer is supported ONLY by the provided context.

Rules:

Only mark Supported = YES if fully grounded.

Unsupported claims must be explicitly listed.

If nothing unsupported → return empty list.

Return only structured output.
"""

# Sentinel returned when the LLM call itself fails (not when the answer is
# wrong).  The workflow uses supported="UNKNOWN" to set verification_failed=True
# in state, which prevents a pointless content-quality retry loop.
_FAILURE_RESPONSE = {
    "supported": "UNKNOWN",
    "unsupported_claims": [],
    "contradictions": [],
    "relevant": "UNKNOWN",
    "additional_details": "verification failed",
    "verification_report": "verification failed",
    "context_used": "",
}


class VerificationAgent:
    MAX_DOCS = 3
    MAX_CHARS = 250
    TIMEOUT = 30

    def __init__(self, llm=None):
        llm = llm or ChatOpenAI(
            model=settings.VERIFY_MODEL,
            temperature=0,
            max_tokens=settings.VERIFY_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
            reasoning_effort="low"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                """Answer:
                {answer}

                Context:
                {context}

                Perform verification.
                """,
            ),
        ])

        self.chain = prompt | llm.with_structured_output(VerificationResult)

    # FIX: exclude LengthFinishReasonError from the retry predicate.
    # Previously the bare @retry caught every exception including token-limit
    # errors, causing three identical failing attempts (each burning TIMEOUT
    # seconds) before surfacing a RetryError.  Now token-limit errors are
    # re-raised immediately so the caller can handle them in one place.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=(
            retry_if_not_exception_type(LengthFinishReasonError)
            if LengthFinishReasonError is not None
            else retry_if_not_exception_type(Exception.__class__)
        ),
        reraise=True,
    )
    async def _call_llm(self, payload: Dict) -> VerificationResult:
        return await asyncio.wait_for(
            self.chain.ainvoke(payload), timeout=self.TIMEOUT
        )

    async def check(self, answer: str, documents: List[Document]) -> Dict:
        context = self._compress_documents(documents)

        try:
            result = await self._call_llm({"answer": answer, "context": context})

        except Exception as e:
            # FIX: distinguish token-limit failures from other errors so
            # callers can act accordingly (see workflow._verify).
            if LengthFinishReasonError is not None and isinstance(e, LengthFinishReasonError):
                logger.warning(
                    "VerificationAgent: token limit reached — returning UNKNOWN. "
                    "Consider raising VERIFY_MAX_TOKENS in settings."
                )
            else:
                logger.error(f"VerificationAgent error: {e}")

            response = dict(_FAILURE_RESPONSE)
            response["context_used"] = context
            return response

        report = self._format(result)
        logger.info("Verification complete")

        return {
            "supported": result.supported,
            "unsupported_claims": result.unsupported_claims,
            "contradictions": result.contradictions,
            "relevant": result.relevant,
            "additional_details": result.additional_details,
            "verification_report": report,
            "context_used": context,
        }

    def _compress_documents(self, docs: List[Document]) -> str:
        trimmed = []
        for doc in docs[: self.MAX_DOCS]:
            text = doc.page_content.strip()
            if len(text) > self.MAX_CHARS:
                text = text[: self.MAX_CHARS] + "..."
            trimmed.append(text)
        return "\n\n".join(trimmed)

    @staticmethod
    def _format(result: VerificationResult) -> str:
        unsupported = ", ".join(result.unsupported_claims) or "None"
        contradictions = ", ".join(result.contradictions) or "None"
        details = result.additional_details or "None"

        return (
            f"**Supported:** {result.supported}\n"
            f"**Unsupported Claims:** {unsupported}\n"
            f"**Contradictions:** {contradictions}\n"
            f"**Relevant:** {result.relevant}\n"
            f"**Additional Details:** {details}\n"
        )