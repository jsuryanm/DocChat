import pytest
from langchain_core.documents import Document

from src.agents.verfication_agent import (
    VerificationAgent,
    VerificationResult
)


class FakeChain:

    async def ainvoke(self,payload):

        return VerificationResult(
            supported="YES",
            unsupported_claims=[],
            contradictions=[],
            relevant="YES",
            additional_details="All claims supported"
        )


@pytest.mark.asyncio
async def test_verification_agent():

    agent = VerificationAgent()

    # CRITICAL FIX
    agent.chain = FakeChain()

    docs = [
        Document(
            page_content="Paris is capital of France",
            metadata={}
        )
    ]

    result = await agent.check(
        "Paris is capital of France",
        docs
    )

    assert result["supported"] == "YES"
    assert result["relevant"] == "YES"