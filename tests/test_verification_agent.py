import pytest
from langchain_core.documents import Document

from src.agents.verfication_agent import (
    VerificationAgent,
    VerificationResult
)


# Fake async chain
class FakeChain:

    async def ainvoke(self, payload):

        return VerificationResult(

            supported="YES",

            unsupported_claims=[],

            contradictions=[],

            relevant="YES",

            additional_details="All claims supported"
        )


class FakeFailChain:

    async def ainvoke(self, payload):

        raise Exception("LLM failure")


@pytest.fixture
def agent():

    agent = VerificationAgent()

    # inject fake chain
    agent.chain = FakeChain()

    return agent


@pytest.fixture
def sample_docs():

    return [

        Document(
            page_content="Paris is the capital of France."
        ),

        Document(
            page_content="France is in Europe."
        )
    ]


@pytest.mark.asyncio
async def test_verification_success(
    agent,
    sample_docs
):

    result = await agent.check(

        answer="Paris is the capital of France.",

        documents=sample_docs
    )

    assert result["supported"] == "YES"

    assert result["relevant"] == "YES"

    assert result["unsupported_claims"] == []

    assert "verification_report" in result


@pytest.mark.asyncio
async def test_empty_documents(agent):

    result = await agent.check(

        answer="Some answer",

        documents=[]
    )

    assert result["context_used"] == ""


@pytest.mark.asyncio
async def test_document_compression(agent):

    docs = [

        Document(
            page_content="A"*1000
        )
    ]

    context = agent._compress_documents(docs)

    assert len(context) <= 410  # 400 + "..."


@pytest.mark.asyncio
async def test_verification_failure():

    agent = VerificationAgent()

    agent.chain = FakeFailChain()

    docs = [

        Document(
            page_content="Test content"
        )
    ]

    with pytest.raises(RuntimeError):

        await agent.check(

            answer="Test",

            documents=docs
        )


@pytest.mark.asyncio
async def test_max_docs_limit(agent):

    docs = [

        Document(
            page_content=f"Doc {i}"
        )

        for i in range(10)
    ]

    context = agent._compress_documents(docs)

    # MAX_DOCS = 4
    assert context.count("Doc") <= 4