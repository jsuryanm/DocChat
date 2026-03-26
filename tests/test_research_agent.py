import pytest
from langchain_core.documents import Document

from src.agents.research_agent import ResearchAgent


class FakeResult:

    def __init__(self):
        self.answer = "Test answer"
        self.confidence = "HIGH"
        self.missing_information = "None"


class FakeChain:

    async def ainvoke(self,payload):
        return FakeResult()

    async def astream(self,payload):

        yield "chunk1"
        yield "chunk2"


@pytest.fixture
def sample_docs():

    return [

        Document(
            page_content="Python is a programming language",
            metadata={"score":0.9}
        ),

        Document(
            page_content="Used in AI",
            metadata={"score":0.8}
        )

    ]


@pytest.fixture
def agent():

    agent = ResearchAgent(llm=None)

    # override chain instead of LLM
    agent.chain = FakeChain()

    return agent


@pytest.mark.asyncio
async def test_generate_success(agent,sample_docs):

    result = await agent.generate(
        question="What is python?",
        documents=sample_docs
    )

    assert result["draft_answer"] == "Test answer"
    assert result["confidence"] == "HIGH"
    assert result["doc_count"] == 2


@pytest.mark.asyncio
async def test_generate_no_match(agent,sample_docs):

    result = await agent.generate(
        question="test",
        documents=sample_docs,
        relevance="NO_MATCH"
    )

    assert result["confidence"] == "LOW"
    assert result["doc_count"] == 0


@pytest.mark.asyncio
async def test_generate_empty_docs(agent):

    result = await agent.generate(
        question="test",
        documents=[]
    )

    assert result["doc_count"] == 0
    assert result["context_used"] == ""


@pytest.mark.asyncio
async def test_stream(agent,sample_docs):

    chunks = []

    async for chunk in agent.stream(
        question="test",
        documents=sample_docs
    ):
        chunks.append(chunk)

    assert chunks == ["chunk1","chunk2"]


def test_compress_documents(agent):

    docs = [

        Document(
            page_content="A"*600,
            metadata={"score":0.5}
        )

    ]

    context = agent._compress_documents(docs)

    assert context.endswith("...")
    assert len(context) <= 503


@pytest.mark.asyncio
async def test_llm_failure(sample_docs):

    class FailingChain:

        async def ainvoke(self,payload):
            raise Exception("LLM failure")

    agent = ResearchAgent(llm=None)

    agent.chain = FailingChain()

    with pytest.raises(RuntimeError):

        await agent.generate(
            question="test",
            documents=sample_docs
        )