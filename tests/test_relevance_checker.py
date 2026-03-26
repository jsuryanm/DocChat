import pytest

from langchain_core.documents import Document

from src.agents.relevance_checker import RelevanceChecker


class FakeRetriever:
    """Mock retriever for testing"""

    def invoke(self, query: str):
        return [
            Document(page_content="GPT-4o is an OpenAI multimodal model"),
            Document(page_content="LangChain helps build RAG systems"),
            Document(page_content="BM25 is keyword retrieval")
        ]


@pytest.fixture
def checker():
    return RelevanceChecker()


@pytest.fixture
def retriever():
    return FakeRetriever()


@pytest.mark.asyncio
async def test_relevance_can_answer(checker, retriever):

    result = await checker.check("What is GPT-4o?",
                                 retriever)

    assert result in {"CAN_ANSWER","PARTIAL","NO_MATCH"}


@pytest.mark.asyncio
async def test_relevance_no_docs(checker):

    class EmptyRetriever:
        def invoke(self, query):
            return []

    result = await checker.check("test question",
                                 EmptyRetriever())

    assert result == "NO_MATCH"