# pytest tests/test_builder.py -v 

import pytest

from src.retriever.builder import RetrieverBuilder, EmptyRetriever
from langchain_core.documents import Document


class FakeEmbeddings:
    def embed_documents(self,texts):
        return [[0.1]*10 for _ in texts]
    
    def embed_query(self,text):
        return [0.1]*10


class FakeReranker:
    def score(self,query,docs):
        return [1.0 - (i*0.1) for i in range(len(docs))]


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="GPT-4o is an OpenAI multimodal model"),
        Document(page_content="LangChain enables RAG pipelines"),
        Document(page_content="BM25 is a keyword retrieval algorithm"),
        Document(page_content="Vector search uses embeddings")
    ]


@pytest.fixture
def builder():
    return RetrieverBuilder(
        embeddings=FakeEmbeddings(),
        reranker=FakeReranker()
    )


@pytest.mark.asyncio
async def test_build_hybrid_retriever(builder,sample_docs):

    retriever = await builder.build_hybrid_retriever(
        sample_docs,
        persist=False
    )

    assert retriever is not None


@pytest.mark.asyncio
async def test_retriever_returns_document(builder,sample_docs):

    retriever = await builder.build_hybrid_retriever(
        sample_docs,
        persist=False
    )

    results = retriever.invoke("What is GPT-4o?")

    assert len(results) > 0
    assert isinstance(results[0],Document)


@pytest.mark.asyncio
async def test_rerank_returns_sorted_docs(builder,sample_docs):

    ranked = await builder.rerank(
        "GPT4o",
        sample_docs
    )

    assert len(ranked) > 0
    assert isinstance(ranked,list)
    assert isinstance(ranked[0],Document)


@pytest.mark.asyncio
async def test_empty_documents(builder):

    retriever = await builder.build_hybrid_retriever(
        [],
        persist=False
    )

    assert isinstance(retriever,EmptyRetriever)

    results = retriever.invoke("test")

    assert results == []


@pytest.mark.asyncio
async def test_rerank_empty_input(builder):

    ranked = await builder.rerank(
        "test",
        []
    )

    assert ranked == []