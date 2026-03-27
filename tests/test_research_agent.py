import pytest
from langchain_core.documents import Document

from src.agents.research_agent import ResearchAgent


@pytest.mark.asyncio
async def test_research_agent_generate():

    agent = ResearchAgent()

    docs = [
        Document(
            page_content="Python is a programming language.",
            metadata={}
        )
    ]

    result = await agent.generate(
        "What is python?",
        docs
    )

    assert "draft_answer" in result
    assert "confidence" in result


@pytest.mark.asyncio
async def test_research_no_docs():

    agent = ResearchAgent()

    result = await agent.generate(
        "test",
        []
    )

    assert isinstance(result,dict)