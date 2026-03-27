import pytest

from langchain_core.documents import Document

from src.agents.relevance_checker import RelevanceChecker


class FakeRetriever:

    def invoke(self,query):

        return [
            Document(
                page_content="Python is programming language",
                metadata={}
            )
        ]


@pytest.mark.asyncio
async def test_relevance_checker():

    checker = RelevanceChecker()

    label = await checker.check(
        "What is python?",
        FakeRetriever()
    )

    assert label in [
        "CAN_ANSWER",
        "PARTIAL",
        "NO_MATCH"
    ]