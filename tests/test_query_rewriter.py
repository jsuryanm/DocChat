import pytest

from src.agents.query_rewriter import QueryRewriter


@pytest.mark.asyncio
async def test_query_rewrite():

    agent = QueryRewriter()

    result = await agent.rewrite(
        "python lang"
    )

    assert isinstance(result,str)