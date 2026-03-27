import pytest


from src.agents.workflow import AgentWorkflow


class FakeRetriever:

    async def ainvoke(self,query):
        return []


class FakeBuilder:

    async def rerank(self,query,docs):
        return docs


class FakeResearch:

    async def generate(self,question,docs):

        return {
            "draft_answer":"test answer",
            "confidence":"HIGH"
        }


class FakeVerify:

    async def check(self,answer,docs):

        return {
            "supported":"YES"
        }


class FakeGrader:

    async def grade(self,q,a):

        return {
            "quality":"HIGH"
        }


class FakeRewrite:

    async def rewrite(self,q):
        return q


class FakeReflect:

    def decide(self,grounded,quality,retries,max_retries):

        return "accept"


@pytest.mark.asyncio
async def test_workflow_runs():

    workflow = AgentWorkflow(
        FakeRetriever(),
        FakeBuilder()
    )

    # inject fake agents
    workflow.research = FakeResearch()
    workflow.verify = FakeVerify()
    workflow.grade = FakeGrader()
    workflow.rewrite = FakeRewrite()
    workflow.reflect = FakeReflect()

    result = await workflow.run(
        "test question"
    )

    assert result["final_answer"] == "test answer"
    assert result["retries"] == 0