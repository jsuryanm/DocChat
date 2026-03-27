import pytest

from src.agents.answer_grader import AnswerGrader


@pytest.mark.asyncio
async def test_answer_grade():

    grader = AnswerGrader()

    result = await grader.grade(
        "What is python?",
        "Python is language"
    )

    assert "quality" in result