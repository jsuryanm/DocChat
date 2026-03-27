from src.agents.reflexion_agent import ReflexionAgent


def test_reflexion_accept():

    agent = ReflexionAgent()

    decision = agent.decide(
        grounded=True,
        quality="HIGH",
        retries=0,
        max_retries=2
    )

    assert decision == "accept"


def test_reflexion_retry():

    agent = ReflexionAgent()

    decision = agent.decide(
        grounded=False,
        quality="LOW",
        retries=0,
        max_retries=2
    )

    assert decision == "retry"


def test_reflexion_stop():

    agent = ReflexionAgent()

    decision = agent.decide(
        grounded=False,
        quality="LOW",
        retries=3,
        max_retries=2
    )

    assert decision == "stop"