from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState
from src.agents.query_rewriter import QueryRewriter
from src.agents.research_agent import ResearchAgent
from src.agents.verfication_agent import VerificationAgent
from src.agents.answer_grader import AnswerGrader
from src.agents.reflexion_agent import ReflexionAgent
from src.agents.relevance_checker import RelevanceChecker
from src.tools.mcp_client import get_tools
from src.a2a.client import call_remote_agent

from src.config.settings import settings
from src.custom_logger.logger import logger


class AgentWorkflow:
    MAX_RETRIES = 2

    def __init__(self, retriever, builder):
        self.retriever = retriever
        self.builder = builder

        self.research = ResearchAgent()
        self.verify = VerificationAgent()
        self.rewrite = QueryRewriter()
        self.grade = AnswerGrader()
        self.reflect = ReflexionAgent()
        self.relevance = RelevanceChecker()

        self.mcp_tools = get_tools()
        self.tool_node = ToolNode(self.mcp_tools) if self.mcp_tools else None

        self.graph = self._build()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)

        g.add_node("rewrite", self._rewrite)
        g.add_node("retrieve", self._retrieve)
        g.add_node("rerank", self._rerank)
        g.add_node("check_relevance", self._check_relevance)
        g.add_node("research", self._research)
        g.add_node("grade", self._grade)
        g.add_node("verify", self._verify)
        g.add_node("reflect", self._reflect)
        g.add_node("finalize", self._finalize)
        g.add_node("delegate", self._delegate)

        if self.tool_node:
            g.add_node("tools", self.tool_node)

        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "rerank")

        g.add_conditional_edges(
            "rerank",
            self._route_after_rerank,
            {"delegate": "delegate", "check_relevance": "check_relevance"},
        )

        g.add_conditional_edges(
            "check_relevance",
            self._route_after_relevance,
            {"research": "research", "finalize": "finalize"},
        )

        if self.tool_node:
            g.add_conditional_edges(
                "research",
                self._route_after_research,
                {"tools": "tools", "grade": "grade"},
            )
            g.add_edge("tools", "research")
        else:
            g.add_edge("research", "grade")

        g.add_edge("grade", "verify")

        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"accept": "finalize", "retry": "reflect", "stop": "finalize"},
        )

        g.add_edge("reflect", "rewrite")
        g.add_edge("delegate", "grade")
        g.add_edge("finalize", END)

        return g.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def _rewrite(self, state):
        logger.info("Rewriting query")
        query = await self.rewrite.rewrite(state["question"])
        return {"rewritten_question": query, "reasoning_steps": ["Query rewritten"]}

    async def _retrieve(self, state):
        logger.info("Retrieving documents")
        docs = await self.retriever.ainvoke(state["rewritten_question"])
        return {
            "documents": docs,
            "reasoning_steps": [f"Retrieved {len(docs)} documents"],
        }

    async def _rerank(self, state):
        logger.info("Reranking documents")
        reranked = await self.builder.rerank(
            state["rewritten_question"], state["documents"]
        )
        return {"reranked_docs": reranked, "reasoning_steps": ["Documents reranked"]}

    async def _check_relevance(self, state):
        logger.info("Checking relevance")
        label = await self.relevance.check(
            question=state["rewritten_question"],
            documents=state["reranked_docs"],
        )
        logger.info(f"Relevance label: {label}")
        return {
            "relevance_label": label,
            "reasoning_steps": [f"Relevance check: {label}"],
        }

    async def _research(self, state):
        logger.info("Generating answer")
        result = await self.research.generate(
            state["rewritten_question"],
            state["reranked_docs"],
            relevance=state.get("relevance_label"),
        )
        return {
            "draft_answer": result["draft_answer"],
            "confidence": result["confidence"],
            "tool_calls": result.get("tool_calls", []),
            "draft_history": [result["draft_answer"]],
            "reasoning_steps": ["Research agent generated an answer"],
        }

    async def _grade(self, state):
        logger.info("Grading the answer")
        result = await self.grade.grade(state["question"], state["draft_answer"])
        return {
            "answer_quality": result["quality"],
            "reasoning_steps": [f"Answer quality = {result['quality']}"],
        }

    async def _verify(self, state):
        logger.info("Verifying grounding")
        result = await self.verify.check(
            state["draft_answer"], state["reranked_docs"]
        )

        # FIX: distinguish a tool/LLM failure from a genuine "not grounded"
        # result.  VerificationAgent.check() returns supported="UNKNOWN" when
        # the underlying LLM call fails (e.g. LengthFinishReasonError, timeout,
        # network error).  In that case we set verification_failed=True so
        # _route_after_verify can short-circuit to "accept" without triggering
        # a content-quality retry loop that would discard a valid answer.
        supported = result.get("supported", "UNKNOWN")
        if supported == "UNKNOWN":
            logger.warning(
                "Verify node: supported=UNKNOWN — treating as tool failure, "
                "not a grounding failure"
            )
            return {
                "grounded": False,
                "verification_failed": True,
                "reasoning_steps": ["Verification tool failed — accepted without retry"],
            }

        grounded = supported == "YES"
        return {
            "grounded": grounded,
            "verification_failed": False,
            "reasoning_steps": [f"Verification grounded = {grounded}"],
        }

    def _reflect(self, state):
        logger.info("Retry triggered")
        retries = state.get("retry_count", 0) + 1
        return {
            "retry_count": retries,
            # FIX: reset verification_failed so a fresh verify attempt on the
            # next pass starts clean and isn't incorrectly short-circuited.
            "verification_failed": False,
            "reasoning_steps": [f"Retry attempt {retries}"],
        }

    async def _delegate(self, state):
        """Call remote A2A agent when local docs are empty."""
        logger.info("Delegating to remote A2A agent")
        try:
            answer = await call_remote_agent(
                settings.REMOTE_AGENT_URL, state["rewritten_question"]
            )
            return {
                "draft_answer": answer,
                "delegated": True,
                "reasoning_steps": ["Answer delegated to remote A2A agent"],
            }
        except Exception as e:
            logger.error(f"A2A delegation failed: {e}")
            return {
                "draft_answer": "Could not retrieve answer from remote agent.",
                "delegated": True,
                "reasoning_steps": ["A2A delegation failed"],
            }

    async def _finalize(self, state):
        draft = state.get("draft_answer", "").strip()

        if not draft:
            label = state.get("relevance_label", "")
            if label == "NO_MATCH":
                draft = (
                    "I couldn't find relevant information in the uploaded documents "
                    "to answer your question. Please try rephrasing or upload "
                    "additional documents that cover this topic."
                )
            else:
                draft = "The agent was unable to produce an answer. Please try again."

            logger.warning(
                f"_finalize: empty draft_answer with relevance_label='{label}' "
                "— using fallback message"
            )

        return {
            "final_answer": draft,
            "reasoning_steps": ["Final answer accepted"],
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_rerank(self, state):
        """Delegate via A2A if no documents were retrieved locally."""
        if not state.get("reranked_docs") and settings.REMOTE_AGENT_URL:
            logger.info("No local docs -> routing to A2A delegate")
            return "delegate"
        return "check_relevance"

    def _route_after_relevance(self, state):
        if state["relevance_label"] == "NO_MATCH":
            logger.info("No relevant docs found — short-circuiting to finalize")
            return "finalize"
        return "research"

    def _route_after_research(self, state):
        """Route to MCP ToolNode if LLM emitted tool calls."""
        if state.get("tool_calls"):
            logger.info(
                f"Tool calls detected: {len(state['tool_calls'])} -> routing to tools"
            )
            return "tools"
        return "grade"

    def _route_after_verify(self, state):
        logger.info("Routing decision after verify")

        # FIX: check verification_failed FIRST, before any other branch.
        # A tool/LLM crash inside the verify node sets this flag.  Allowing
        # such a crash to fall through to reflect.decide() (which sees
        # grounded=False) would trigger a content-quality retry on what is
        # purely an infrastructure failure — potentially discarding a valid
        # first-round answer, as observed in the production logs.
        if state.get("verification_failed"):
            logger.warning(
                "Verification tool failed — accepting current answer without retry"
            )
            return "accept"

        # Delegated answers must never trigger a retry: there are no local
        # documents to retrieve against and re-running the loop would cycle.
        if state.get("delegated"):
            grounded = state.get("grounded", False)
            quality = state.get("answer_quality", "")
            logger.info(
                f"Delegated answer — skipping retry | grounded={grounded} quality={quality}"
            )
            return "accept"

        # FIX: moved NO_MATCH check AFTER verification_failed and delegated.
        # Previously this was the first branch, which meant a NO_MATCH set
        # on a previous pass could incorrectly short-circuit a retry that had
        # since produced a valid CAN_ANSWER result.
        if state.get("relevance_label") == "NO_MATCH":
            logger.info("No relevant docs -> stopping")
            return "stop"

        decision = self.reflect.decide(
            grounded=state["grounded"],
            quality=state["answer_quality"],
            retries=state.get("retry_count", 0),
            max_retries=self.MAX_RETRIES,
        )

        if decision == "accept":
            logger.info("Answer accepted")
            return "accept"

        if decision == "retry":
            return "retry"

        return "stop"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, question: str) -> dict:
        initial = {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "reranked_docs": [],
            "draft_answer": "",
            "final_answer": "",
            "confidence": "",
            "answer_quality": "",
            "grounded": False,
            # FIX: initialise verification_failed so the field is always
            # present in state from the first node onwards.
            "verification_failed": False,
            "retry_count": 0,
            "failure_reason": "",
            "draft_history": [],
            "reasoning_steps": [],
            "tool_calls": [],
            # FIX: key matches AgentState field name (was mcp_tool_results in
            # run() but mcp_tool_call_results in the TypedDict — now aligned).
            "mcp_tool_results": [],
            "delegated": False,
            "relevance_label": "",
        }

        final = await self.graph.ainvoke(initial)

        return {
            "final_answer": final["final_answer"],
            "draft_history": final["draft_history"],
            "reasoning_steps": final["reasoning_steps"],
            "retries": final["retry_count"],
            "delegated": final.get("delegated", False),
        }