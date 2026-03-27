from langgraph.graph import StateGraph,START,END

from src.agents.state import AgentState
from src.agents.query_rewriter import QueryRewriter
from src.agents.research_agent import ResearchAgent
from src.agents.verfication_agent import VerificationAgent
from src.agents.answer_grader import AnswerGrader
from src.agents.reflexion_agent import ReflexionAgent

from src.custom_logger.logger import logger 

class AgentWorkflow:
    MAX_RETRIES = 2 

    def __init__(self,retriever,builder):
        self.retriever = retriever
        self.builder = builder 

        self.research = ResearchAgent()
        self.verify = VerificationAgent()
        self.rewrite = QueryRewriter()
        self.grade = AnswerGrader()
        self.reflect = ReflexionAgent()

        self.graph = self._build()
    
    def _build(self):
        g = StateGraph(AgentState)

        g.add_node("rewrite",self._rewrite)
        g.add_node("retrieve",self._retrieve)
        g.add_node("rerank",self._rerank)
        g.add_node("research",self._research)
        g.add_node("grade",self._grade)
        g.add_node("verify",self._verify)
        g.add_node("reflect",self._reflect)
        g.add_node("finalize",self._finalize)

        g.add_edge(START,"rewrite")
        g.add_edge("rewrite","retrieve")
        g.add_edge("retrieve","rerank")
        g.add_edge("rerank","research")
        g.add_edge("research","grade")
        g.add_edge("grade","verify")

        g.add_conditional_edges("verify",
                                self._route_after_verify,
                                {"accept":"finalize",
                                 "retry":"reflect",
                                 "stop":"finalize"})
        
        g.add_edge("reflect","rewrite")
        g.add_edge("finalize",END)

        return g.compile()
    
    async def _rewrite(self,state):
        logger.info("Rewriting query")

        query = await self.rewrite.rewrite(state['question'])

        return {"rewritten_question":query,
                "reasoning_steps":["Query rewritten"]}
    
    async def  _retrieve(self,state):
        logger.info("Retrieving documents")

        docs = await self.retriever.ainvoke(state["rewritten_question"])
        
        return {"documents":docs,
                "reasoning_steps":[f"Retrieved {len(docs)} documents"]}
    
    async def _rerank(self,state):
        logger.info("Reranking documents")

        reranked = await self.builder.rerank(state["rewritten_question"],
                                             state["documents"])
        
        return {"reranked_docs":reranked,
                "reasoning_steps":["Documents reranked"]}
    
    async def _research(self,state):
        logger.info("Generating answer")

        result = await self.research.generate(state["rewritten_question"],
                                              state["reranked_docs"])
        
        return {"draft_answer":result['draft_answer'],
                "confidence":result["confidence"],
                "draft_history":[result["draft_answer"]],
                "reasoning_steps":["Research agent generated an answer"]}
    
    async def _grade(self,state):
        logger.info("Grading the answer")

        result = await self.grade.grade(state["question"],state["draft_answer"])

        return {"answer_quality":result["quality"],
                "reasoning_steps":[f"Answer quality = {result['quality']}"]}
    
    async def _verify(self,state):
        logger.info("Verifying ground")

        result = await self.verify.check(state["draft_answer"],
                                   state["reranked_docs"])
        
        grounded = result["supported"] == "YES"

        return {"grounded":grounded,
                "reasoning_steps":[f"Verification grounded = {grounded}"]}
    
    def _reflect(self,state):
        logger.info("Retry triggered")
        retries = state.get("retry_count",0) + 1
        return {"retry_count":retries,
                "reasoning_steps":[f"Retry attempt {retries}"]}
    
    def _route_after_verify(self,state):

        logger.info("Routing decision")

        decision = self.reflect.decide(grounded=state["grounded"],
                                       quality=state["answer_quality"],
                                       retries=state.get("retry_count",0),
                                       max_retries=self.MAX_RETRIES)

        if decision == "accept":
            logger.info("Answer accepted")
            state["final_answer"] = state["draft_answer"]
            return "accept"

        if decision == "stop":
            logger.info("Retry limit reached")
            state["final_answer"] = state["draft_answer"]
            return "stop"

        logger.info("Retrying research")
        return "retry"
    
    async def _finalize(self,state):
        return {"final_answer":state["draft_answer"],
                "reasoning_steps":["Final answer accepted"]}
    
    async def run(self,question):

        initial = {"question":question,
                   "rewritten_question":"",
                   "documents":[],
                   "reranked_docs":[],
                   "draft_answer":"",
                   "final_answer":"",
                   "confidence":"",
                   "answer_quality":"",
                   "grounded":False,
                   "retry_count":0,
                   "failure_reason":"",
                   "draft_history":[],
                   "reasoning_steps":[]}
        
        final = await self.graph.ainvoke(initial)

        return {"final_answer":final["final_answer"],
                "draft_history":final["draft_history"],
                "reasoning_steps":final["reasoning_steps"],
                "retries":final["retry_count"]}
        