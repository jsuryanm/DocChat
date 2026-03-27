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
        g.add_node("research",self._researh)
        g.add_node("grade",self._grade)
        g.add_node("verify",self._verify)
        g.add_node("reflect",self._reflect)

        g.add_edge(START,"rewrite")
        g.add_edge("rewrite","retrieve")
        g.add_edge("retrieve","rerank")
        g.add_edge("rerank","research")
        g.add_edge("research","grade")
        g.add_edge("grade","verify")

        g.add_conditional_edges("verify",
                                self._route_after_verify,
                                {"accept":END,
                                 "retry":"reflect",
                                 "stop":END})
        
        g.add_edge("reflect","research")

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
        
        return {"draft_answer":result['draft_answer']}