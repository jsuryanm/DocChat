# src/agents/web_search_agent.py
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, ToolMessage

from src.tools.mcp_client import get_tools
from src.config.settings import settings
from src.custom_logger.logger import logger

_SYSTEM = """You are a web research agent.
You have been given web search results for the user's question.
Write a clear, accurate, concise answer based only on the search results provided.
Cite sources where possible.
If the results are not useful, say so clearly."""


class WebSearchAgent:
    """Fires ONLY when local retrieval returns NO_MATCH.
    Calls the Tavily search tool directly, then summarizes with LLM."""

    def __init__(self):
        tools = get_tools()
        self._search_tool = next(
            (t for t in tools if t.name == "tavily_search"), None
        )

        if not self._search_tool:
            logger.warning("WebSearchAgent: tavily_search tool not available")
            self._available = False
            return

        self._available = True

        self._llm = ChatOpenAI(
            model=settings.RESEARCH_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )

        self._summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", "Question: {question}\n\nSearch results:\n{search_results}"),
        ])

        self._chain = self._summarize_prompt | self._llm | StrOutputParser()

        logger.info("WebSearchAgent ready with tool: tavily_search")

    @property
    def available(self) -> bool:
        return self._available

    async def search(self, question: str) -> Dict:
        if not self._available:
            return {
                "draft_answer": "Web search is not available.",
                "web_used": False,
                "confidence": "LOW",
            }

        try:
            # Step 1: call Tavily tool directly — no LLM involved here
            raw_results = await self._search_tool.ainvoke({"query": question})

            if not raw_results:
                logger.warning("WebSearchAgent: Tavily returned empty results")
                return {
                    "draft_answer": "Web search returned no useful results.",
                    "web_used": False,
                    "confidence": "LOW",
                }

            logger.info(f"WebSearchAgent: raw results chars={len(str(raw_results))}")

            # Step 2: summarize the raw results with LLM
            answer = await self._chain.ainvoke({
                "question": question,
                "search_results": str(raw_results),
            })

            answer = answer.strip()

            if not answer:
                logger.warning("WebSearchAgent: summarization returned empty string")
                return {
                    "draft_answer": "Web search returned no useful results.",
                    "web_used": False,
                    "confidence": "LOW",
                }

            logger.info(f"WebSearchAgent completed | chars={len(answer)}")
            return {
                "draft_answer": answer,
                "web_used": True,
                "confidence": "MEDIUM",
            }

        except Exception as e:
            logger.error(f"WebSearchAgent failed: {e}")
            return {
                "draft_answer": "Web search failed.",
                "web_used": False,
                "confidence": "LOW",
            }