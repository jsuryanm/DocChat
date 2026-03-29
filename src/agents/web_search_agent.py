from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.mcp_client import get_tools
from src.config.settings import settings
from src.custom_logger.logger import logger

_SYSTEM = """You are a web research agent.
Use the web_search tool to find an accurate, concise answer.
Cite sources where possible.
If no results are useful, say so clearly."""

class WebSearchAgent:
    """Fires ONLY when local retrieval returns NO_MATCH.
    Uses the Tavily MCP tool already loaded at startup."""

    def __init__(self):
        tools = get_tools()
        tavily_tools = [t for t in tools if "search" in t.name.lower()]

        if not tavily_tools:
            logger.warning("WebSearchAgent: no search tool available")
            self._available = False
            return

        self._available = True
        llm = ChatOpenAI(
            model=settings.RESEARCH_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        ).bind_tools(tavily_tools)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", "{question}"),
        ])

        self.chain = prompt | llm | StrOutputParser()
        logger.info(f"WebSearchAgent ready with tools: {[t.name for t in tavily_tools]}")

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
            answer = await self.chain.ainvoke({"question": question})
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