from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.tools.mcp_client import startup as mcp_startup, shutdown as mcp_shutdown
from src.a2a.client import startup as a2a_startup, shutdown as a2a_shutdown
from api.dependencies import init_singletons
from api.router import router
from src.custom_logger.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DocChat API")
    await mcp_startup()          
    await a2a_startup()          
    await init_singletons()      
    logger.info("DocChat API ready")
    yield
    logger.info("Shutting down DocChat API")
    await mcp_shutdown()
    await a2a_shutdown()

app = FastAPI(
    title="DocChat API",
    version="0.1.0",
    description="Agentic RAG with document grounding and Tavily web search fallback",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")