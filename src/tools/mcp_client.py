# src/tools/mcp_client.py
from typing import List, Dict, Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import BaseTool

from src.tools.mcp_tools import MCP_SERVERS, _filter_servers
from src.config.settings import settings
from src.custom_logger.logger import logger


_client: MultiServerMCPClient | None = None
_tools: List[BaseTool] = []


def _log_exception(name: str, exc: Exception) -> None:
    """Unwrap ExceptionGroup if present, then log each sub-exception."""
    if isinstance(exc, ExceptionGroup):
        for sub in exc.exceptions:
            logger.warning(f"MCP server '{name}' failed probe: {sub}")
    else:
        logger.warning(f"MCP server '{name}' failed probe: {exc}")


async def startup() -> None:
    """Initialize MCP client and load all configured tools.
    Servers that fail to connect are skipped — others still load.
    Must be awaited before any call to get_tools()."""

    global _client, _tools

    if _client is not None:
        logger.debug("MCP client already started - skipping")
        return

    all_servers = _filter_servers(settings.MCP_ENABLED_SERVERS)

    if not all_servers:
        logger.info("MCP disabled (MCP_ENABLED_SERVERS is empty) — no tools loaded")
        return

    healthy_servers: Dict[str, Any] = {}

    for name, cfg in all_servers.items():
        try:
            probe = MultiServerMCPClient({name: cfg})
            await probe.get_tools()
            healthy_servers[name] = cfg
            logger.info(f"MCP server '{name}' — OK")
        except Exception as exc:
            _log_exception(name, exc)

    if not healthy_servers:
        logger.warning("All MCP servers failed — running without tools")
        return

    try:
        _client = MultiServerMCPClient(healthy_servers)
        _tools = await _client.get_tools()
        logger.info(
            f"MCP client ready | {len(_tools)} tools from "
            f"{list(healthy_servers.keys())}: {[t.name for t in _tools]}"
        )
    except Exception as e:
        logger.error(f"MCP client final init failed: {e}")
        _client = None
        _tools = []


async def shutdown() -> None:
    """Gracefully close the MCP client and release all subprocesses."""
    global _client, _tools

    if _client is None:
        return

    logger.info("Shutting down MCP client")

    try:
        if hasattr(_client, "close"):
            await _client.close()
    except Exception as e:
        logger.warning(f"MCP client shutdown error (non-fatal): {e}")
    finally:
        _client = None
        _tools = []


def get_tools() -> List[BaseTool]:
    return _tools


def is_ready() -> bool:
    return _client is not None and len(_tools) > 0