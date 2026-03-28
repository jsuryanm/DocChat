# src/tools/mcp_tools.py
from typing import List, Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import BaseTool

from src.config.settings import settings
from src.custom_logger.logger import logger


MCP_SERVERS: Dict[str, Any] = {
    **(
        {
            "tavily": {
                "transport": "streamable_http",
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.TAVILY_API_KEY}",
            }
        }
        if settings.TAVILY_API_KEY
        else {}
    ),
}


def _filter_servers(enabled: str) -> Dict[str, Any]:
    """Return only the servers listed in a comma-separated enabled string.
    If `enabled` is empty or '*', return all servers.

    Parameters
    ----------
    enabled : str
        Value of settings.MCP_ENABLED_SERVERS, e.g. 'tavily'
    """
    if not enabled or enabled.strip() == "*":
        return MCP_SERVERS

    names = {s.strip() for s in enabled.split(",") if s.strip()}
    filtered = {key: val for key, val in MCP_SERVERS.items() if key in names}

    unknown = names - set(MCP_SERVERS.keys())
    if unknown:
        logger.warning(f"MCP_ENABLED_SERVERS references unknown servers: {unknown}")

    return filtered