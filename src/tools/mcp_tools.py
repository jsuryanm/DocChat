from typing import List,Any,Dict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import BaseTool

from src.config.settings import settings 
from src.custom_logger.logger import logger

MCP_SERVERS: Dict[str,Any] = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/docs"],
        "transport": "stdio",
    },
    "tavily":{
            "transport":"streamable_http",
            "url":f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.TAVILY_API_KEY}",
        }
}

def _filter_servers(enabled: str) -> Dict[str,Any]:
    """Return only the servers listed comma-separated enabled string. 
    If `enabled` is empty or "*", return all servers.

    Parameters
    ----------
    enabled : str
        Value of settings.MCP_ENABLED_SERVERS, e.g. "filesystem,brave_search"""

    if not enabled or enabled.strip() == "*":
        return MCP_SERVERS

    names = {s.strip() for s in enabled.split(",") if s.strip()}

    filtered = {key:val for key,val in MCP_SERVERS.items() if key in names}

    unknown = names - set(MCP_SERVERS.keys())
    if unknown:
        logger.warning(f"MCP_ENABLED_SERVERS references unknown servers: {unknown}")

    return filtered

async def load_mcp_tools(enabled_servers: str = "") -> List[BaseTool]:
    """Open a shorted lived MultiServerClient collect all tools and close
    Parameters
    ----------
    enabled_servers : str
        Comma-separated server names to load. Empty = all servers.

    Returns
    -------
    List[BaseTool]
        LangChain-compatible tool objects ready to be passed to bind_tools()
        or ToolNode. Returns [] on any failure so callers degrade gracefully.
    """

    servers = _filter_servers(enabled_servers)

    if not servers:
        logger.info("No MCP servers configured - skipping tool load")
        return []
    
    logger.info(f"Loading MCP tools from server: {list(servers.keys())}")

    try:
        async with MultiServerMCPClient(servers) as client:
            tools: List[BaseTool] = client.get_tools()
            logger.info(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")
            return tools

    except Exception as e:
        logger.error(f"MCP tool load failed: {e}")
        return []