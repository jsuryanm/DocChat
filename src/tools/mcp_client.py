from typing import List 

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import BaseTool 

from src.tools.mcp_tools import MCP_SERVERS,_filter_servers
from src.config.settings import settings 
from src.custom_logger.logger import logger


_client: MultiServerMCPClient | None  = None 
_tools: List[BaseTool] = []

async def startup() -> None:
    """Initialized MCP client and load all configured tools.
    Must be awaited before any call to get_tools()"""

    global _client,_tools

    if _client is not None:
        logger.debug("MCP client already started - skipping")
        return 

    servers = _filter_servers(settings.MCP_ENABLED_SERVERS)

    if not servers:
        logger.info("MCP disabled (MCP_ENABLED_SERVERS is empty) no tools loaded")
        return
    
    try:
        _client = MultiServerMCPClient(servers)
        await _client.__aenter__()
        _tools = _client.get_tools()
        logger.info(f"MCP client ready | {len(_tools)} tools: {[tool.name for tool in _tools]}")
    
    except Exception as e:
        logger.error(f"MCP client startup failed: {e}")
        _client = None 
        _tools = []
    
async def shutdown():
    """Gracefully close the MCP client and release all subprocess"""
    global _client,_tools 

    if _client is None:
        return 
    
    logger.info("Shutting down MCP client")

    try:
        await _client.__aexit__(None,None,None)
    except Exception as e:
        logger.warning(f"MCP client shutdown error (non-fatal):{e}")
    finally:
        _client = None 
        _tools = []

def get_tools() -> List[BaseTool]:
    return _tools

def is_ready() -> bool:
    return _client is not None and len(_tools) > 0