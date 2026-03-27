from typing import List 

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool 

from src.tools.mcp_tools import MCP_SERVERS,_filter_servers
from src.config.settings import settings 
from src.custom_logger.logger import logger


_client: MultiServerMCPClient | None  = None 
_tools = []

