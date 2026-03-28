from typing import Optional,Dict
import httpx 
from a2a.client import A2AClient,A2ACardResolver
# handles agent discovery,protocol formatting, message sending and response parsing
from a2a.types import MessageSendParams,SendMessageRequest
from a2a.utils import new_agent_text_message

from src.custom_logger.logger import logger 

DEFAULT_TIMEOUT = 60.0

_CLIENT_CACHE: Dict[str,A2AClient] = {}


async def _get_a2a_client(agent_url: str,
                          http: httpx.AsyncClient) -> A2AClient:
    """Build or reuse A2A client for remote agent
    Uses AgentCard discovery via A2ACardResolver"""

    if agent_url in _CLIENT_CACHE:
        return _CLIENT_CACHE[agent_url]
    
    logger.info(f"A2A discovery: fetching agent card from {agent_url}")

    resolver = A2ACardResolver(httpx_client=http,
                               base_url=agent_url)
    
    agent_card = await resolver.get_agent_card()

    client = A2AClient(httpx_client=http,
                       agent_card=agent_card)
    
    _CLIENT_CACHE[agent_url] =  client
    return client


async def call_remote_agent(agent_url: str,
                            question: str,
                            timeout: float=DEFAULT_TIMEOUT) -> str:
    """
    Send a question to a remote A2A agent and return its text answer.

    The function:
    1. Fetches the remote agent's AgentCard from {agent_url}/.well-known/agent.json
    2. Sends the question as a blocking A2A message
    3. Extracts and returns the first text part from the response

    Parameters
    ----------
    agent_url : str
        Base URL of the remote A2A agent, e.g. "http://specialist:9001"
    question : str
        The question to send.
    timeout : float
        HTTP client timeout in seconds.

    Returns
    -------
    str
        The text answer from the remote agent.
        Falls back to a descriptive error string on any failure —
        callers should handle this gracefully rather than raising.
    """

    if not agent_url:
        raise ValueError("agent_url must not be empty")
    
    logger.info(f"A2A client contacting: {agent_url} | question: {question[:60]}..")

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            client = await _get_a2a_client(agent_url,http)

            request = SendMessageRequest(message=new_agent_text_message(question),
                                         configuration=MessageSendParams(blocking=True))
            
            response = await client.send_message(request)
            answer = _extract_text(response)

            logger.info(f"A2A client: response received | chars = {len(answer)}")
    
    except httpx.ConnectError as e:
        logger.info(f"A2A client connection refused: {agent_url}")
        return ("Remote specialist agent unavailable connection refused")
    
    except httpx.TimeoutException:
        logger.error(f"A2A client timeout: {agent_url}")
        return "Remote specialist agent timed out"
    
    except Exception as e:
        logger.error(f"A2A client unexpected error: {e}")
        return ("Remote agent call failed")

async def ping_remote_agent(agent_url: str,
                            timeout: float = 5.0) -> bool:
    """Health check remote agent. Returns True if reachable never raises"""
    if not agent_url:
        return False 
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(f"{agent_url}/.well-known/agent-card.json")
            reachable = resp.status_code == 200 

            logger.info(f"A2A ping {agent_url} -> {"OK" if reachable else "FAIL"}")
            return reachable
    
    except Exception as e:
        logger.warning(f"A2A ping failed: {e}")
        return False
    
def _extract_text(response) -> str:
    """
    Extract first text response from A2A message.
    """
    try:
        parts = response.result.message.parts

        for part in parts:
            root = getattr(part, "root", part)
            text = getattr(root, "text", None)

            if text and text.strip():
                return text.strip()

    except Exception as e:
        logger.warning(f"A2A response parse error: {e}")
    
    return "Remote agent returned no text response."