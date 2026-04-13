from typing import Optional, Dict
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest
from a2a.utils import new_agent_text_message

from src.custom_logger.logger import logger

DEFAULT_TIMEOUT = 60.0
 

_http_client: httpx.AsyncClient | None = None
_client_cache: Dict[str, A2AClient] = {}


async def startup() -> None:
    global _http_client
    if _http_client is not None:
        return
    _http_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    logger.info("A2A HTTP client started")


async def shutdown() -> None:
    global _http_client, _client_cache
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        _client_cache.clear()
        logger.info("A2A HTTP client closed")


def _require_http() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError(
            "A2A HTTP client is not initialised. "
            "Call `await a2a.client.startup()` at application startup."
        )
    return _http_client


async def _get_a2a_client(agent_url: str) -> A2AClient:
    if agent_url in _client_cache:
        return _client_cache[agent_url]

    http = _require_http()
    logger.info(f"A2A discovery: fetching agent card from {agent_url}")

    resolver = A2ACardResolver(httpx_client=http, base_url=agent_url)
    agent_card = await resolver.get_agent_card()

    client = A2AClient(httpx_client=http, agent_card=agent_card)
    _client_cache[agent_url] = client
    return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)
async def _send_with_retry(client: A2AClient, request: SendMessageRequest):
    return await client.send_message(request)


async def call_remote_agent(
    agent_url: str,
    question: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    if not agent_url:
        raise ValueError("agent_url must not be empty")

    logger.info(f"A2A client contacting: {agent_url} | question: {question[:60]}..")

    try:
        client = await _get_a2a_client(agent_url)

        request = SendMessageRequest(
            message=new_agent_text_message(question),
            configuration=MessageSendParams(blocking=True),
        )

        response = await _send_with_retry(client, request)
        answer = _extract_text(response)

        logger.info(f"A2A client: response received | chars = {len(answer)}")
        return answer

    except httpx.ConnectError:
        logger.error(f"A2A client connection refused: {agent_url}")
        return "Remote specialist agent unavailable — connection refused"

    except httpx.TimeoutException:
        logger.error(f"A2A client timeout after retries: {agent_url}")
        return "Remote specialist agent timed out"

    except Exception as e:
        logger.error(f"A2A client unexpected error: {e}")
        return "Remote agent call failed"


async def ping_remote_agent(agent_url: str, timeout: float = 5.0) -> bool:
    """Health-check a remote agent. Returns True if reachable, never raises.
    """
    if not agent_url:
        return False

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            # Correct endpoint: matches what A2ACardResolver fetches
            resp = await http.get(f"{agent_url}/.well-known/agent.json")
            reachable = resp.status_code == 200
            logger.info(f"A2A ping {agent_url} -> {'OK' if reachable else 'FAIL'}")
            return reachable

    except Exception as e:
        logger.warning(f"A2A ping failed: {e}")
        return False


def _extract_text(response) -> str:
    """Extract the first text part from an A2A response."""
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