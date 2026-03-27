import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.mcp_tools import MCP_SERVERS,_filter_servers,load_mcp_tools


class TestFilterServers:

    def test_empty_string_returns_all_servers(self):
        result = _filter_servers("")
        assert result == MCP_SERVERS

    def test_wildcard_returns_all_servers(self):
        result = _filter_servers("*")
        assert result == MCP_SERVERS

    def test_single_known_server_returned(self):
        result = _filter_servers("filesystem")
        assert "filesystem" in result
        assert "tavily" not in result

    def test_multiple_known_servers_returned(self):
        result = _filter_servers("filesystem,tavily")
        assert "filesystem" in result
        assert "tavily" in result

    def test_unknown_server_is_ignored(self):
        result = _filter_servers("nonexistent_server")
        assert result == {}

    def test_mixed_known_unknown_returns_only_known(self):
        result = _filter_servers("filesystem,nonexistent")
        assert "filesystem" in result
        assert "nonexistent" not in result

    def test_whitespace_in_server_names_is_stripped(self):
        result = _filter_servers("  filesystem , tavily  ")
        assert "filesystem" in result
        assert "tavily" in result

class TestLoadMcpTools:

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_servers_configured(self):
        # Patch _filter_servers to return empty dict
        with patch("src.tools.mcp_tools._filter_servers", return_value={}):
            result = await load_mcp_tools(enabled_servers="nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_tools_from_client(self):
        fake_tool = MagicMock()
        fake_tool.name = "read_file"

        fake_client = MagicMock()
        fake_client.get_tools.return_value = [fake_tool]
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.tools.mcp_tools.MultiServerMCPClient",
            return_value=fake_client,
        ):
            result = await load_mcp_tools(enabled_servers="filesystem")

        assert len(result) == 1
        assert result[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_client_exception(self):
        fake_client = MagicMock()
        fake_client.__aenter__ = AsyncMock(
            side_effect=RuntimeError("MCP server unreachable")
        )
        fake_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.tools.mcp_tools.MultiServerMCPClient",
            return_value=fake_client,
        ):
            result = await load_mcp_tools(enabled_servers="filesystem")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_multiple_tools(self):
        tool_names = ["read_file", "write_file", "list_directory"]
        fake_tools = [MagicMock(name=n) for n in tool_names]
        for t, n in zip(fake_tools, tool_names):
            t.name = n

        fake_client = MagicMock()
        fake_client.get_tools.return_value = fake_tools
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.tools.mcp_tools.MultiServerMCPClient",
            return_value=fake_client,
        ):
            result = await load_mcp_tools()

        assert len(result) == 3
        assert {t.name for t in result} == set(tool_names)
