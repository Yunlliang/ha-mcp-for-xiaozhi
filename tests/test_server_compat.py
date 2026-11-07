"""Exercise the integration against the real MCP SDK without a running HA instance."""

import ast
import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anyio
from mcp import ClientSession, types

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.ws_mcp_server.server import (
    META_DEVICE_ID,
    SNAPSHOT_RESOURCE_URI,
    create_server,
)
from custom_components.ws_mcp_server.websocket_transport import _parse_jsonrpc_message


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "ws_mcp_server"


class ServerCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api = SimpleNamespace(
            tools=[SimpleNamespace(name="homeassistant__GetLiveContext")],
            async_call_tool=AsyncMock(),
        )
        self.get_api = self.enterContext(patch(
            "custom_components.ws_mcp_server.server.llm.async_get_api",
            new_callable=AsyncMock,
            return_value=self.api,
        ))
        self.context = llm.LLMContext(
            platform="ws_mcp_server", context={}, language="en",
            assistant="conversation", device_id=None,
        )
        self.server = await create_server(None, "assist", self.context, read_only=False)
        self.request_context = SimpleNamespace(meta=None)

    async def call_handler(self, method, params):
        entry = self.server.get_request_handler(method)
        self.assertIsNotNone(entry)
        return await entry.handler(self.request_context, params)

    def test_manifest_is_readable_as_utf8_json(self):
        manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["domain"], "ws_mcp_server")

    def test_transport_supplies_required_factory_arguments(self):
        tree = ast.parse((COMPONENT / "websocket_transport.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "create_server"]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        bound = inspect.signature(create_server).bind(
            *[None for arg in call.args],
            **{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords},
        )
        self.assertIs(bound.arguments["read_only"], False)

    async def test_tool_result_preserves_data_and_error_flag(self):
        for error in (False, True):
            with self.subTest(error=error):
                data = {"message": "测试", "nested": [1, {"success": not error}]}
                self.api.async_call_tool.return_value = llm.ToolResult(data=data, error=error)
                result = await self.call_handler(
                    "tools/call", types.CallToolRequestParams(name="test", arguments={})
                )
                self.assertEqual(json.loads(result.content[0].text), data)
                self.assertIs(result.is_error, error)
                self.assertIs(result.model_dump(by_alias=True)["isError"], error)

    async def test_context_resource_reads_tool_result_data(self):
        self.api.async_call_tool.return_value = llm.ToolResult(
            data={"result": "Sony XB10 is idle"}
        )
        result = await self.call_handler(
            "resources/read", types.ReadResourceRequestParams(uri=SNAPSHOT_RESOURCE_URI)
        )
        self.assertEqual(result.contents[0].text, "Sony XB10 is idle")

    async def test_context_resource_propagates_tool_error(self):
        self.api.async_call_tool.return_value = llm.ToolResult(
            data={"error": "Context unavailable"}, error=True
        )
        with self.assertRaisesRegex(HomeAssistantError, "Unable to read"):
            await self.call_handler(
                "resources/read", types.ReadResourceRequestParams(uri=SNAPSHOT_RESOURCE_URI)
            )

    async def test_context_resource_is_hidden_without_live_context_tool(self):
        self.api.tools = []
        result = await self.call_handler("resources/list", None)
        self.assertEqual(result.resources, [])
        with self.assertRaisesRegex(ValueError, "Unknown resource"):
            await self.call_handler(
                "resources/read", types.ReadResourceRequestParams(uri=SNAPSHOT_RESOURCE_URI)
            )
        self.api.async_call_tool.assert_not_awaited()

    async def test_device_id_is_forwarded_to_home_assistant(self):
        self.request_context.meta = {META_DEVICE_ID: "device-1"}
        await self.call_handler("resources/list", None)
        self.assertEqual(self.get_api.call_args.args[2].device_id, "device-1")

    async def test_invalid_device_id_is_rejected(self):
        self.request_context.meta = {META_DEVICE_ID: 42}
        with self.assertRaisesRegex(ValueError, "must be a string"):
            await self.call_handler("resources/list", None)
        self.get_api.assert_not_awaited()

    def test_transport_decodes_sdk_jsonrpc_messages(self):
        for payload in (
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Invalid"}},
        ):
            with self.subTest(payload=payload):
                message = _parse_jsonrpc_message(payload)
                self.assertEqual(
                    json.loads(message.model_dump_json(by_alias=True, exclude_none=True)),
                    payload,
                )

    async def test_real_sdk_handshake_and_requests(self):
        """Negotiate and exchange requests using real client/server sessions."""
        client_write, server_read = anyio.create_memory_object_stream(0)
        server_write, client_read = anyio.create_memory_object_stream(0)
        self.api.tools = []
        self.api.async_call_tool.return_value = llm.ToolResult(data={"ok": True})
        task = asyncio.create_task(self.server.run(
            server_read, server_write, self.server.create_initialization_options()
        ))
        try:
            async with client_write, server_read, server_write, client_read:
                async with ClientSession(client_read, client_write) as client:
                    async with asyncio.timeout(10):
                        initialized = await client.initialize()
                        self.assertEqual(initialized.server_info.name, "home-assistant")
                        await client.send_ping()
                        self.assertEqual((await client.list_tools()).tools, [])
                        result = await client.call_tool("test", {})
                        self.assertEqual(json.loads(result.content[0].text), {"ok": True})
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    unittest.main()
