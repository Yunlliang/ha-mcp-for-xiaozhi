"""Outbound WebSocket transport: Home Assistant connects to the Xiaozhi MCP endpoint."""

import asyncio
from contextlib import suppress
import logging

import aiohttp
import anyio

from mcp import types
from mcp.shared.message import SessionMessage

try:  # mcp >= 2.x: JSONRPCMessage is a union alias, decode it with its TypeAdapter
    from mcp_types.jsonrpc import jsonrpc_message_adapter as _jsonrpc_adapter
except ImportError:  # pragma: no cover - mcp 1.x fallback
    _jsonrpc_adapter = None

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .const import DOMAIN
from .server import create_server
from .session import Session
from .types import WsMCPServerConfigEntry

_LOGGER = logging.getLogger(__name__)

RETRY_SECONDS = 20
HEARTBEAT_SECONDS = 50


def _parse_jsonrpc_message(payload: dict):
    """Decode a raw JSON-RPC payload into an MCP message.

    In mcp 2.x ``JSONRPCMessage`` is ``JSONRPCRequest | JSONRPCNotification |
    JSONRPCResponse | JSONRPCError`` (a typing alias), so it has no
    ``model_validate``; the SDK exposes a ``TypeAdapter`` for decoding instead.
    """
    if _jsonrpc_adapter is not None:
        return _jsonrpc_adapter.validate_python(payload)
    return types.JSONRPCMessage.model_validate(payload)


def _set_status(hass: HomeAssistant, entry: WsMCPServerConfigEntry, state: str) -> None:
    """Expose the connection state as a state attribute for debugging."""
    hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", state)


async def async_setup_entry(
    hass: HomeAssistant, entry: WsMCPServerConfigEntry
) -> bool:
    """Set up the outbound MCP client connection for a config entry."""
    task = hass.async_create_task(
        _connect_loop(hass, entry), f"{DOMAIN} connect loop {entry.entry_id}"
    )
    hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})["task"] = task
    return True


def async_get_config_entry(hass: HomeAssistant) -> WsMCPServerConfigEntry:
    """Get the first enabled MCP server config entry."""
    config_entries: list[WsMCPServerConfigEntry] = (
        hass.config_entries.async_loaded_entries(DOMAIN)
    )
    if not config_entries:
        raise RuntimeError("Model Context Protocol server is not configured")
    if len(config_entries) > 1:
        raise RuntimeError("Found multiple Model Context Protocol configurations")
    return config_entries[0]


async def _connect_loop(
    hass: HomeAssistant, entry: WsMCPServerConfigEntry
) -> None:
    """Reconnect forever until the config entry is unloaded.

    Never exit on a failed or finished session: the Xiaozhi endpoint is a
    remote service and any disconnect (idle timeout, server restart, network
    change) must result in a new attempt. The task is cancelled from
    ``async_unload_entry`` which is the only way out of this loop.
    """
    while True:
        try:
            await _connect_to_client(hass, entry)
        except asyncio.CancelledError:
            _LOGGER.debug("mcp connect loop cancelled")
            raise
        except Exception:
            _LOGGER.exception("mcp connection attempt failed")
        _LOGGER.info("mcp reconnect in %s seconds", RETRY_SECONDS)
        await asyncio.sleep(RETRY_SECONDS)


async def _connect_to_client(
    hass: HomeAssistant, entry: WsMCPServerConfigEntry
) -> None:
    """Connect to the Xiaozhi MCP endpoint and serve MCP over the WebSocket."""
    session_manager = entry.runtime_data
    endpoint = entry.data.get("client_endpoint")
    if not endpoint:
        raise RuntimeError("No client endpoint configured in config entry")

    _LOGGER.info("mcp connecting to MCP client at: %s", endpoint)
    _set_status(hass, entry, "connecting")

    context = llm.LLMContext(
        platform=DOMAIN,
        context={},
        language="*",
        assistant=conversation.DOMAIN,
        device_id=None,
    )
    llm_api_id = entry.data[CONF_LLM_HASS_API]
    _LOGGER.debug("mcp llm_api_id: %s", llm_api_id)

    server = await create_server(hass, llm_api_id, context, read_only=False)
    options = await hass.async_add_executor_job(server.create_initialization_options)

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    # No total timeout: the WebSocket is long lived. Only bound the handshake.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    try:
        async with session_manager.create(Session(read_stream_writer)) as session_id:
            async with aiohttp.ClientSession(timeout=timeout) as client_session:
                async with client_session.ws_connect(endpoint) as ws:
                    _LOGGER.info("mcp websocket connected (session %s)", session_id)
                    _set_status(hass, entry, "connected")

                    async def ws_reader() -> None:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    message = _parse_jsonrpc_message(msg.json())
                                    _LOGGER.debug("mcp <- client: %s", message)
                                    await read_stream_writer.send(
                                        SessionMessage(message)
                                    )
                                except Exception as err:  # noqa: BLE001
                                    _LOGGER.error(
                                        "mcp invalid message from client: %s", err
                                    )
                            elif msg.type == aiohttp.WSMsgType.CLOSE:
                                _LOGGER.warning(
                                    "mcp websocket closed by peer: %s", msg.extra
                                )
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                _LOGGER.error("mcp websocket error: %s", msg.data)
                        _LOGGER.info("mcp websocket read loop finished")
                        tg.cancel_scope.cancel()

                    async def ws_writer() -> None:
                        async for session_message in write_stream_reader:
                            _LOGGER.debug("mcp -> client: %s", session_message)
                            await ws.send_str(
                                session_message.message.model_dump_json(
                                    by_alias=True, exclude_none=True
                                )
                            )
                        _LOGGER.info("mcp websocket write loop finished")
                        await ws.close()

                    async def heartbeat() -> None:
                        while True:
                            await asyncio.sleep(HEARTBEAT_SECONDS)
                            try:
                                await ws.ping()
                            except Exception as err:  # noqa: BLE001
                                _LOGGER.info("mcp heartbeat ping failed: %s", err)
                                break

                    async with anyio.create_task_group() as tg:
                        tg.start_soon(ws_reader)
                        tg.start_soon(ws_writer)
                        tg.start_soon(heartbeat)
                        await server.run(read_stream, write_stream, options)
    finally:
        _set_status(hass, entry, "disconnected")
        for stream in (
            read_stream_writer,
            read_stream,
            write_stream,
            write_stream_reader,
        ):
            with suppress(Exception):
                await stream.aclose()
