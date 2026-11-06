"""The Model Context Protocol Server integration."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from . import websocket_transport
from .const import DOMAIN
from .session import SessionManager
from .types import WsMCPServerConfigEntry

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "CONFIG_SCHEMA",
    "DOMAIN",
    "async_setup",
    "async_setup_entry",
    "async_unload_entry",
]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Model Context Protocol component."""
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: WsMCPServerConfigEntry
) -> bool:
    """Set up Model Context Protocol Server from a config entry."""

    async def _system_started(event) -> None:
        try:
            # 为每个配置实例创建独立的会话管理器
            session_manager = SessionManager()
            # 将会话管理器存储在 hass.data 中,使用 entry.entry_id 作为唯一键
            hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
                "session_manager": session_manager,
                "config": entry.data,
            }
            hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", "starting")

            entry.runtime_data = session_manager
            await websocket_transport.async_setup_entry(hass, entry)
        except Exception as ex:  # noqa: BLE001
            hass.states.async_set(f"{DOMAIN}.{entry.entry_id}_status", "error")
            _LOGGER.error("MCP setup failed: %s", ex)
            # 清理已添加的 session_manager
            if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
                hass.data[DOMAIN].pop(entry.entry_id)

    if hass.is_running:
        await _system_started(None)
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _system_started)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: WsMCPServerConfigEntry
) -> bool:
    """Unload a config entry and stop its reconnect loop."""
    data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    if data is not None:
        # Cancel the reconnect loop first, otherwise a reload leaves a second
        # loop behind that keeps fighting the new one over the same endpoint.
        task: asyncio.Task | None = data.get("task")
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        session_manager: SessionManager = data["session_manager"]
        session_manager.close()

    hass.states.async_remove(f"{DOMAIN}.{entry.entry_id}_status")
    return True
