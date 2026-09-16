"""Offline discovery integration: metadata does not opt a group into chat."""

import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from test_runtime import Context, Event, ROOT, STUBS, runtime


class FakeConfig(dict):
    def __init__(self, values=None):
        super().__init__(values or {})
        self.schema = json.loads(
            (ROOT / "_conf_schema.json").read_text(encoding="utf-8")
        )
        self.save_config = Mock()


class GroupDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.plugins = []
        self.now = 1000.0
        self.monotonic = 50.0
        self.netguard = patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network forbidden")
        )
        self.netguard.start()
        self.modules = patch.dict(sys.modules, STUBS)
        self.modules.start()

    async def asyncTearDown(self):
        for plugin in self.plugins:
            await plugin.terminate()
        self.modules.stop()
        self.netguard.stop()
        self.temp.cleanup()

    def plugin(self, values=None, *, path=None, open_store=True):
        config = FakeConfig(values)
        plugin = runtime.AstrbotGroupChatLite(Context(), config)
        plugin._clock = lambda: self.now
        plugin._monotonic = lambda: self.monotonic
        plugin._decide = AsyncMock(return_value=True)
        plugin._provider = AsyncMock(
            side_effect=AssertionError("No model call allowed")
        )
        if open_store:
            plugin.store = runtime.Store(
                path or Path(self.temp.name) / f"chat-{len(self.plugins)}.sqlite3"
            )
        self.plugins.append(plugin)
        return plugin, config

    @staticmethod
    def event(ident=1, name="测试群", **kwargs):
        event = Event(ident, **kwargs)
        event.message_obj.group = types.SimpleNamespace(group_name=name)
        return event

    async def consume(self, plugin, event):
        requests = []
        async for req in plugin.on_group_message(event):
            requests.append(req)
            await plugin.on_llm_request(event, req)
            await plugin.on_llm_response(
                event, types.SimpleNamespace(role="assistant", completion_text="answer")
            )
        return requests

    def assert_metadata_only(self, plugin):
        for table in ("windows", "messages", "summaries"):
            self.assertEqual(
                plugin.store.connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0],
                0,
                table,
            )
        plugin._decide.assert_not_awaited()
        plugin._provider.assert_not_awaited()

    async def test_empty_allowlist_discovers_without_reading_body_or_claiming_event(
        self,
    ):
        plugin, config = self.plugin()
        event = self.event(text="PRIVATE BODY MUST NOT BE STORED")
        event.get_message_str = Mock(side_effect=AssertionError("No body read"))
        self.assertEqual(await self.consume(plugin, event), [])
        self.assertFalse(event.call_llm)
        candidate = plugin.store.get_discovered_group(event.unified_msg_origin)
        self.assertEqual(candidate["display_name"], "测试群")
        self.assertEqual(candidate["group_id"], "-10")
        self.assertNotIn("PRIVATE BODY", repr(candidate))
        self.assert_metadata_only(plugin)
        self.assertEqual(
            config.schema["selected_groups"]["options"], [event.unified_msg_origin]
        )
        self.assertEqual(plugin.settings.group_ids, ())
        self.assertEqual(dict(config), {})
        config.save_config.assert_not_called()

    async def test_excluded_event_types_and_switches_are_not_discovered(self):
        for case in (
            "private",
            "non_tg",
            "bot",
            "stopped",
            "disabled",
            "discovery_off",
        ):
            with self.subTest(case=case):
                values = {"enabled": False} if case == "disabled" else {}
                if case == "discovery_off":
                    values["group_discovery_enabled"] = False
                plugin, config = self.plugin(values)
                event = self.event(
                    private=case == "private",
                    platform="discord" if case == "non_tg" else "telegram",
                )
                if case == "bot":
                    event.message_obj.raw_message.message.from_user.is_bot = True
                if case == "stopped":
                    event.stop_event()
                self.assertEqual(await self.consume(plugin, event), [])
                self.assertEqual(plugin.store.list_discovered_groups(), [])
                self.assertFalse(event.call_llm)
                self.assert_metadata_only(plugin)
                config.save_config.assert_not_called()

    async def test_commands_discover_but_do_not_reply_or_record_body(self):
        plugin, config = self.plugin({"group_ids": ["-10"]})
        event = self.event(text="/help")
        self.assertEqual(await self.consume(plugin, event), [])
        self.assertFalse(event.call_llm)
        self.assertIsNotNone(
            plugin.store.get_discovered_group(event.unified_msg_origin)
        )
        self.assert_metadata_only(plugin)
        config.save_config.assert_not_called()

    async def test_selected_and_manual_groups_are_union_without_auto_selection(self):
        selected = "bot:GroupMessage:-20#7"
        plugin, config = self.plugin(
            {"group_ids": ["-10"], "selected_groups": [selected]}
        )
        original = copy.deepcopy(dict(config))
        for index, group in enumerate(("-10", "-20#7"), 1):
            event = self.event(index, group=group, direct=True)
            self.assertEqual(len(await self.consume(plugin, event)), 1)
            self.assertTrue(event.call_llm)
        unselected = self.event(3, group="-30", direct=True)
        self.assertEqual(await self.consume(plugin, unselected), [])
        self.assertFalse(unselected.call_llm)
        self.assertIsNone(plugin.store.active_window(unselected.unified_msg_origin))
        self.assertEqual(dict(config), original)
        self.assertEqual(plugin.settings.selected_groups, (selected,))
        self.assertIn(
            unselected.unified_msg_origin, config.schema["selected_groups"]["options"]
        )
        config.save_config.assert_not_called()

    async def test_refresh_updates_schema_only_and_keeps_missing_selected_candidate(
        self,
    ):
        missing = "bot:GroupMessage:-99#3"
        plugin, config = self.plugin({"selected_groups": [missing]})
        original = copy.deepcopy(dict(config))
        event = self.event(name="Discovery Name")
        await self.consume(plugin, event)
        field = config.schema["selected_groups"]
        self.assertEqual(field["options"], [event.unified_msg_origin, missing])
        self.assertEqual(
            field["labels"], ["Discovery Name · -10 · Telegram/bot", missing]
        )
        self.assertEqual(dict(config), original)
        config.save_config.assert_not_called()
        self.assertFalse(plugin._in_scope(event))

    async def test_initialize_loads_persisted_candidates_after_restart(self):
        path = Path(self.temp.name) / "restart" / "chat.sqlite3"
        first, _ = self.plugin(path=path)
        event = self.event(name="持久候选")
        await self.consume(first, event)
        await first.terminate()
        second, config = self.plugin(open_store=False)
        tools = types.SimpleNamespace(get_data_dir=Mock(return_value=path.parent))
        with patch.object(runtime, "StarTools", tools):
            await second.initialize()
        tools.get_data_dir.assert_called_once_with("Astrbot_GroupChatLite")
        self.assertEqual(
            config.schema["selected_groups"]["options"], [event.unified_msg_origin]
        )
        self.assertIn("持久候选", config.schema["selected_groups"]["labels"][0])
        self.assertFalse(second._tasks)
        self.assert_metadata_only(second)
        self.assertEqual(second.settings.selected_groups, ())
        config.save_config.assert_not_called()

    async def test_name_change_refreshes_label_inside_throttle_interval(self):
        plugin, config = self.plugin()
        first = self.event(name="旧名")
        await self.consume(plugin, first)
        self.now += 1
        self.monotonic += 1
        await self.consume(plugin, self.event(2, name="新名"))
        field = config.schema["selected_groups"]
        self.assertEqual(field["options"], [first.unified_msg_origin])
        self.assertIn("新名", field["labels"][0])
        self.assertNotIn("旧名", field["labels"][0])
        saved = plugin.store.get_discovered_group(first.unified_msg_origin)
        self.assertEqual((saved["first_seen"], saved["last_seen"]), (1000, 1001))
        config.save_config.assert_not_called()

    async def test_empty_name_does_not_clear_persisted_name_or_label(self):
        plugin, config = self.plugin()
        first = self.event(name="保留名")
        await self.consume(plugin, first)
        self.now += 1
        self.monotonic += 1
        await self.consume(plugin, self.event(2, name=""))
        self.assertEqual(
            plugin.store.get_discovered_group(first.unified_msg_origin)["display_name"],
            "保留名",
        )
        self.assertIn("保留名", config.schema["selected_groups"]["labels"][0])
        self.assert_metadata_only(plugin)

    async def test_raw_title_and_topic_are_used_when_group_name_is_absent(self):
        plugin, config = self.plugin()
        event = self.event(group="-20#7", name="")
        event.message_obj.raw_message.message.chat = types.SimpleNamespace(
            title="群名称"
        )
        event.message_obj._telegram_topic_name = "话题名称"
        await self.consume(plugin, event)
        self.assertEqual(
            plugin.store.get_discovered_group(event.unified_msg_origin)["display_name"],
            "群名称-话题名称",
        )
        self.assertIn("-20#7", config.schema["selected_groups"]["labels"][0])

    async def test_same_name_events_are_throttled_without_losing_later_last_seen(self):
        plugin, config = self.plugin()
        event = self.event()
        await self.consume(plugin, event)
        with patch.object(
            plugin, "_refresh_group_options", wraps=plugin._refresh_group_options
        ) as refresh:
            self.now += 10
            self.monotonic += 10
            await self.consume(plugin, self.event(2))
            self.assertEqual(
                plugin.store.get_discovered_group(event.unified_msg_origin)[
                    "last_seen"
                ],
                1000,
            )
            self.now += 60
            self.monotonic += 60
            await self.consume(plugin, self.event(3))
            self.assertEqual(
                plugin.store.get_discovered_group(event.unified_msg_origin)[
                    "last_seen"
                ],
                1070,
            )
            refresh.assert_not_called()
        self.assertEqual(len(plugin.store.list_discovered_groups()), 1)
        config.save_config.assert_not_called()

    async def test_recently_active_old_candidate_returns_to_bounded_options(self):
        plugin, config = self.plugin()
        for index in range(1, 202):
            group = str(-1000 - index)
            plugin.store.observe_group(
                f"bot:GroupMessage:{group}", "bot", group, f"Group {index}", index
            )
        plugin._refresh_group_options()
        event = self.event(group="-1001", name="Group 1")
        self.assertEqual(len(config.schema["selected_groups"]["options"]), 200)
        self.assertNotIn(
            event.unified_msg_origin, config.schema["selected_groups"]["options"]
        )
        self.assertEqual(await self.consume(plugin, event), [])
        options = config.schema["selected_groups"]["options"]
        self.assertEqual(len(options), 200)
        self.assertEqual(options[0], event.unified_msg_origin)
        self.assertFalse(event.call_llm)
        self.assertEqual(plugin.settings.selected_groups, ())
        self.assertEqual(dict(config), {})
        config.save_config.assert_not_called()
        self.assert_metadata_only(plugin)

    async def test_malformed_identity_and_missing_message_id_are_not_discovered(self):
        for case in ("positive", "wildcard", "bad_topic", "wrong_umo", "missing_id"):
            with self.subTest(case=case):
                plugin, _ = self.plugin()
                group = {"positive": "10", "wildcard": "*", "bad_topic": "-10#0"}.get(
                    case, "-10"
                )
                event = self.event(group=group)
                if case == "wrong_umo":
                    event.unified_msg_origin = "other:GroupMessage:-10"
                if case == "missing_id":
                    event.message_obj.message_id = None
                self.assertEqual(await self.consume(plugin, event), [])
                self.assertEqual(plugin.store.list_discovered_groups(), [])
                self.assert_metadata_only(plugin)


if __name__ == "__main__":
    unittest.main()
