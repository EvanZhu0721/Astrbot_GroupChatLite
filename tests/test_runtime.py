"""Offline runtime integration: real plugin/store, narrow AstrBot API doubles."""

import asyncio
import dataclasses
import importlib.util
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]


class Component:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class At(Component):
    pass


class Reply(Component):
    pass


class Image(Component):
    async def convert_to_file_path(self):
        return self.path


class Record(Component):
    async def convert_to_file_path(self):
        return self.path


class TextPart(Component):
    pass


class Star:
    def __init__(self, context):
        self.context = context


class Filters:
    EventMessageType = types.SimpleNamespace(GROUP_MESSAGE="group")

    def __getattr__(self, name):
        def decorator(*args, **kwargs):
            def apply(func):
                return func

            return apply

        return decorator


def module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


STUBS = {
    "astrbot": module("astrbot"),
    "astrbot.api": module(
        "astrbot.api",
        AstrBotConfig=dict,
        logger=types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None
        ),
    ),
    "astrbot.api.event": module(
        "astrbot.api.event", AstrMessageEvent=object, filter=Filters()
    ),
    "astrbot.api.message_components": module(
        "astrbot.api.message_components", At=At, Reply=Reply, Image=Image, Record=Record
    ),
    "astrbot.api.star": module(
        "astrbot.api.star", Context=object, Star=Star, StarTools=object
    ),
    "astrbot.core.agent.message": module(
        "astrbot.core.agent.message", TextPart=TextPart
    ),
    "astrbot.core.pipeline.process_stage.follow_up": module(
        "astrbot.core.pipeline.process_stage.follow_up", _ACTIVE_AGENT_RUNNERS={}
    ),
}
package = module("_gcl_runtime_tests")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    package.__name__ + ".main", ROOT / "main.py"
)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
with patch.dict(sys.modules, STUBS):
    spec.loader.exec_module(runtime)


class Event:
    def __init__(
        self,
        ident,
        text="hello",
        group="-10",
        platform="telegram",
        private=False,
        direct=False,
    ):
        self.group, self.platform, self.private = group, platform, private
        self.unified_msg_origin = "bot:GroupMessage:" + group
        self.text = text
        self.extra = {}
        self.call_llm = False
        self._has_send_oper = False
        self.stopped = False
        self.message_obj = types.SimpleNamespace(
            message_id=str(ident),
            timestamp=1000,
            message=[At(qq="bot")] if direct else [],
            raw_message=types.SimpleNamespace(
                message=types.SimpleNamespace(
                    text=text, from_user=types.SimpleNamespace(is_bot=False), date=None
                )
            ),
        )

    def get_platform_name(self):
        return self.platform

    def get_platform_id(self):
        return "bot"

    def is_private_chat(self):
        return self.private

    def get_group_id(self):
        return self.group

    def get_self_id(self):
        return "bot"

    def get_sender_id(self):
        return "human"

    def get_sender_name(self):
        return "speaker"

    def get_message_str(self):
        return self.text

    def get_extra(self, name, default=None):
        return self.extra.get(name, default)

    def set_extra(self, name, value):
        self.extra[name] = value

    def is_stopped(self):
        return self.stopped

    def stop_event(self):
        self.stopped = True

    def request_llm(self, **kwargs):
        return types.SimpleNamespace(
            **kwargs, extra_user_content_parts=[], system_prompt=""
        )


class Context:
    def __init__(self):
        self.config = {
            "provider_ltm_settings": {"group_icl_enable": False},
            "platform_settings": {},
        }
        self.stars = []
        self.original = types.SimpleNamespace(
            history='[{"role":"user","content":"OLD_PRIVATE"}]',
            persona_id="persona",
            cid="cid",
        )
        self.persona_manager = types.SimpleNamespace(
            resolve_selected_persona=AsyncMock(return_value=(None, None, None, False))
        )
        self.conversation_manager = types.SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="cid"),
            get_conversation=AsyncMock(side_effect=lambda *a, **k: self.original),
        )

    def get_config(self, **kwargs):
        return self.config

    def get_all_stars(self):
        return self.stars


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.context = Context()
        self.plugin = runtime.AstrbotGroupChatLite(
            self.context,
            {
                "group_ids": ["-10", "-20"],
                "merge_wait_seconds": 0.01,
                "decision_cooldown_seconds": 0,
                "reply_cooldown_seconds": 0,
            },
        )
        self.plugin.store = runtime.Store(Path(self.temp.name) / "chat.sqlite3")
        self.plugin._media = runtime.OwnedMediaCache(
            Path(self.temp.name) / "image_cache", clock=lambda: self.plugin._monotonic()
        )
        self.now = 1000.0
        self.plugin._clock = lambda: self.now
        self.plugin._decide = AsyncMock(return_value=True)
        self.netguard = patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network forbidden")
        )
        self.netguard.start()
        self.modules = patch.dict(sys.modules, STUBS)
        self.modules.start()

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.modules.stop()
        self.netguard.stop()
        self.temp.cleanup()

    async def consume(self, event, reply="answer"):
        requests = []
        async for req in self.plugin.on_group_message(event):
            requests.append(req)
            await self.plugin.on_llm_request(event, req)
            await self.plugin.on_llm_response(
                event, types.SimpleNamespace(role="assistant", completion_text=reply)
            )
        return requests

    async def test_raw_identity_and_true_reply_are_stored(self):
        event = Event(900, direct=True)
        event.get_sender_name = lambda: "Unknown"
        raw = event.message_obj.raw_message.message
        raw.from_user = types.SimpleNamespace(
            id=42, full_name="Alice Example", is_bot=False
        )
        raw.reply_to_message = types.SimpleNamespace(
            message_id=899, from_user=types.SimpleNamespace(id=77)
        )
        await self.consume(event)
        window = self.plugin.store.active_window(event.unified_msg_origin)
        human, bot = self.plugin.store.window_messages(
            event.unified_msg_origin, window["id"]
        )
        self.assertEqual(human["sender_id"], "42")
        self.assertEqual(human["sender_name"], "Alice Example")
        self.assertEqual(human["reply_to_message_id"], "899")
        self.assertEqual(human["reply_to_sender_id"], "77")
        # A generated final without a Telegram receipt is not a confirmed response link.
        self.assertEqual(bot["response_to_sender_id"], "")

    async def test_anonymous_chat_identity_and_topic_root_are_not_person_reply(self):
        event = Event(901)
        raw = event.message_obj.raw_message.message
        raw.sender_chat = types.SimpleNamespace(id=-900, title="Anonymous group")
        raw.is_topic_message = True
        raw.message_thread_id = 100
        raw.reply_to_message = types.SimpleNamespace(
            message_id=100, from_user=types.SimpleNamespace(id=77)
        )
        identity = self.plugin._message_identity(event)
        self.assertEqual(identity["sender_id"], "-900")
        self.assertEqual(identity["sender_name"], "Anonymous group")
        self.assertEqual(identity["reply_to_message_id"], "")
        self.assertEqual(identity["reply_to_sender_id"], "")

    async def test_unknown_display_name_falls_back_to_stable_sender_id(self):
        event = Event(902)
        event.get_sender_name = lambda: "Unknown"
        self.assertEqual(self.plugin._message_identity(event)["sender_name"], "human")

    async def test_successful_multisegment_delivery_keeps_original_response_target(
        self,
    ):
        event = Event(903, direct=True)
        callbacks = {}

        def install(*args, **kwargs):
            callbacks.update(kwargs)
            return None

        with patch.object(runtime, "install_send_observer", side_effect=install):
            await self.plugin.observe_external_replies(event)
        generator = self.plugin.on_group_message(event)
        await anext(generator)
        for ident in (1001, 1002):
            receipt = types.SimpleNamespace(
                message_id=ident, from_user=types.SimpleNamespace(id=9000)
            )
            callbacks["on_delivery"](receipt, str(ident), True)
        event.get_sender_id = lambda: "later-mutation"
        await self.plugin.on_llm_response(
            event, types.SimpleNamespace(role="assistant", completion_text="answer")
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(generator)
        window = self.plugin.store.active_window(event.unified_msg_origin)
        records = self.plugin.store.window_messages(
            event.unified_msg_origin, window["id"]
        )
        bot = records[-1]
        self.assertEqual(bot["sender_id"], "9000")
        self.assertEqual(bot["response_to_message_id"], "903")
        self.assertEqual(bot["response_to_sender_id"], "human")
        self.assertEqual(bot["reply_to_message_id"], "")

    async def test_out_of_scope_and_commands_are_inert(self):
        for event in [
            Event(1, platform="discord"),
            Event(2, private=True),
            Event(3, group="-99"),
            Event(4, text="/help"),
        ]:
            self.assertEqual(await self.consume(event), [])
            self.assertFalse(event.call_llm)
        self.plugin.settings = dataclasses.replace(self.plugin.settings, group_ids=())
        event = Event(5, direct=True)
        self.assertEqual(await self.consume(event), [])
        self.assertFalse(event.call_llm)
        self.plugin._decide.assert_not_awaited()

    async def test_bot_sender_is_inert(self):
        event = Event(1, direct=True)
        event.message_obj.raw_message.message.from_user.is_bot = True
        self.assertEqual(await self.consume(event), [])
        self.assertFalse(event.call_llm)
        self.plugin._decide.assert_not_awaited()

    async def test_continuous_arrivals_have_bounded_merge_wait(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings,
            merge_wait_seconds=0.05,
            max_merge_wait_seconds=0.08,
            decision_cooldown_seconds=0,
            reply_cooldown_seconds=0,
        )
        tasks = []
        completed_during_chatter = False
        for ident in range(1, 13):
            tasks.append(asyncio.create_task(self.consume(Event(ident))))
            await asyncio.sleep(0.02)
            if any(task.done() and task.result() for task in tasks):
                completed_during_chatter = True
        results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
        self.assertTrue(
            completed_during_chatter, "Continuous traffic must not starve replies"
        )
        count = sum(map(len, results))
        self.assertGreaterEqual(count, 2)
        self.assertLess(count, len(tasks))
        room = self.plugin._rooms[Event(1).unified_msg_origin]
        self.assertEqual(room.consumed_revision, 12)
        self.assertFalse(room.lock.locked())

    async def test_arrival_during_decision_gets_next_batch_and_direct_is_retained(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings,
            merge_wait_seconds=0,
            max_merge_wait_seconds=0,
            decision_cooldown_seconds=0,
            reply_cooldown_seconds=0,
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def decide(event, messages):
            if event.message_obj.message_id == "1":
                entered.set()
                await release.wait()
            return True

        self.plugin._decide = decide
        first = asyncio.create_task(self.consume(Event(1)))
        await entered.wait()
        second = asyncio.create_task(self.consume(Event(2)))
        await asyncio.sleep(0)
        release.set()
        self.assertEqual([len(x) for x in await asyncio.gather(first, second)], [1, 1])
        direct1 = asyncio.create_task(self.consume(Event(3, direct=True)))
        direct2 = asyncio.create_task(self.consume(Event(4, direct=True)))
        self.assertEqual(
            [len(x) for x in await asyncio.gather(direct1, direct2)], [1, 1]
        )

    async def test_conflicts_fail_before_claiming(self):
        self.context.config["provider_ltm_settings"]["group_icl_enable"] = True
        self.assertEqual(await self.consume(Event(1, direct=True)), [])
        self.context.config["provider_ltm_settings"]["group_icl_enable"] = False
        self.context.stars = [
            types.SimpleNamespace(
                name="astrbot_plugin_group_chat_plus", activated=True, config={}
            )
        ]
        self.assertEqual(await self.consume(Event(2, direct=True)), [])
        self.context.stars[0].activated = False
        self.assertEqual(len(await self.consume(Event(3, direct=True))), 1)

    async def test_cooldown_retains_only_latest_without_holding_lock(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings,
            merge_wait_seconds=0,
            max_merge_wait_seconds=0,
            decision_cooldown_seconds=0.12,
            reply_cooldown_seconds=0,
        )
        event = Event(1)
        room = self.plugin._rooms.setdefault(event.unified_msg_origin, runtime.Room())
        room.last_decision = self.plugin._monotonic()
        older = asyncio.create_task(self.consume(event))
        await asyncio.sleep(0.01)
        self.assertFalse(room.lock.locked())
        self.assertEqual(room.consumed_revision, 0)
        newer = asyncio.create_task(self.consume(Event(2)))
        results = await asyncio.wait_for(asyncio.gather(older, newer), 1)
        self.assertEqual([len(result) for result in results], [0, 1])
        self.plugin._decide.assert_awaited_once()
        self.assertEqual(room.pending, {})

    async def test_reference_is_independent_but_scores_and_mention_does_not(self):
        quote = Event(1)
        quote.message_obj.message = [Reply(sender_id="bot", id="90", chain=[])]
        quote.message_obj.raw_message.message.reply_to_message = types.SimpleNamespace(
            message_id=90, from_user=types.SimpleNamespace(id=123, username="bot")
        )
        captured = []

        async def decide(event, messages):
            captured.append(dict(event.get_extra(runtime.SNAPSHOT)["score_features"]))
            return True

        self.plugin._decide = AsyncMock(side_effect=decide)
        self.assertEqual(len(await self.consume(quote)), 1)
        self.plugin._decide.assert_awaited_once()
        self.assertEqual(captured[0]["reply_hops"], 1)
        self.assertEqual(len(await self.consume(Event(2, direct=True))), 1)
        self.plugin._decide.assert_awaited_once()

    async def test_quote_remains_independent_when_later_ordinary_arrives(self):
        quote = Event(101)
        quote.message_obj.message = [Reply(sender_id="bot", id="900", chain=[])]
        ordinary = Event(102)
        first, second = await asyncio.gather(
            self.consume(quote), self.consume(ordinary)
        )
        self.assertEqual([len(first), len(second)], [1, 1])
        self.assertEqual(self.plugin._decide.await_count, 2)

    async def test_pending_features_exclude_old_processed_and_freeze_batch(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings,
            merge_wait_seconds=0.02,
            max_merge_wait_seconds=0.04,
            decision_cooldown_seconds=0,
            reply_cooldown_seconds=0,
        )
        await self.consume(Event(1, direct=True))
        seen = []

        async def decide(event, messages):
            seen.append(dict(event.get_extra(runtime.SNAPSHOT)["score_features"]))
            return False

        self.plugin._decide = decide
        first = asyncio.create_task(self.consume(Event(2)))
        await asyncio.sleep(0)
        second = asyncio.create_task(self.consume(Event(3)))
        await asyncio.gather(first, second)
        self.assertEqual([entry["pending_count"] for entry in seen], [2])
        await self.consume(Event(4))
        self.assertEqual([entry["pending_count"] for entry in seen], [2, 1])

    async def test_generated_text_without_receipt_is_not_a_successful_reply_target(
        self,
    ):
        event = Event(1, direct=True)
        await self.consume(event)
        state = self.plugin._scoring_state.snapshot(
            event.unified_msg_origin, "1", "human"
        )
        self.assertFalse(state.recent_target)
        self.assertEqual(
            self.plugin._rooms[event.unified_msg_origin].last_reply, -float("inf")
        )

    async def test_score_features_do_not_change_during_model_wait(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings, merge_wait_seconds=0, max_merge_wait_seconds=0
        )
        entered, release = asyncio.Event(), asyncio.Event()
        frozen = []

        async def decide(event, messages):
            features = event.get_extra(runtime.SNAPSHOT)["score_features"]
            frozen.append(dict(features))
            if event.message_obj.message_id == "1":
                entered.set()
                await release.wait()
                self.assertEqual(features, frozen[0])
            return False

        self.plugin._decide = decide
        first = asyncio.create_task(self.consume(Event(1)))
        await entered.wait()
        second = asyncio.create_task(self.consume(Event(2)))
        await asyncio.sleep(0)
        self.now += 5
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual([f["pending_count"] for f in frozen], [1, 1])
        self.assertEqual([f["recent_message_count"] for f in frozen], [1, 2])
        self.assertEqual([f["now"] for f in frozen], [1000, 1005])

    async def test_cooldown_expiry_and_cancel_clear_pending(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings,
            merge_wait_seconds=0,
            max_merge_wait_seconds=0,
            decision_cooldown_seconds=1,
            reply_cooldown_seconds=0,
            stale_message_seconds=0.03,
        )
        event = Event(1)
        room = self.plugin._rooms.setdefault(event.unified_msg_origin, runtime.Room())
        room.last_decision = self.plugin._monotonic()
        self.assertEqual(await asyncio.wait_for(self.consume(event), 1), [])
        self.assertFalse(room.lock.locked())
        self.assertEqual(room.pending, {})
        task = asyncio.create_task(self.consume(Event(2)))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(room.lock.locked())
        self.assertEqual(room.pending, {})

    async def test_debounce_merges_ordinary_but_keeps_direct(self):
        first = asyncio.create_task(self.consume(Event(1, "first")))
        await asyncio.sleep(0)
        second = asyncio.create_task(self.consume(Event(2, "second")))
        a, b = await asyncio.gather(first, second)
        self.assertEqual((len(a), len(b)), (0, 1))
        self.assertEqual(self.plugin._decide.await_count, 1)
        x, y = await asyncio.gather(
            self.consume(Event(3, direct=True)), self.consume(Event(4, direct=True))
        )
        self.assertEqual((len(x), len(y)), (1, 1))

    async def test_one_request_final_saved_and_no_old_history_or_media_loss(self):
        event = Event(1, direct=True)
        image_path = Path(self.temp.name) / "image.png"
        image_path.write_bytes(b"offline test image")
        event.message_obj.message += [
            Image(path=str(image_path)),
            Reply(sender_id="someone", chain=[Record(path="audio.wav")]),
        ]
        generator = self.plugin.on_group_message(event)
        req = await anext(generator)
        self.assertTrue(event.call_llm)
        self.assertEqual(req.conversation.persona_id, "persona")
        self.assertEqual(req.conversation.history, "[]")
        self.assertIn("OLD_PRIVATE", self.context.original.history)
        req.system_prompt = "native persona"
        req.contexts = [{"role": "assistant", "content": "persona example"}]
        req.extra_user_content_parts.append(TextPart(text="current quote"))
        await self.plugin.on_llm_request(event, req)
        self.assertIsNone(req.conversation)
        self.assertEqual(req.system_prompt, "native persona")
        self.assertEqual(len(req.image_urls), 1)
        self.assertNotEqual(req.image_urls, [str(image_path)])
        self.assertEqual(Path(req.image_urls[0]).read_bytes(), image_path.read_bytes())
        self.assertEqual(req.audio_urls, ["audio.wav"])
        self.assertEqual(req.extra_user_content_parts[0].text, "current quote")
        await self.plugin.on_llm_response(
            event, types.SimpleNamespace(role="assistant", completion_text="final")
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(generator)
        window = self.plugin.store.active_window(event.unified_msg_origin)
        records = self.plugin.store.window_messages(
            event.unified_msg_origin, window["id"]
        )
        self.assertEqual([r["role"] for r in records], ["user", "assistant"])
        self.assertFalse(self.plugin._rooms[event.unified_msg_origin].lock.locked())

    async def test_tool_cannot_read_other_scope_by_id(self):
        await self.consume(Event(1, "secret-other", group="-20", direct=True))
        event = Event(2, direct=True)
        await self.consume(event)
        result = await self.plugin.groupchat_history(
            event, action="read", message_ids="1,2"
        )
        self.assertNotIn("secret-other", result)

    async def test_cancellation_and_failure_release_room(self):
        entered = asyncio.Event()

        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()

        self.plugin._decide = blocked
        event = Event(1)
        task = asyncio.create_task(self.consume(event))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        room = self.plugin._rooms[event.unified_msg_origin]
        self.assertFalse(room.active)
        self.assertFalse(room.lock.locked())
        self.plugin._decide = AsyncMock(side_effect=RuntimeError("failure"))
        self.assertEqual(await self.consume(Event(2)), [])
        self.assertFalse(room.lock.locked())

    async def test_terminate_cancels_owned_handlers(self):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings, merge_wait_seconds=10
        )
        task = asyncio.create_task(self.consume(Event(1)))
        await asyncio.sleep(0)
        await self.plugin.terminate()
        self.assertTrue(task.cancelled())
        self.assertFalse(self.plugin._tasks)
        self.assertFalse(self.plugin._handlers)

    async def test_old_final_stays_with_original_window(self):
        event = Event(1, direct=True)
        generator = self.plugin.on_group_message(event)
        await anext(generator)
        old_id = event.get_extra(runtime.MARKER)["window_id"]
        self.now += 1000
        newer = self.plugin.store.add_human(
            event.unified_msg_origin,
            "tg:2",
            "new topic",
            self.now,
            self.now,
            self.plugin.settings.idle_seconds,
        )
        self.assertNotEqual(old_id, newer["window"]["id"])
        await self.plugin.on_llm_response(
            event, types.SimpleNamespace(role="assistant", completion_text="old answer")
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(generator)
        old = self.plugin.store.window_messages(event.unified_msg_origin, old_id)
        new = self.plugin.store.window_messages(
            event.unified_msg_origin, newer["window"]["id"]
        )
        self.assertEqual(old[-1]["text"], "old answer")
        self.assertNotIn("old answer", [x["text"] for x in new])

    async def test_aborted_final_not_saved(self):
        event = Event(1, direct=True)
        generator = self.plugin.on_group_message(event)
        await anext(generator)
        registry = STUBS[
            "astrbot.core.pipeline.process_stage.follow_up"
        ]._ACTIVE_AGENT_RUNNERS
        registry[event.unified_msg_origin] = types.SimpleNamespace(
            was_aborted=lambda: True
        )
        try:
            await self.plugin.on_llm_response(
                event,
                types.SimpleNamespace(role="assistant", completion_text="abort notice"),
            )
            with self.assertRaises(StopAsyncIteration):
                await anext(generator)
        finally:
            registry.clear()
        window = self.plugin.store.active_window(event.unified_msg_origin)
        self.assertEqual(
            len(
                self.plugin.store.window_messages(
                    event.unified_msg_origin, window["id"]
                )
            ),
            1,
        )

    def seed_summary_window(self):
        umo = Event(1).unified_msg_origin
        for index in range(6):
            saved = self.plugin.store.add_human(
                umo,
                f"tg:{index}",
                "context",
                1000 + index,
                1000 + index,
                self.plugin.settings.idle_seconds,
            )
        self.now = 3000
        self.plugin.store.idle_close(self.now, self.plugin.settings.idle_seconds)
        return self.plugin.store.get_window(umo, saved["window"]["id"])

    async def test_summary_cas_rejects_change_and_new_message_is_not_blocked(self):
        window = self.seed_summary_window()
        entered, release = asyncio.Event(), asyncio.Event()

        async def summarize(**kwargs):
            entered.set()
            await release.wait()
            return types.SimpleNamespace(completion_text="stale summary")

        self.plugin._provider = AsyncMock(
            return_value=types.SimpleNamespace(text_chat=summarize)
        )
        task = self.plugin._spawn(self.plugin._summarize(window))
        await entered.wait()
        self.plugin.store.add_bot(
            window["umo"], window["id"], "late-final", "late", 3000, 3000
        )
        event = Event(99, "new current", direct=True)
        event.message_obj.timestamp = self.now
        self.assertEqual(len(await asyncio.wait_for(self.consume(event), 0.5)), 1)
        release.set()
        await task
        self.assertIsNone(self.plugin.store.get_summary(window["umo"], window["id"]))

    async def test_summary_failure_attempts_are_bounded(self):
        window = self.seed_summary_window()
        call = AsyncMock(side_effect=RuntimeError("offline provider failure"))
        self.plugin._provider = AsyncMock(
            return_value=types.SimpleNamespace(text_chat=call)
        )
        for _ in range(5):
            await self.plugin._summary_tick()
            if self.plugin._tasks:
                await asyncio.gather(*list(self.plugin._tasks))
            self.now += 1000
        self.assertEqual(call.await_count, self.plugin.settings.summary_attempts)
        self.assertIsNone(self.plugin.store.get_summary(window["umo"], window["id"]))

    async def test_summary_tick_does_not_close_busy_room(self):
        event = Event(1, direct=True)
        generator = self.plugin.on_group_message(event)
        await anext(generator)
        self.now += 1000
        await self.plugin._summary_tick()
        self.assertIsNotNone(self.plugin.store.active_window(event.unified_msg_origin))
        self.assertFalse(self.plugin._tasks)
        await generator.aclose()
        self.assertFalse(self.plugin._rooms[event.unified_msg_origin].lock.locked())


if __name__ == "__main__":
    unittest.main()
