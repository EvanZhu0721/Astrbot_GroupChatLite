"""External text history integration; fake Telegram receipts, no network."""

import types
import unittest
from unittest.mock import AsyncMock

import test_runtime as support
from test_runtime import Event, runtime


class FakeClient:
    def __init__(self, event):
        self.event = event

    async def send_message(self, **kwargs):
        return await self.event.deliver(**kwargs)


class SendingEvent(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.receipt_id = 100
        self.fail = False
        self.swallow = False
        self.client = FakeClient(self)

    async def deliver(self, **kwargs):
        if self.fail:
            raise RuntimeError("delivery failed")
        chat, _, thread = self.group.partition("#")
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(id=int(chat), is_forum=bool(thread)),
            message_id=self.receipt_id,
            text=kwargs["text"],
            message_thread_id=int(thread) if thread else None,
            is_topic_message=bool(thread),
        )

    async def send(self, text):
        try:
            result = await self.client.send_message(text=text)
        except RuntimeError:
            if not self.swallow:
                raise
            return None
        self._has_send_oper = True
        return result


class ExternalHistoryTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = support.RuntimeTests.asyncSetUp
    asyncTearDown = support.RuntimeTests.asyncTearDown
    consume = support.RuntimeTests.consume

    def records(self, event, window=None):
        if window is None:
            window = self.plugin.store.active_window(event.unified_msg_origin)
        if not window:
            return []
        return self.plugin.store.window_messages(event.unified_msg_origin, window["id"])

    async def test_observation_is_inert_until_verified_send(self):
        event = SendingEvent(1, text="question")
        await self.plugin.observe_external_replies(event)
        self.assertEqual(self.records(event), [])
        self.assertFalse(event.call_llm)
        self.plugin._decide.assert_not_awaited()
        event.text = "mutated by another plugin"
        await event.send("actual answer")
        self.assertEqual(
            [(r["role"], r["text"]) for r in self.records(event)],
            [("user", "question"), ("assistant", "actual answer")],
        )
        self.assertEqual(await self.consume(event), [])
        self.plugin._decide.assert_not_awaited()

    async def test_next_request_sees_external_exchange(self):
        event = SendingEvent(1, text="original question")
        await self.plugin.observe_external_replies(event)
        await event.send("external answer")
        requests = await self.consume(Event(2, text="follow up", direct=True))
        history = str(requests[0].extra_user_content_parts[0].text)
        self.assertIn("original question", history)
        self.assertIn("external answer", history)

    async def test_ordinary_decision_receives_external_exchange(self):
        provider = types.SimpleNamespace(
            text_chat=AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text='{"score":0,"reason":"不参与"}'
                )
            )
        )
        self.plugin._provider = AsyncMock(return_value=provider)
        self.plugin._decide = types.MethodType(
            runtime.AstrbotGroupChatLite._decide, self.plugin
        )
        event = SendingEvent(1, text="original question")
        await self.plugin.observe_external_replies(event)
        await event.send("external answer")
        provider.text_chat.assert_not_awaited()
        followup = Event(2, text="ordinary continuation")
        self.assertEqual(await self.consume(followup), [])
        provider.text_chat.assert_awaited_once()
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        self.assertIn("original question", prompt)
        self.assertIn("external answer", prompt)
        self.assertIn("ordinary continuation", prompt)
        self.assertIn('"role":"assistant"', prompt.replace(" ", ""))
        current = self.records(followup)[-1]
        self.assertEqual(current["text"], "ordinary continuation")
        self.assertIn(f'"current_input_message_id":{current["id"]}', prompt)

    async def test_reinstall_and_duplicate_receipt_do_not_duplicate(self):
        event = SendingEvent(1)
        await self.plugin.observe_external_replies(event)
        first = event.send
        await self.plugin.observe_external_replies(event)
        self.assertIs(event.send, first)
        await event.send("answer")
        await event.send("answer")
        self.assertEqual(len(self.records(event)), 2)

    async def test_own_marker_uses_existing_writeback_only(self):
        event = SendingEvent(1, direct=True)
        await self.plugin.observe_external_replies(event)
        generator = self.plugin.on_group_message(event)
        request = await anext(generator)
        await self.plugin.on_llm_request(event, request)
        await event.send("own answer")
        self.assertEqual(len(self.records(event)), 1)
        await self.plugin.on_llm_response(
            event, types.SimpleNamespace(role="assistant", completion_text="own answer")
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(generator)
        self.assertEqual(len(self.records(event)), 2)
        self.assertFalse(
            self.records(event)[1]["source_message_id"].startswith("external:")
        )

    async def test_raise_or_swallowed_failure_never_creates_history(self):
        for swallow in (False, True):
            event = SendingEvent(1)
            event.fail, event.swallow = True, swallow
            await self.plugin.observe_external_replies(event)
            original_client = event.client
            if swallow:
                self.assertIsNone(await event.send("unsent"))
            else:
                with self.assertRaises(RuntimeError):
                    await event.send("unsent")
            self.assertIs(event.client, original_client)
            self.assertEqual(self.records(event), [])

    async def test_scope_and_stopped_events_have_no_observer(self):
        events = [
            SendingEvent(1, group="-99"),
            SendingEvent(2, private=True),
            SendingEvent(3, platform="discord"),
            SendingEvent(4),
            SendingEvent(5),
        ]
        events[3].stopped = True
        events[4].message_obj.raw_message.message.from_user.is_bot = True
        for event in events:
            await self.plugin.observe_external_replies(event)
            self.assertIsNone(event.get_extra(runtime.OBSERVER))
            self.assertFalse(event.call_llm)

    async def test_stop_after_install_prevents_history(self):
        event = SendingEvent(1)
        await self.plugin.observe_external_replies(event)
        event.stop_event()
        await event.send("answer")
        self.assertEqual(self.records(event), [])

    async def test_scoring_tracks_success_only_and_deduplicates_human_arrival(self):
        event = SendingEvent(1)
        await self.plugin.observe_external_replies(event)
        await self.plugin.observe_external_replies(event)
        state = self.plugin._scoring_state
        before = state.snapshot(event.unified_msg_origin, "1", "human")
        self.assertEqual(before.arrival_count, 1)
        self.assertFalse(before.recent_target)
        event.fail, event.swallow = True, True
        await event.send("unsent")
        self.assertFalse(
            state.snapshot(event.unified_msg_origin, "1", "human").recent_target
        )
        event.fail = False
        await event.send("delivered")
        self.assertTrue(
            state.snapshot(event.unified_msg_origin, "1", "human").recent_target
        )
        self.assertFalse(
            state.snapshot(event.unified_msg_origin, "1", "another").recent_target
        )
        self.assertFalse(
            state.snapshot("bot:GroupMessage:-20", "1", "human").recent_target
        )
        self.assertEqual(await self.consume(event), [])
        self.assertEqual(
            state.snapshot(event.unified_msg_origin, "1", "human").arrival_count, 1
        )

    async def test_late_response_updates_original_window_and_invalidates_summary(self):
        event = SendingEvent(1)
        await self.plugin.observe_external_replies(event)
        await event.send("first answer")
        umo = event.unified_msg_origin
        old = self.plugin.store.active_window(umo)
        self.plugin.store.idle_close(3000, 60)
        self.assertTrue(
            self.plugin.store.save_summary(umo, old["id"], "summary", old["version"])
        )
        newer = self.plugin.store.add_human(umo, "tg:2", "new window", 3000, 3000, 60)
        self.now = 3100
        event.receipt_id += 1
        await event.send("late answer")
        self.assertEqual(len(self.records(event, old)), 3)
        self.assertEqual(len(self.records(event, newer["window"])), 1)
        self.assertIsNone(self.plugin.store.get_summary(umo, old["id"]))

    async def test_terminate_detaches_before_store_close(self):
        event = SendingEvent(1)
        original = event.send
        await self.plugin.observe_external_replies(event)
        handle = event.get_extra(runtime.OBSERVER)
        await self.plugin.terminate()
        self.assertFalse(handle.active)
        self.assertEqual(event.send, original)
        self.assertEqual((await event.send("after unload")).text, "after unload")


if __name__ == "__main__":
    unittest.main()
