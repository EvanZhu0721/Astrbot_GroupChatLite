"""Offline successful-delivery and lifecycle tests."""

import asyncio
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

spec = importlib.util.spec_from_file_location(
    "lite_send_observer", Path(__file__).resolve().parents[1] / "send_observer.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def receipt(mid=1, text="delivered", chat=-100, thread=7, forum=True):
    return SimpleNamespace(
        message_id=mid,
        text=text,
        chat=SimpleNamespace(id=chat, is_forum=forum),
        message_thread_id=thread,
        is_topic_message=thread is not None,
    )


class Client:
    def __init__(self, results):
        self.results = iter(results)
        self.before_return = None

    async def send_message(self, *args, **kwargs):
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        if self.before_return:
            self.before_return()
        return result

    async def edit_message_text(self, *args, **kwargs):
        return await self.send_message(*args, **kwargs)


class Event:
    def __init__(self, client, group="-100#7"):
        self.client = client
        self.group = group

    def get_group_id(self):
        return self.group

    async def send(self, count=1):
        for _ in range(count):
            await self.client.send_message(text="original payload")
        return "original return"

    async def send_streaming(self):
        await self.client.send_message(text="draft")
        await self.client.edit_message_text(text="final")
        return "stream return"


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    def install(self, results, group="-100#7", predicate=lambda: True, callback=None):
        self.records = []
        self.client = Client(results)
        self.event = Event(self.client, group)
        self.handle = module.install_send_observer(
            self.event,
            callback or (lambda text, key: self.records.append((text, key))),
            predicate,
        )
        self.addCleanup(self.handle.detach)

    async def test_confirmed_text_and_return_preserved(self):
        self.install([receipt(text="actual returned text")])
        self.assertEqual(await self.event.send(), "original return")
        self.assertEqual(self.records, [("actual returned text", "telegram:-100:7:1")])
        self.assertIs(self.event.client, self.client)

    async def test_partial_failure_keeps_only_success(self):
        self.install([receipt(1), RuntimeError("private failure")])
        with self.assertRaises(RuntimeError):
            await self.event.send(2)
        self.assertEqual(len(self.records), 1)
        self.assertIs(self.event.client, self.client)

    async def test_cancel_restores_without_false_success(self):
        self.install([asyncio.CancelledError()])
        with self.assertRaises(asyncio.CancelledError):
            await self.event.send()
        self.assertEqual(self.records, [])
        self.assertIs(self.event.client, self.client)

    async def test_retry_and_duplicate_receipts(self):
        self.install([ValueError("markdown"), receipt(2), receipt(2)])
        original = self.handle._original_send

        async def retry():
            try:
                await original()
            except ValueError:
                await original()
            await original()

        self.handle._original_send = retry
        await self.event.send()
        self.assertEqual(len(self.records), 1)
        self.assertTrue(self.records[0][1].endswith(":2"))

    async def test_response_scope_and_missing_text_are_ignored(self):
        self.install([receipt(chat=-200), receipt(thread=8), receipt(text=None), None])
        await self.event.send(4)
        self.assertEqual(self.records, [])

    async def test_general_forum_topic_matches_native_conversion(self):
        self.install(
            [receipt(thread=1), receipt(thread=None), receipt(thread=8)], "-100"
        )
        await self.event.send(3)
        self.assertEqual(len(self.records), 1)
        self.assertIn(":general:", self.records[0][1])

    async def test_predicate_before_and_after_send(self):
        allowed = False
        self.install([receipt()], predicate=lambda: allowed)
        await self.event.send()
        self.assertEqual(self.records, [])
        allowed = True
        self.install([receipt()], predicate=lambda: allowed)

        def stopped_during_send():
            nonlocal allowed
            allowed = False

        self.client.before_return = stopped_during_send
        await self.event.send()
        self.assertEqual(self.records, [])

    async def test_other_task_calls_are_not_observed(self):
        self.install([receipt()])

        async def delegate():
            return await asyncio.create_task(
                self.event.client.send_message(text="other")
            )

        self.handle._original_send = delegate
        await self.event.send()
        self.assertEqual(self.records, [])

    async def test_callback_failure_does_not_change_send_result_or_leak_text(self):
        def bad_callback(*args):
            raise ValueError("private content")

        self.install([receipt()], callback=bad_callback)
        with self.assertLogs(module.logger, level="WARNING") as logs:
            self.assertEqual(await self.event.send(), "original return")
        self.assertNotIn("private content", " ".join(logs.output))

    async def test_detach_preserves_later_wrapper(self):
        self.install([receipt()])
        wrapper = self.event.send

        async def later():
            return await wrapper()

        self.event.send = later
        self.handle.detach()
        self.assertIs(self.event.send, later)
        await self.event.send()
        self.assertEqual(self.records, [])

    async def test_concurrent_sends_restore_shared_client(self):
        self.install([receipt(1), receipt(2)])
        await asyncio.gather(self.event.send(), self.event.send())
        self.assertEqual(len(self.records), 2)
        self.assertIs(self.event.client, self.client)
        self.handle.detach()
        self.assertNotIn("send", vars(self.event))

    async def test_stream_edits_refresh_delivery_without_external_history(self):
        self.install([receipt(21, "draft"), receipt(21, "final")])
        delivered = []
        self.handle.on_delivery = lambda value, key, stream: delivered.append(
            (value.text, key, stream)
        )
        self.assertEqual(await self.event.send_streaming(), "stream return")
        self.assertEqual([item[0] for item in delivered], ["draft", "final"])
        self.assertTrue(all(item[2] for item in delivered))
        self.assertEqual(delivered[0][1], delivered[1][1])
        self.assertEqual(self.records, [])
        self.assertIs(self.event.client, self.client)

    async def test_own_reply_delivery_has_separate_predicate(self):
        self.install([receipt()], predicate=lambda: False)
        delivered = []
        self.handle.should_observe = lambda: True
        self.handle.on_delivery = lambda *args: delivered.append(args)
        await self.event.send()
        self.assertEqual(len(delivered), 1)
        self.assertEqual(self.records, [])

    async def test_client_identity_never_changes_during_send(self):
        self.install([receipt(1), receipt(2), receipt(2)])
        observed_ids = []
        self.client.before_return = lambda: observed_ids.append(id(self.event.client))
        await self.event.send()
        await self.event.send_streaming()
        self.assertEqual(observed_ids, [id(self.client)] * 3)

    async def test_scope_outside_event_and_other_bot_are_transparent(self):
        self.install([receipt(1), receipt(2)])
        result = await self.client.send_message(text="outside")
        self.assertEqual(result.message_id, 1)
        other = Client([receipt(3)])

        async def mixed():
            await other.send_message(text="different bot")
            return await self.client.send_message(text="this bot")

        self.handle._original_send = mixed
        result = await self.event.send()
        self.assertEqual(result.message_id, 2)
        self.assertEqual(len(self.records), 1)
        self.assertTrue(self.records[0][1].endswith(":2"))

    async def test_async_callbacks_and_failures_preserve_result(self):
        self.install([receipt()])
        values = []

        async def delivered(value, key, stream):
            await asyncio.sleep(0)
            values.append(value.message_id)
            raise ValueError("private callback contents")

        self.handle.on_delivery = delivered
        with self.assertLogs(module.logger, level="WARNING") as logs:
            result = await self.event.send()
        self.assertEqual(result, "original return")
        self.assertEqual(values, [1])
        self.assertEqual(len(self.records), 1)
        self.assertNotIn("private callback", " ".join(logs.output))

    async def test_shared_class_reference_count_and_inherited_restore(self):
        class DerivedClient(Client):
            pass

        self.assertNotIn("send_message", vars(DerivedClient))
        one, two = (
            Event(DerivedClient([receipt(1)])),
            Event(DerivedClient([receipt(2)])),
        )
        first = module.install_send_observer(one, lambda *args: None, lambda: True)
        second = module.install_send_observer(two, lambda *args: None, lambda: True)
        self.addCleanup(first.detach)
        self.addCleanup(second.detach)
        installed = DerivedClient.send_message
        first.detach()
        self.assertIs(DerivedClient.send_message, installed)
        await two.send()
        second.detach()
        self.assertNotIn("send_message", vars(DerivedClient))
        self.assertNotIn("edit_message_text", vars(DerivedClient))

    async def test_detach_keeps_later_class_wrapper(self):
        class DerivedClient(Client):
            pass

        event = Event(DerivedClient([receipt()]))
        handle = module.install_send_observer(event, lambda *args: None, lambda: True)
        original = DerivedClient.send_message

        async def later(client, *args, **kwargs):
            return await original(client, *args, **kwargs)

        DerivedClient.send_message = later
        handle.detach()
        self.assertIs(DerivedClient.send_message, later)
        self.assertEqual(await event.send(), "original return")

    async def test_gc_restores_class_without_retaining_event(self):
        class DerivedClient(Client):
            pass

        event = Event(DerivedClient([]))
        handle = module.install_send_observer(event, lambda *args: None, lambda: True)
        event_ref, handle_ref = weakref.ref(event), weakref.ref(handle)
        del event, handle
        gc.collect()
        self.assertIsNone(event_ref())
        self.assertIsNone(handle_ref())
        self.assertNotIn(DerivedClient, module._CLIENT_PATCHES)
        self.assertNotIn("send_message", vars(DerivedClient))

    async def test_bettertg_wrapper_retains_original_client_identity(self):
        self.install([receipt(1), receipt(1)])

        class PacedClient:
            def __init__(self, client):
                self.original_client = client

            async def send_message(self, **kwargs):
                return await self.original_client.send_message(**kwargs)

            async def edit_message_text(self, **kwargs):
                return await self.original_client.edit_message_text(**kwargs)

        delivered = []
        self.handle.on_delivery = lambda *args: delivered.append(args)
        original = self.handle._original_streaming

        async def paced_stream():
            client = self.event.client
            try:
                self.event.client = PacedClient(client)
                return await original()
            finally:
                self.event.client = client

        self.handle._original_streaming = paced_stream
        await self.event.send_streaming()
        self.assertEqual(len(delivered), 2)
        self.assertIs(self.event.client, self.client)


if __name__ == "__main__":
    unittest.main()
