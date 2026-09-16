"""Offline successful-delivery and lifecycle tests."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

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

    async def send_message(self, *args, **kwargs):
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


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
        original = self.client.send_message

        async def stopped_during_send(*args, **kwargs):
            nonlocal allowed
            result = await original(*args, **kwargs)
            allowed = False
            return result

        self.client.send_message = stopped_during_send
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


if __name__ == "__main__":
    unittest.main()
