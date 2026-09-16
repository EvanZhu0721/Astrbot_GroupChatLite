"""Offline multimodal flow: local files, fake provider, real store and cache."""

import asyncio
import dataclasses
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock

import test_runtime as support
from test_runtime import Event, Image, Record, Reply, runtime


class MultimodalRuntimeTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = support.RuntimeTests.asyncSetUp
    asyncTearDown = support.RuntimeTests.asyncTearDown
    consume = support.RuntimeTests.consume

    def configure(self, **changes):
        self.plugin.settings = dataclasses.replace(
            self.plugin.settings,
            **dict(
                dict(
                    merge_wait_seconds=0,
                    decision_cooldown_seconds=0,
                    reply_cooldown_seconds=0,
                ),
                **changes,
            ),
        )
        self.provider = types.SimpleNamespace(
            text_chat=AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text='{"score":0.9,"reason":"相关"}',
                    reasoning_content=None,
                )
            )
        )
        self.plugin._provider = AsyncMock(return_value=self.provider)
        self.plugin._decide = types.MethodType(
            runtime.AstrbotGroupChatLite._decide, self.plugin
        )

    def image(self, name="image.png"):
        path = Path(self.temp.name) / name
        path.write_bytes(b"offline image placeholder; no decoder/network used")
        return Image(path=str(path))

    async def test_current_and_later_text_share_images_with_formal_agent(self):
        self.configure()
        photo = self.image()
        first = Event(1, "picture")
        first.message_obj.message.append(photo)
        req1 = (await self.consume(first))[0]
        self.assertEqual(
            self.provider.text_chat.await_args.kwargs["image_urls"], [photo.path]
        )
        self.assertEqual(req1.image_urls, [photo.path])
        second = Event(2, "what was in the image?")
        req2 = (await self.consume(second))[0]
        self.assertEqual(
            self.provider.text_chat.await_args.kwargs["image_urls"], [photo.path]
        )
        self.assertEqual(req2.image_urls, [photo.path])
        self.assertEqual(self.provider.text_chat.await_count, 2)

    async def test_merged_picture_survives_owner_becoming_text_message(self):
        self.configure(merge_wait_seconds=0.04)
        first = Event(1)
        photo = self.image()
        first.message_obj.message.append(photo)
        earlier = asyncio.create_task(self.consume(first))
        await asyncio.sleep(0.01)
        later = asyncio.create_task(self.consume(Event(2, "follow up")))
        results = await asyncio.gather(earlier, later)
        self.assertEqual([len(x) for x in results], [0, 1])
        self.assertEqual(results[1][0].image_urls, [photo.path])
        self.assertEqual(self.provider.text_chat.await_count, 1)

    async def test_previous_window_bridge_summary_and_image_reach_decision(self):
        self.configure()
        photo = self.image()
        event = Event(1, "old picture", direct=True)
        event.message_obj.message.append(photo)
        await self.consume(event)
        umo = event.unified_msg_origin
        window = self.plugin.store.active_window(umo)
        self.now += 700
        self.plugin.store.idle_close(self.now, self.plugin.settings.idle_seconds)
        window = self.plugin.store.get_window(umo, window["id"])
        self.assertTrue(
            self.plugin.store.save_summary(
                umo, window["id"], "SUMMARY_SENTINEL", window["version"]
            )
        )
        new = Event(2, "new topic")
        new.message_obj.timestamp = self.now
        req = (await self.consume(new))[0]
        kwargs = self.provider.text_chat.await_args.kwargs
        self.assertIn("SUMMARY_SENTINEL", kwargs["prompt"])
        self.assertIn("old picture", kwargs["prompt"])
        self.assertEqual(kwargs["image_urls"], [photo.path])
        self.assertIn("SUMMARY_SENTINEL", req.extra_user_content_parts[-1].text)

    async def test_disabled_images_preserve_audio_and_do_not_convert_image(self):
        self.configure(image_input_enabled=False)
        photo = Image()
        photo.convert_to_file_path = AsyncMock(
            side_effect=AssertionError("must not fetch")
        )
        event = Event(1, direct=True)
        event.message_obj.message.extend(
            [photo, Reply(sender_id="other", chain=[Record(path="audio.wav")])]
        )
        req = (await self.consume(event))[0]
        self.assertEqual(req.image_urls, [])
        self.assertEqual(req.audio_urls, ["audio.wav"])
        photo.convert_to_file_path.assert_not_awaited()
        self.assertIn("不能推断", req.prompt)

    async def test_current_quote_and_history_share_total_cap_and_scope(self):
        self.configure(max_context_images=2)
        other = Event(1, group="-20", direct=True)
        alien = self.image("other.png")
        other.message_obj.message.append(alien)
        await self.consume(other)
        photos = [self.image(f"{i}.png") for i in range(4)]
        current = Event(2)
        current.message_obj.message.extend(
            [photos[0], Reply(sender_id="x", chain=photos[1:])]
        )
        req = (await self.consume(current))[0]
        self.assertEqual(len(req.image_urls), 2)
        self.assertNotIn(alien.path, req.image_urls)
        self.assertEqual(
            req.image_urls, self.provider.text_chat.await_args.kwargs["image_urls"]
        )

    async def test_expired_or_deleted_images_are_not_sent_again(self):
        self.configure(image_retention_minutes=1)
        clock = [100.0]
        self.plugin._monotonic = lambda: clock[0]
        image = self.image()
        first = Event(1, direct=True)
        first.message_obj.message.append(image)
        await self.consume(first)
        clock[0] += 61
        req = (await self.consume(Event(2)))[0]
        self.assertEqual(req.image_urls, [])
        self.assertIn("不能推断", self.provider.text_chat.await_args.kwargs["prompt"])
        fresh = Event(3, direct=True)
        fresh.message_obj.message.append(image)
        await self.consume(fresh)
        Path(image.path).unlink()
        req = (await self.consume(Event(4)))[0]
        self.assertEqual(req.image_urls, [])

    async def test_unselected_stopped_and_stale_messages_do_not_capture(self):
        self.configure()
        for event in (Event(1, group="-99"), Event(2), Event(3)):
            if event.message_obj.message_id == "2":
                event.stop_event()
            if event.message_obj.message_id == "3":
                event.message_obj.timestamp = 1
            image = Image()
            image.convert_to_file_path = AsyncMock(
                side_effect=AssertionError("not allowed")
            )
            event.message_obj.message.append(image)
            self.assertEqual(await self.consume(event), [])
            image.convert_to_file_path.assert_not_awaited()

    async def test_capture_cancel_unload_releases_task_before_store_close(self):
        self.configure()
        entered = asyncio.Event()

        async def capture():
            entered.set()
            await asyncio.Event().wait()

        image = Image()
        image.convert_to_file_path = capture
        event = Event(1)
        event.message_obj.message.append(image)
        task = asyncio.create_task(self.consume(event))
        await entered.wait()
        await self.plugin.terminate()
        self.assertTrue(task.cancelled())
        self.assertFalse(self.plugin._handlers)
        self.assertIsNone(self.plugin.store)

    async def test_arrival_during_decision_does_not_mutate_frozen_snapshot(self):
        self.configure()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def decide(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                entered.set()
                await release.wait()
            return types.SimpleNamespace(
                completion_text='{"score":0.9,"reason":"相关"}'
            )

        self.provider.text_chat = decide
        first = asyncio.create_task(self.consume(Event(1, "first")))
        await entered.wait()
        event = Event(2, "NEXT_BATCH_SENTINEL")
        image = self.image()
        event.message_obj.message.append(image)
        second = asyncio.create_task(self.consume(event))
        await asyncio.sleep(0.01)
        release.set()
        results = await asyncio.gather(first, second)
        self.assertEqual([len(r) for r in results], [1, 1])
        self.assertEqual(calls[0]["image_urls"], [])
        self.assertNotIn("NEXT_BATCH_SENTINEL", calls[0]["prompt"])
        self.assertEqual(calls[1]["image_urls"], [image.path])

    async def test_deleted_during_decision_is_removed_before_formal_request(self):
        self.configure()
        photo = self.image()

        async def decide(**kwargs):
            self.assertEqual(kwargs["image_urls"], [photo.path])
            Path(photo.path).unlink()
            return types.SimpleNamespace(
                completion_text='{"score":0.9,"reason":"相关"}'
            )

        self.provider.text_chat = decide
        event = Event(1)
        event.message_obj.message.append(photo)
        request = (await self.consume(event))[0]
        self.assertEqual(request.image_urls, [])
        self.assertIn("不能推断", request.prompt)

    async def test_core_caption_cannot_bypass_disabled_images_through_quote(self):
        self.configure(image_input_enabled=False)
        self.context.config["provider_settings"] = {
            "default_image_caption_provider_id": "caption"
        }
        self.context.get_using_provider_async = AsyncMock(
            return_value=types.SimpleNamespace(provider_config={"modalities": ["text"]})
        )
        photo = Image()
        photo.convert_to_file_path = AsyncMock(side_effect=AssertionError("disabled"))
        event = Event(1, direct=True)
        event.message_obj.message.append(Reply(sender_id="x", chain=[photo]))
        self.assertEqual(await self.consume(event), [])
        self.assertTrue(event.call_llm)
        photo.convert_to_file_path.assert_not_awaited()
        self.context.conversation_manager.get_conversation.assert_not_awaited()
        self.assertFalse(self.plugin._rooms[event.unified_msg_origin].lock.locked())

    async def test_visual_main_provider_passes_caption_guard_and_override_is_honored(
        self,
    ):
        self.configure()
        self.context.config["provider_settings"] = {
            "default_image_caption_provider_id": "caption"
        }
        vision = types.SimpleNamespace(
            provider_config={"modalities": ["text", "image"]}
        )
        text_only = types.SimpleNamespace(provider_config={"modalities": ["text"]})
        self.context.get_using_provider_async = AsyncMock(return_value=vision)
        event = Event(1, direct=True)
        event.message_obj.message.append(self.image())
        self.assertEqual(len(await self.consume(event)), 1)
        self.context.get_provider_by_id = lambda ident: text_only
        override = Event(2, direct=True)
        override.message_obj.message.append(self.image("override.png"))
        override.set_extra("selected_provider", "nonvisual-override")
        self.assertEqual(await self.consume(override), [])
        self.context.get_using_provider_async.assert_awaited_once()

    async def test_core_empty_modalities_migration_compatibility(self):
        self.configure()
        self.context.config["provider_settings"] = {
            "default_image_caption_provider_id": "caption"
        }
        self.context.get_using_provider_async = AsyncMock(
            return_value=types.SimpleNamespace(provider_config={"modalities": []})
        )
        event = Event(1, direct=True)
        event.message_obj.message.append(self.image())
        self.assertEqual(len(await self.consume(event)), 1)

    async def test_caption_config_does_not_block_text_or_disabled_direct_image(self):
        self.configure(image_input_enabled=False)
        self.context.config["provider_settings"] = {
            "default_image_caption_provider_id": "caption"
        }
        self.context.get_using_provider_async = AsyncMock(
            return_value=types.SimpleNamespace(provider_config={"modalities": ["text"]})
        )
        self.assertEqual(len(await self.consume(Event(1, direct=True))), 1)
        direct_image = Event(2, direct=True)
        direct_image.message_obj.message.append(self.image())
        self.assertEqual(len(await self.consume(direct_image)), 1)
        self.context.get_using_provider_async.assert_not_awaited()
