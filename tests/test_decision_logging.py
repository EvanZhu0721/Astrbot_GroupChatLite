"""Optional decision reasoning logs; all providers are local test doubles."""

import asyncio
import json
import socket
import types
import unicodedata
import unittest
from unittest.mock import AsyncMock, Mock, patch

from test_runtime import Context, Event, runtime


class DecisionLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = runtime.AstrbotGroupChatLite(Context(), {"group_ids": ["-10"]})
        self.cfg = types.SimpleNamespace(
            decision_provider_id="", decision_max_chars=4000,
            decision_timeout=0.1, decision_log_reasoning=False,
        )
        self.plugin._settings = lambda *args: self.cfg
        self.provider = types.SimpleNamespace(text_chat=AsyncMock())
        self.plugin._provider = AsyncMock(return_value=self.provider)
        self.log = Mock()
        self.logger_patch = patch.object(runtime, "logger", self.log)
        self.logger_patch.start()
        self.network_patch = patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network forbidden")
        )
        self.network_patch.start()

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.network_patch.stop()
        self.logger_patch.stop()

    async def decide(self, reasoning=None, verdict="yes"):
        self.provider.text_chat.return_value = types.SimpleNamespace(
            completion_text=verdict, reasoning_content=reasoning,
        )
        return await self.plugin._decide(Event(1), [{
            "id": 1, "role": "user", "text": "private prompt input",
            "event_at": 1000, "sender_id": "user", "sender_name": "user",
        }])

    def logged_payload(self):
        self.log.info.assert_called_once()
        return json.loads(self.log.info.call_args.args[1])

    async def test_disabled_does_not_log_reasoning_or_verdict(self):
        self.assertTrue(await self.decide("provider reasoning"))
        self.log.info.assert_not_called()
        self.log.warning.assert_not_called()

    async def test_enabled_logs_explicit_field_and_keeps_request_unchanged(self):
        self.cfg.decision_log_reasoning = True
        self.assertTrue(await self.decide("模型显式返回的简短推理"))
        self.assertEqual(self.logged_payload(), {
            "decision": "yes", "reasoning": "模型显式返回的简短推理", "truncated": False,
        })
        kwargs = self.provider.text_chat.await_args.kwargs
        self.assertEqual(kwargs["system_prompt"], runtime.DECISION_PROMPT)
        self.assertEqual(kwargs["contexts"], [])
        self.assertNotIn("reasoning", kwargs)
        self.assertNotIn("private prompt input", self.log.info.call_args.args[1])
        self.assertNotIn("provider reasoning", kwargs["prompt"])

    async def test_enabled_without_reasoning_marks_missing(self):
        self.cfg.decision_log_reasoning = True
        self.assertFalse(await self.decide(None, "no"))
        self.assertEqual(self.logged_payload(), {
            "decision": "no", "reasoning": "未返回推理内容", "truncated": False,
        })

    async def test_reasoning_is_capped_at_4000_characters(self):
        self.cfg.decision_log_reasoning = True
        self.assertTrue(await self.decide("长" * 4000 + "TAIL_SECRET"))
        payload = self.logged_payload()
        self.assertEqual(payload["reasoning"], "长" * 4000)
        self.assertTrue(payload["truncated"])
        self.assertNotIn("TAIL_SECRET", self.log.info.call_args.args[1])

    async def test_controls_and_unicode_line_separators_are_escaped(self):
        self.cfg.decision_log_reasoning = True
        reasoning = '中文\n[INFO] forged\r\t\x00\x1b\x85\u2028\u2029\u202e\U000e0001"end"'
        self.assertTrue(await self.decide(reasoning))
        self.assertEqual(self.logged_payload()["reasoning"], reasoning)
        encoded = self.log.info.call_args.args[1]
        self.assertFalse(any(unicodedata.category(c) in {"Cc", "Cf", "Zl", "Zp"} for c in encoded))
        self.assertIn("中文", encoded)

    async def test_failure_and_timeout_keep_one_safe_warning(self):
        for enabled in (False, True):
            for exception in (RuntimeError("SECRET_PROMPT https://secret.invalid"), asyncio.TimeoutError()):
                with self.subTest(enabled=enabled, exception=type(exception).__name__):
                    self.log.reset_mock()
                    self.cfg.decision_log_reasoning = enabled
                    self.provider.text_chat.side_effect = exception
                    self.assertFalse(await self.plugin._decide(Event(1), []))
                    self.log.info.assert_not_called()
                    self.log.warning.assert_called_once()
                    template, error_type = self.log.warning.call_args.args
                    self.assertEqual(error_type, type(exception).__name__)
                    self.assertEqual("未返回推理内容" in template, enabled)
                    self.assertNotIn("SECRET_PROMPT", template)
                    self.assertNotIn("secret.invalid", template)
