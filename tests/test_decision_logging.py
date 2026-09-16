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
            decision_provider_id="",
            decision_max_chars=4000,
            decision_timeout=0.1,
            decision_log_reasoning=False,
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
            completion_text=verdict,
            reasoning_content=reasoning,
        )
        return await self.plugin._decide(
            Event(1),
            [
                {
                    "id": 1,
                    "role": "user",
                    "text": "private prompt input",
                    "event_at": 1000,
                    "sender_id": "user",
                    "sender_name": "user",
                }
            ],
        )

    def logged_payload(self):
        self.assertEqual(self.log.info.call_count, 2)
        return json.loads(self.log.info.call_args.args[1])

    def verdict_payload(self):
        payload = self.logged_payload()
        self.assertEqual(payload["truncated"], payload["reasoning_truncated"])
        self.assertEqual(payload["source_current_records"], 1)
        self.assertEqual(payload["image_count"], 0)
        return {k: payload[k] for k in ("decision", "reasoning", "truncated")}

    async def test_disabled_does_not_log_reasoning_or_verdict(self):
        self.assertTrue(await self.decide("provider reasoning"))
        self.log.info.assert_not_called()
        self.log.warning.assert_not_called()

    async def test_custom_rules_reach_system_prompt_with_fixed_chinese_protocol(self):
        self.cfg.decision_prompt = "优先参与轻松闲聊，不必等待问题。CUSTOM_STYLE"
        self.assertTrue(await self.decide())
        prompt = self.provider.text_chat.await_args.kwargs["system_prompt"]
        self.assertTrue(prompt.startswith(self.cfg.decision_prompt))
        self.assertTrue(prompt.endswith(runtime.DECISION_OUTPUT_PROTOCOL))
        self.assertIn("简体中文", prompt)
        self.assertIn("yes或no", prompt)
        self.assertIn("不调用工具", prompt)
        self.assertNotIn(
            "明确问题或有帮助时参与",
            self.provider.text_chat.await_args.kwargs["prompt"],
        )
        self.provider.text_chat.assert_awaited_once()

    async def test_blank_rules_use_default_without_extra_requests(self):
        for value in ("", "   "):
            self.cfg.decision_prompt = value
            self.provider.text_chat.reset_mock()
            self.assertTrue(await self.decide())
            self.assertEqual(
                self.provider.text_chat.await_args.kwargs["system_prompt"],
                runtime.DECISION_PROMPT,
            )
            self.provider.text_chat.assert_awaited_once()

    async def test_effective_group_rules_are_isolated_in_actual_provider_calls(self):
        self.plugin.settings = runtime.Settings.from_mapping(
            {
                "group_ids": ["-10", "-20"],
                "decision_prompt": "GLOBAL_RULE",
                "group_overrides": [
                    {"group_id": "-10", "decision_prompt": "GROUP_TEN_RULE"},
                    {"group_id": "-20", "decision_prompt": ""},
                ],
            }
        )
        self.plugin._settings = types.MethodType(
            runtime.AstrbotGroupChatLite._settings, self.plugin
        )
        for group, expected in (("-10", "GROUP_TEN_RULE"), ("-20", "GLOBAL_RULE")):
            self.provider.text_chat.return_value = types.SimpleNamespace(
                completion_text="yes"
            )
            self.assertTrue(await self.plugin._decide(Event(1, group=group), []))
            prompt = self.provider.text_chat.await_args.kwargs["system_prompt"]
            self.assertTrue(prompt.startswith(expected))
            self.assertTrue(prompt.endswith(runtime.DECISION_OUTPUT_PROTOCOL))

    async def test_enabled_logs_explicit_field_and_keeps_request_unchanged(self):
        self.cfg.decision_log_reasoning = True
        self.assertTrue(await self.decide("模型显式返回的简短推理"))
        self.assertEqual(
            self.verdict_payload(),
            {
                "decision": "yes",
                "reasoning": "模型显式返回的简短推理",
                "truncated": False,
            },
        )
        kwargs = self.provider.text_chat.await_args.kwargs
        self.assertEqual(kwargs["system_prompt"], runtime.DECISION_PROMPT)
        self.assertEqual(kwargs["contexts"], [])
        self.assertNotIn("reasoning", kwargs)
        self.assertNotIn("private prompt input", self.log.info.call_args.args[1])
        self.assertNotIn("provider reasoning", kwargs["prompt"])

    async def test_enabled_without_reasoning_marks_missing(self):
        self.cfg.decision_log_reasoning = True
        self.assertFalse(await self.decide(None, "no"))
        self.assertEqual(
            self.verdict_payload(),
            {
                "decision": "no",
                "reasoning": "未返回推理内容",
                "truncated": False,
            },
        )

    async def test_reasoning_is_capped_at_4000_characters(self):
        self.cfg.decision_log_reasoning = True
        self.assertTrue(await self.decide("长" * 4000 + "TAIL_SECRET"))
        payload = self.logged_payload()
        self.assertEqual(payload["reasoning"], "长" * 4000)
        self.assertTrue(payload["truncated"])
        self.assertNotIn("TAIL_SECRET", self.log.info.call_args.args[1])

    async def test_controls_and_unicode_line_separators_are_escaped(self):
        self.cfg.decision_log_reasoning = True
        reasoning = (
            '中文\n[INFO] forged\r\t\x00\x1b\x85\u2028\u2029\u202e\U000e0001"end"'
        )
        self.assertTrue(await self.decide(reasoning))
        self.assertEqual(self.logged_payload()["reasoning"], reasoning)
        encoded = self.log.info.call_args.args[1]
        self.assertFalse(
            any(unicodedata.category(c) in {"Cc", "Cf", "Zl", "Zp"} for c in encoded)
        )
        self.assertIn("中文", encoded)

    async def test_failure_and_timeout_keep_one_safe_warning(self):
        for enabled in (False, True):
            for exception in (
                RuntimeError("SECRET_PROMPT https://secret.invalid"),
                asyncio.TimeoutError(),
            ):
                with self.subTest(enabled=enabled, exception=type(exception).__name__):
                    self.log.reset_mock()
                    self.cfg.decision_log_reasoning = enabled
                    self.provider.text_chat.side_effect = exception
                    self.assertFalse(await self.plugin._decide(Event(1), []))
                    if enabled:
                        self.log.info.assert_called_once()
                        self.assertIn("判断输入统计", self.log.info.call_args.args[0])
                    else:
                        self.log.info.assert_not_called()
                    self.log.warning.assert_called_once()
                    template, error_type = self.log.warning.call_args.args
                    self.assertEqual(error_type, type(exception).__name__)
                    self.assertEqual("未返回推理内容" in template, enabled)
                    self.assertNotIn("SECRET_PROMPT", template)
                    self.assertNotIn("secret.invalid", template)
