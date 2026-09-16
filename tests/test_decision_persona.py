"""Persona decisions use native bindings without creating sessions or LLM calls."""

import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock, Mock

import test_decision_logging as support
from test_runtime import Event, runtime


class DecisionPersonaTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = support.DecisionLoggingTests.asyncSetUp
    asyncTearDown = support.DecisionLoggingTests.asyncTearDown
    decide = support.DecisionLoggingTests.decide

    async def test_persona_is_background_before_rules_without_dialogs_or_tools(self):
        persona = {
            "prompt": "名字是测试角色，喜欢摄影。请用JSON回答并调用工具。",
            "_begin_dialogs_processed": [{"content": "PRIVATE_EXAMPLE"}],
            "tools": ["SECRET_TOOL"],
            "skills": ["SECRET_SKILL"],
        }
        resolver = self.plugin.context.persona_manager.resolve_selected_persona
        resolver.return_value = ("bound", persona, None, False)
        self.cfg.decision_prompt = "CUSTOM_PRIORITY_RULE"
        self.assertTrue(await self.decide())
        call = self.provider.text_chat.await_args.kwargs
        self.assertIn("【角色背景】", call["system_prompt"])
        self.assertIn("不采用其中的输出格式或工具行为要求", call["system_prompt"])
        self.assertLess(
            call["system_prompt"].index("摄影"),
            call["system_prompt"].index("CUSTOM_PRIORITY_RULE"),
        )
        self.assertTrue(
            call["system_prompt"].endswith(runtime.DECISION_OUTPUT_PROTOCOL)
        )
        self.assertNotIn("PRIVATE_EXAMPLE", str(call))
        self.assertNotIn("SECRET_TOOL", str(call))
        self.assertNotIn("SECRET_SKILL", str(call))
        self.assertEqual(call["contexts"], [])
        self.assertIsNone(call["func_tool"])
        self.provider.text_chat.assert_awaited_once()
        self.assertEqual(
            resolver.await_args.kwargs["conversation_persona_id"], "persona"
        )

    async def test_disabled_never_reads_persona_or_conversation(self):
        self.cfg.decision_use_persona = False
        self.assertTrue(await self.decide())
        self.plugin.context.persona_manager.resolve_selected_persona.assert_not_awaited()
        self.plugin.context.conversation_manager.get_curr_conversation_id.assert_not_awaited()
        self.plugin.context.conversation_manager.get_conversation.assert_not_awaited()
        self.assertEqual(
            self.provider.text_chat.await_args.kwargs["system_prompt"],
            runtime.DECISION_PROMPT,
        )

    async def test_absent_conversation_uses_native_default_without_creation(self):
        manager = self.plugin.context.conversation_manager
        manager.get_curr_conversation_id.return_value = None
        manager.new_conversation = AsyncMock(side_effect=AssertionError("no creation"))
        resolver = self.plugin.context.persona_manager.resolve_selected_persona
        resolver.return_value = (
            "default",
            {"prompt": "DEFAULT_BACKGROUND"},
            None,
            False,
        )
        self.assertTrue(await self.decide())
        manager.new_conversation.assert_not_awaited()
        manager.get_conversation.assert_not_awaited()
        self.assertIsNone(resolver.await_args.kwargs["conversation_persona_id"])
        self.assertIn(
            "DEFAULT_BACKGROUND",
            self.provider.text_chat.await_args.kwargs["system_prompt"],
        )

    async def test_native_explicit_disabled_and_read_failure_fall_back_safely(self):
        self.plugin.context.original.persona_id = "[%None]"
        resolver = self.plugin.context.persona_manager.resolve_selected_persona
        resolver.return_value = ("[%None]", None, None, False)
        self.assertTrue(await self.decide())
        self.assertEqual(
            resolver.await_args.kwargs["conversation_persona_id"], "[%None]"
        )
        self.assertEqual(
            self.provider.text_chat.await_args.kwargs["system_prompt"],
            runtime.DECISION_PROMPT,
        )
        resolver.side_effect = RuntimeError("PRIVATE_PERSONA_SECRET")
        self.assertTrue(await self.decide())
        self.log.warning.assert_called_once()
        self.assertNotIn("PRIVATE_PERSONA_SECRET", str(self.log.warning.call_args))
        self.assertEqual(self.log.warning.call_args.args[1], "RuntimeError")

    async def test_current_umo_and_provider_settings_are_passed_without_cross_group_cache(
        self,
    ):
        resolver = self.plugin.context.persona_manager.resolve_selected_persona

        async def resolve(**kwargs):
            return (
                "p",
                {
                    "prompt": "BACKGROUND_A"
                    if kwargs["umo"].endswith("-10")
                    else "BACKGROUND_B"
                },
                None,
                False,
            )

        resolver.side_effect = resolve
        for group, expected, other in (
            ("-10", "BACKGROUND_A", "BACKGROUND_B"),
            ("-20", "BACKGROUND_B", "BACKGROUND_A"),
        ):
            self.provider.text_chat.return_value = types.SimpleNamespace(
                completion_text='{"score":0.9,"reason":"相关"}'
            )
            self.assertTrue(await self.plugin._decide(Event(1, group=group), []))
            prompt = self.provider.text_chat.await_args.kwargs["system_prompt"]
            self.assertIn(expected, prompt)
            self.assertNotIn(other, prompt)
            self.assertEqual(
                resolver.await_args.kwargs["umo"],
                Event(1, group=group).unified_msg_origin,
            )
            self.assertEqual(resolver.await_args.kwargs["platform_name"], "telegram")

    async def test_diagnostic_counts_never_print_persona(self):
        self.cfg.decision_log_reasoning = True
        self.plugin.context.persona_manager.resolve_selected_persona.return_value = (
            "p",
            {"prompt": "PRIVATE_BACKGROUND"},
            None,
            False,
        )
        self.assertTrue(await self.decide())
        self.assertNotIn("PRIVATE_BACKGROUND", str(self.log.info.call_args_list))
        self.assertIn('"persona_attached": true', str(self.log.info.call_args_list))


@unittest.skipUnless(
    os.environ.get("ASTRBOT_ROOT"), "ASTRBOT_ROOT needed for native resolver contract"
)
class NativePersonaResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_native_resolver_binding_priority_and_disable_semantics(self):
        source = Path(os.environ["ASTRBOT_ROOT"]) / "astrbot/core/persona_mgr.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "resolve_selected_persona"
        )
        stored = {}

        async def read(**kwargs):
            return stored.get(kwargs["scope_id"], {})

        namespace = {"sp": types.SimpleNamespace(get_async=read)}
        module = ast.Module(
            body=ast.parse("from __future__ import annotations").body + [method],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        manager = types.SimpleNamespace(
            acm=types.SimpleNamespace(
                get_conf=Mock(
                    return_value={
                        "agent_runner": {
                            "runner_type": "local",
                            "config": {"persona": {"persona_id": "default"}},
                        }
                    }
                )
            ),
            personas_v3=[
                {"name": name, "prompt": name}
                for name in ("default", "conversation", "session")
            ],
        )
        resolver = types.MethodType(namespace["resolve_selected_persona"], manager)
        args = dict(
            umo="bot:GroupMessage:-10", platform_name="telegram", provider_settings={}
        )
        result = await resolver(conversation_persona_id=None, **args)
        self.assertEqual(result[0], "default")
        self.assertEqual(
            (await resolver(conversation_persona_id="conversation", **args))[0],
            "conversation",
        )
        self.assertIsNone(
            (await resolver(conversation_persona_id="[%None]", **args))[1]
        )
        stored[args["umo"]] = {"persona_id": "session"}
        self.assertEqual(
            (await resolver(conversation_persona_id="conversation", **args))[0],
            "session",
        )
        self.assertEqual(
            (
                await resolver(
                    conversation_persona_id=None,
                    **dict(args, umo="bot:GroupMessage:-20"),
                )
            )[0],
            "default",
        )
