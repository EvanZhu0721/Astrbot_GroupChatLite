"""Offline contracts extracted from an optional local AstrBot source checkout.

No application, provider or Telegram client is imported or started.
Set ASTRBOT_ROOT to run these compatibility checks against your target version.
"""

import ast
import copy
import dataclasses
import os
from pathlib import Path
import types
import unittest


ROOT = Path(os.environ["ASTRBOT_ROOT"]) if os.environ.get("ASTRBOT_ROOT") else None


def source(relative):
    return (ROOT / relative).read_text(encoding="utf-8-sig")


def extract(relative, name):
    return next(
        node
        for node in ast.walk(ast.parse(source(relative)))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def load_node(node, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<native-contract-extract>", "exec"),
        namespace,
    )
    return namespace[node.name]


@unittest.skipUnless(
    ROOT and ROOT.is_dir(), "Set ASTRBOT_ROOT for native source contracts"
)
class NativeContracts(unittest.IsolatedAsyncioTestCase):
    def test_conversation_copy_retains_persona_without_mutating_old_history(self):
        cls = load_node(
            extract("astrbot/core/db/po.py", "Conversation"),
            {"dataclass": dataclasses.dataclass},
        )
        original = cls(
            platform_id="test",
            user_id="group",
            cid="conversation",
            history='[{"role":"user","content":"old"}]',
            persona_id="persona",
        )
        scoped = copy.copy(original)
        scoped.history = "[]"
        self.assertEqual(scoped.persona_id, "persona")
        self.assertEqual(scoped.cid, original.cid)
        self.assertNotEqual(scoped.history, original.history)

    async def test_detached_request_skips_native_conversation_write(self):
        node = extract(
            "astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py",
            "_save_to_history",
        )
        function = load_node(node, {})
        # The actual native method must return before touching event/runner/db.
        import inspect

        kwargs = {
            name: None
            for name in inspect.signature(function).parameters
            if name != "self"
        }
        kwargs["req"] = types.SimpleNamespace(conversation=None)
        await function(object(), **kwargs)

    def test_build_before_hook_and_reset_after_hook(self):
        text = source(
            "astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py"
        )
        hook = text.index("EventType.OnLLMRequestEvent")
        self.assertLess(
            text.index("build_main_agent(", text.index("async def process")), hook
        )
        self.assertGreater(text.index("await reset_coro", hook), hook)
        main = source("astrbot/core/astr_main_agent.py")
        self.assertIn("conversation_persona_id=req.conversation.persona_id", main)
        self.assertIn("req.contexts = json.loads(req.conversation.history)", main)

    def test_stream_response_hook_is_terminal_not_each_tool_round(self):
        done = extract("astrbot/core/astr_agent_hooks.py", "on_agent_done")
        self.assertIn("OnLLMResponseEvent", ast.unparse(done))
        tool = extract("astrbot/core/astr_agent_hooks.py", "on_tool_end")
        self.assertNotIn("OnLLMResponseEvent", ast.unparse(tool))
        runner = source("astrbot/core/agent/runners/tool_loop_agent_runner.py")
        aborted = runner[runner.index("async def _finalize_aborted_step") :]
        self.assertLess(
            aborted.index("self._aborted = True"),
            aborted.index("self.agent_hooks.on_agent_done"),
        )

    def test_explicit_request_and_default_fallback_guard_are_distinct(self):
        text = source("astrbot/core/pipeline/process_stage/stage.py")
        self.assertIn("isinstance(resp, ProviderRequest)", text)
        self.assertIn("and not event.call_llm", text)
        self.assertIn("not event._has_send_oper", text)
        self.assertIn("event.is_at_or_wake_command", text)


if __name__ == "__main__":
    unittest.main()
