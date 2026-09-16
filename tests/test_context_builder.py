"""Behavior checks for context budgets, provenance, and window boundaries."""

import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "gcl_context", Path(__file__).resolve().parents[1] / "context_builder.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def message(mid, text, role="user"):
    return {
        "id": mid,
        "role": role,
        "sender_name": "user",
        "event_at": 100 + mid,
        "text": text,
    }


class ContextTests(unittest.TestCase):
    def test_current_input_not_duplicated_and_previous_sources_labeled(self):
        result = builder.render_context(
            [message(1, "old topic"), message(2, "current unique input")],
            current_message_id=2,
            previous_messages=[message(0, "previous exchange", "assistant")],
            previous_summary={"text": "previous summary", "window_id": 8},
        )
        self.assertNotIn("current unique input", result)
        self.assertIn("old topic", result)
        self.assertIn("窗口=8", result)
        self.assertIn('"role":"assistant"', result)

    def test_hard_budget_and_latest_records_win(self):
        records = [message(i, f"message-{i}:" + '长文\n"' * 1000) for i in range(20)]
        for size in (512, 1024, 16000):
            result = builder.render_context(records, max_chars=size)
            self.assertLessEqual(len(result), size)
            self.assertIn("message-19:", result)
            self.assertNotIn("message-0:", result)
            for line in result.splitlines():
                if line.startswith("{"):
                    json.loads(line)

    def test_summary_requires_partial_notice(self):
        result = builder.render_summary(
            [message(1, "last record")], window_id=5, total_messages=100
        )
        self.assertIn("范围不完整", result)
        self.assertIn("窗口编号：5", result)
        self.assertIn("last record", result)

    def test_all_renderers_bounded_with_escaped_characters(self):
        records = [message(i, '\n"\\' * 10000) for i in range(3)]
        for render in (builder.render_decision, builder.render_history):
            self.assertLessEqual(len(render(records, max_chars=512)), 512)
        self.assertLessEqual(
            len(builder.render_summary(records, window_id=1, max_chars=512)), 512
        )

    def test_unknown_time_never_becomes_now(self):
        result = builder.render_history([{"id": 1, "text": "unknown date"}])
        self.assertIn('"at":"unknown"', result)

    def test_same_text_from_two_people_is_preserved(self):
        one, two = message(1, "same"), message(2, "same")
        two["sender_name"] = "another"
        result = builder.render_context([one, two])
        self.assertIn('"id":1', result)
        self.assertIn('"id":2', result)

    def test_tiny_and_invalid_budgets(self):
        renders = (
            builder.render_context,
            builder.render_decision,
            builder.render_history,
            lambda rows, **kw: builder.render_summary(rows, window_id=1, **kw),
        )
        for render in renders:
            for budget in (0, 1, 200, 511):
                self.assertLessEqual(
                    len(render([message(1, "text")], max_chars=budget)), budget
                )
            for budget in (float("nan"), float("inf"), None, "invalid"):
                self.assertLessEqual(
                    len(render([message(1, "text")], max_chars=budget)), 512
                )

    def test_untrusted_identifiers_cannot_break_budget(self):
        huge = '\n"' * 10000
        records = [message(1, "raw")]
        output = builder.render_context(
            records,
            previous_summary={"text": "summary", "window_id": huge},
            max_chars=512,
        )
        self.assertLessEqual(len(output), 512)
        output = builder.render_summary(records, window_id=huge, max_chars=512)
        self.assertLessEqual(len(output), 512)
        records[0]["id"] = {"unexpected": huge}
        self.assertLessEqual(len(builder.render_history(records, max_chars=512)), 512)

    def test_summary_never_resummarizes_summary_records(self):
        result = builder.render_summary(
            [message(1, "raw user"), message(2, "derived summary unique", "summary")],
            window_id=1,
        )
        self.assertIn("raw user", result)
        self.assertNotIn("derived summary unique", result)
        self.assertIn("范围不完整", result)

    def test_current_input_is_also_excluded_from_bridge(self):
        current = message(99, "live input unique")
        output = builder.render_context(
            [current], previous_messages=[current], current_message_id=99
        )
        self.assertNotIn("live input unique", output)

    def test_newest_records_retained_when_input_exceeds_store_cap(self):
        records = [message(i, f"raw-{i}") for i in range(205)]
        output = builder.render_summary(records, window_id=1, max_chars=128000)
        self.assertIn("raw-204", output)
        self.assertNotIn('"text":"raw-0"', output)
        self.assertIn("范围不完整", output)


if __name__ == "__main__":
    unittest.main()
