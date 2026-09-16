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
    def test_decision_data_header_does_not_override_custom_participation_rules(self):
        output = builder.render_decision(
            [message(7, "ordinary conversation")], current_message_id=7
        )
        self.assertIn("判断规则", output)
        self.assertIn("历史已答问题不是新请求", output)
        self.assertNotIn("明确问题或有帮助时参与", output)
        self.assertNotIn("只输出 yes 或 no", output)

    def test_decision_and_reply_share_previous_window_sources(self):
        records = [message(7, "current unique question")]
        previous = [message(6, "previous bot offered help", "assistant")]
        kwargs = {
            "previous_messages": previous,
            "previous_summary": {"window_id": 2, "text": "summary unique topic"},
        }
        for output in (
            builder.render_decision(records, **kwargs),
            builder.render_context(records, **kwargs),
        ):
            self.assertIn("previous bot offered help", output)
            self.assertIn("summary unique topic", output)
            self.assertIn("current unique question", output)
            self.assertIn('"role":"assistant"', output)
        self.assertIn("历史已答问题不是新请求", builder.render_decision(records))

    def test_multiple_images_have_explicit_source_ids_even_when_current_excluded(self):
        sources = [
            {"message_id": 7, "image_index": 1},
            {"message_id": 7, "image_index": 2},
            {"message_id": 6, "image_index": 3},
        ]
        kwargs = {"image_sources": sources, "max_chars": 4000}
        for output in (
            builder.render_decision([message(7, "image text")], **kwargs),
            builder.render_context(
                [message(7, "image text")], current_message_id=7, **kwargs
            ),
        ):
            parsed = [
                json.loads(line) for line in output.splitlines() if line.startswith("{")
            ]
            mapped = [line for line in parsed if "image_index" in line]
            self.assertEqual(
                mapped,
                [
                    {"image_index": x["image_index"], "message_id": x["message_id"]}
                    for x in sources
                ],
            )
            self.assertIn("未实际提供的图片不可推断", output)

    def test_multimodal_source_sections_respect_escaped_budget(self):
        records = [message(i, '\n"\\' * 1000) for i in range(20)]
        kwargs = {
            "previous_messages": records[:4],
            "previous_summary": {"text": '\n"\\' * 5000, "window_id": '"\\' * 100},
            "image_sources": [
                {"message_id": '"\\' * 100, "image_index": i + 1} for i in range(8)
            ],
        }
        for size in (512, 1024, 4000):
            for render in (builder.render_context, builder.render_decision):
                output = render(records, max_chars=size, **kwargs)
                self.assertLessEqual(len(output), size)
                for line in output.splitlines():
                    if line.startswith("{"):
                        json.loads(line)

    def test_summary_marks_image_placeholders_as_unseen(self):
        output = builder.render_summary(
            [message(1, "[非文本消息：Image]")], window_id=1
        )
        self.assertIn("本次无图片输入", output)
        self.assertIn("[非文本消息：Image]", output)

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

    def test_decision_focus_keeps_body_when_not_last_record(self):
        output = builder.render_decision(
            [message(7, "trigger body"), message(8, "later record")],
            current_message_id=7,
        )
        self.assertIn('"current_input_message_id":7', output)
        self.assertIn("trigger body", output)
        self.assertIn("later record", output)
        self.assertLessEqual(
            len(
                builder.render_decision(
                    [], current_message_id="\x00" * 1000, max_chars=512
                )
            ),
            512,
        )

    def test_current_image_mapping_keeps_live_input_identity_within_budget(self):
        current = message(7, "live unique text" + '\n"' * 500)
        output = builder.render_context(
            [current],
            current_message_id=7,
            image_sources=[{"message_id": 7, "image_index": 1}],
            max_chars=512,
        )
        self.assertLessEqual(len(output), 512)
        self.assertNotIn("live unique text", output)
        self.assertIn('"current_input_message_id":7', output)
        self.assertIn('"image_index":1,"message_id":7', output)

    def test_newest_records_retained_when_input_exceeds_store_cap(self):
        records = [message(i, f"raw-{i}") for i in range(205)]
        output = builder.render_summary(records, window_id=1, max_chars=128000)
        self.assertIn("raw-204", output)
        self.assertNotIn('"text":"raw-0"', output)
        self.assertIn("范围不完整", output)


if __name__ == "__main__":
    unittest.main()
