"""Offline score parsing, additive rules and timezone boundaries."""

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scoring = load("lite_scoring_tests_module", ROOT / "scoring.py")
config = load("lite_scoring_config", ROOT / "config.py")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config.Settings()
        self.day = datetime(2026, 1, 1, 4, tzinfo=timezone.utc).timestamp()

    def score(self, base=0.4, **kwargs):
        values = dict(
            reply_hops=None,
            recent_reply_target=False,
            recent_message_count=3,
            pending_count=2,
            now=self.day,
            cfg=self.cfg,
        )
        values.update(kwargs)
        return scoring.score_decision(base, **values)

    def test_parse_exact_json(self):
        self.assertEqual(
            scoring.parse_model_score('{"score":0.7,"reason":"可承接"}'),
            {"score": 0.7, "reason": "可承接"},
        )
        self.assertEqual(
            scoring.parse_model_score(' {"score":0} '), {"score": 0.0, "reason": ""}
        )

    def test_parse_rejects_nonfinite_outside_and_wrong_protocol(self):
        for text in (
            "yes",
            '```json\n{"score":1}\n```',
            '{"score":NaN}',
            '{"score":Infinity}',
            '{"score":true}',
            '{"score":"0.8"}',
            '{"score":1.1}',
            '{"score":-0.1}',
            '{"score":0.3,"score":1}',
            '{"score":1,"decision":"yes"}',
            '{"reason":"missing score"}',
            "[1]",
            '{"score":1,"reason":null}',
        ):
            with self.subTest(text=text):
                self.assertIsNone(scoring.parse_model_score(text))
        self.assertIsNone(
            scoring.parse_model_score('{"score":1,"reason":"' + "x" * 201 + '"}')
        )

    def test_direct_and_indirect_quote_decay(self):
        for hops, expected in ((1, 1.0), (2, 0.5), (3, 0.25), (8, 1 / 128)):
            self.assertEqual(self.score(reply_hops=hops)["bonuses"]["quote"], expected)

    def test_zero_decay_preserves_direct_but_disables_indirect(self):
        cfg = replace(self.cfg, score_quote_decay=0)
        self.assertEqual(self.score(reply_hops=1, cfg=cfg)["bonuses"]["quote"], 1.0)
        self.assertEqual(self.score(reply_hops=2, cfg=cfg)["bonuses"]["quote"], 0.0)

    def test_recent_target_only_when_caller_confirms(self):
        self.assertEqual(self.score(recent_reply_target=True)["bonuses"]["recent"], 0.2)
        self.assertEqual(self.score()["bonuses"]["recent"], 0.0)

    def test_high_frequency_and_singleton_both_apply(self):
        result = self.score(0.6, recent_message_count=6, pending_count=1)
        self.assertEqual(result["bonuses"]["high_frequency"], -0.1)
        self.assertEqual(result["bonuses"]["low_or_singleton"], 0.2)
        self.assertAlmostEqual(result["total"], 0.7)
        self.assertTrue(result["should_reply"])

    def test_low_or_singleton_does_not_double_count(self):
        self.assertEqual(
            self.score(recent_message_count=1, pending_count=1)["bonuses"][
                "low_or_singleton"
            ],
            0.2,
        )
        self.assertEqual(
            self.score(recent_message_count=2, pending_count=2)["bonuses"][
                "low_or_singleton"
            ],
            0.0,
        )

    def test_final_clamp_happens_after_every_term(self):
        result = self.score(0.9, reply_hops=1, recent_message_count=6)
        self.assertEqual(result["total"], 1.0)
        self.assertEqual(self.score(0, recent_message_count=6)["total"], 0.0)

    def test_night_zone_and_exclusive_end(self):
        for hour, minute, expected in (
            (15, 59, 0.0),
            (16, 0, 0.1),
            (22, 59, 0.1),
            (23, 0, 0.0),
        ):
            stamp = datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc).timestamp()
            self.assertEqual(self.score(now=stamp)["bonuses"]["night"], expected)

    def test_cross_midnight_and_equal_times(self):
        cfg = replace(self.cfg, score_night_start="23:00", score_night_end="02:00")
        for utc_hour, expected in ((14, 0.0), (15, 0.1), (17, 0.1), (18, 0.0)):
            stamp = datetime(2026, 1, 1, utc_hour, tzinfo=timezone.utc).timestamp()
            self.assertEqual(
                self.score(now=stamp, cfg=cfg)["bonuses"]["night"], expected
            )
        cfg = replace(cfg, score_night_end="23:00")
        self.assertEqual(self.score(cfg=cfg)["bonuses"]["night"], 0.0)

    def test_all_bonus_weights_zero(self):
        cfg = replace(
            self.cfg,
            score_quote_bonus=0,
            score_recent_bonus=0,
            score_high_penalty=0,
            score_low_bonus=0,
            score_night_bonus=0,
        )
        result = self.score(
            0.7,
            reply_hops=1,
            recent_reply_target=True,
            recent_message_count=9,
            pending_count=1,
            cfg=cfg,
        )
        self.assertTrue(all(value == 0 for value in result["bonuses"].values()))
        self.assertTrue(result["should_reply"])

    def test_invalid_runtime_inputs_raise(self):
        for base in (float("nan"), float("inf"), True, -1, 2):
            with self.assertRaises(ValueError):
                self.score(base)
        for overrides in (
            {"reply_hops": 0},
            {"reply_hops": 9},
            {"reply_hops": True},
            {"recent_reply_target": 1},
            {"recent_message_count": -1},
            {"pending_count": 1.1},
            {"now": float("nan")},
        ):
            with self.assertRaises(ValueError):
                self.score(**overrides)

    def test_dst_transition_uses_configured_zone(self):
        cfg = replace(
            self.cfg,
            score_timezone="America/New_York",
            score_night_start="01:00",
            score_night_end="03:00",
        )
        cases = (
            (datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc), 0.1),
            (datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc), 0.0),
            (datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc), 0.1),
            (datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc), 0.1),
        )
        for moment, expected in cases:
            self.assertEqual(
                self.score(now=moment.timestamp(), cfg=cfg)["bonuses"]["night"],
                expected,
            )

    def test_threshold_endpoints(self):
        self.assertTrue(
            self.score(0, cfg=replace(self.cfg, score_threshold=0))["should_reply"]
        )
        self.assertFalse(
            self.score(0.99, cfg=replace(self.cfg, score_threshold=1))["should_reply"]
        )
        self.assertTrue(
            self.score(1, cfg=replace(self.cfg, score_threshold=1))["should_reply"]
        )


if __name__ == "__main__":
    unittest.main()
