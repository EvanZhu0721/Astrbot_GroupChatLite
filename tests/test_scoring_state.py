"""Pure metadata scoring state tests; no Telegram, network, or persistence."""

import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location(
    "lite_scoring_state", Path(__file__).resolve().parents[1] / "scoring_state.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ScoringStateTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.state = module.ScoringState(clock=lambda: self.now)

    def incoming(self, mid, sender="human", **kwargs):
        return self.state.observe_message("one", mid, sender, bot_id="9", **kwargs)

    def snap(self, mid, sender="human", **kwargs):
        return self.state.snapshot("one", mid, sender, **kwargs)

    def delivered(self, mid, sender="human", target="1"):
        self.state.record_delivery(
            "one", mid, target_message_id=target, target_sender_id=sender, bot_id="9"
        )

    def test_human_arrivals_deduplicate_and_bots_do_not_count(self):
        self.assertTrue(self.incoming("1"))
        self.assertFalse(self.incoming("1"))
        self.assertFalse(self.incoming("2", is_bot=True))
        self.delivered("3")
        self.assertEqual(self.snap("1").arrival_count, 1)

    def test_unknown_parent_or_other_bot_has_zero_hops(self):
        self.incoming("1", reply_to_message_id="unknown")
        self.incoming("2", reply_to_message_id="otherbot", reply_to_sender_id="8")
        self.assertEqual(self.snap("1").reply_hops, 0)
        self.assertEqual(self.snap("2").reply_hops, 0)

    def test_direct_raw_current_bot_author_does_not_need_cached_parent(self):
        self.incoming("1", reply_to_message_id="unknown", reply_to_sender_id="9")
        self.assertEqual(self.snap("1").reply_hops, 1)

    def test_username_fallback_matches_only_current_bot(self):
        self.state.observe_message(
            "one",
            "1",
            "human",
            reply_to_message_id="old",
            reply_to_sender_username="MYBOT",
            bot_id="@MyBot",
        )
        self.state.observe_message(
            "one",
            "2",
            "human",
            reply_to_message_id="old",
            reply_to_sender_username="OtherBot",
            bot_id="@MyBot",
        )
        self.assertEqual(self.snap("1").reply_hops, 1)
        self.assertEqual(self.snap("2").reply_hops, 0)

    def test_actual_reply_chain_counts_distance_to_current_bot(self):
        self.delivered("10")
        self.incoming("11", reply_to_message_id="10")
        self.incoming("12", reply_to_message_id="11")
        self.incoming("13", reply_to_message_id="12")
        self.assertEqual(
            [self.snap(str(i)).reply_hops for i in (11, 12, 13)], [1, 2, 3]
        )

    def test_no_reply_edge_is_invented_from_successful_target(self):
        self.incoming("1")
        self.delivered("10", target="1")
        self.incoming("2", reply_to_message_id="1")
        self.assertEqual(self.snap("2").reply_hops, 0)

    def test_graph_cycles_and_depth_limit_are_bounded(self):
        self.incoming("1", reply_to_message_id="2")
        self.incoming("2", reply_to_message_id="1")
        self.assertEqual(self.snap("1").reply_hops, 0)
        self.delivered("10")
        for mid in range(11, 20):
            self.incoming(str(mid), reply_to_message_id=str(mid - 1))
        self.assertEqual(self.snap("18").reply_hops, 8)
        self.assertEqual(self.snap("19").reply_hops, 0)

    def test_recent_target_requires_success_and_matching_human(self):
        self.incoming("1")
        self.assertFalse(self.snap("1").recent_target)
        self.delivered("2")
        self.assertTrue(self.snap("1").recent_target)
        self.assertFalse(self.snap("1", sender="other").recent_target)
        self.now += 120
        self.assertFalse(self.snap("1").recent_target)

    def test_successful_edit_refreshes_recency_without_arrival(self):
        self.incoming("1")
        self.delivered("2")
        self.now += 119
        self.delivered("2")
        self.now += 2
        self.assertTrue(self.snap("1").recent_target)
        self.assertEqual(self.snap("1").arrival_count, 0)

    def test_frequency_and_recency_window_are_configurable(self):
        self.incoming("1")
        self.delivered("2")
        self.now += 90
        self.assertEqual(self.snap("1").arrival_count, 0)
        self.assertEqual(self.snap("1", frequency_seconds=100).arrival_count, 1)
        self.assertFalse(self.snap("1", recent_seconds=30).recent_target)
        self.assertTrue(self.snap("1", recent_seconds=100).recent_target)

    def test_umo_and_topic_isolation(self):
        self.delivered("10")
        self.state.observe_message(
            "two#7", "11", "human", reply_to_message_id="10", bot_id="9"
        )
        state = self.state.snapshot("two#7", "11", "human")
        self.assertEqual(state.reply_hops, 0)
        self.assertFalse(state.recent_target)
        self.assertEqual(state.arrival_count, 1)

    def test_ttl_expiry_drops_old_graph_and_arrivals(self):
        self.delivered("10")
        self.now += 3600
        self.incoming("11", reply_to_message_id="10")
        state = self.snap("11")
        self.assertEqual(state.reply_hops, 0)
        self.assertFalse(state.recent_target)
        self.assertEqual(state.arrival_count, 1)

    def test_frozen_snapshot_and_pending_are_not_live_views(self):
        self.incoming("1")
        snapshot = self.snap("1", pending=3)
        self.incoming("2")
        self.assertEqual(snapshot.arrival_count, 1)
        self.assertEqual(snapshot.pending, 3)
        with self.assertRaises(AttributeError):
            snapshot.pending = 2

    def test_capacity_and_clear(self):
        state = module.ScoringState(
            clock=lambda: self.now, max_umos=2, max_messages=2, max_arrivals=3
        )
        for umo in ("one", "two", "three"):
            for mid in range(5):
                state.observe_message(umo, str(mid + 1), "human")
        self.assertEqual(len(state._rooms), 2)
        self.assertTrue(
            all(
                len(room.nodes) == 2 and len(room.arrivals) == 3
                for room in state._rooms.values()
            )
        )
        self.assertEqual(state.snapshot("one", "1", "human").arrival_count, 0)
        state.clear()
        self.assertEqual(len(state._rooms), 0)

    def test_arrival_capacity_supports_maximum_configured_high_threshold(self):
        for mid in range(1100):
            self.incoming(str(mid + 1))
        self.assertEqual(self.snap("1100").arrival_count, 1100)

    def test_invalid_metadata_never_creates_state(self):
        for umo, mid, sender in (
            ("", "1", "x"),
            ("x" * 1025, "1", "x"),
            ("one", None, "x"),
            ("one", True, "x"),
            ("one", "1", ""),
        ):
            self.assertFalse(self.state.observe_message(umo, mid, sender))
        self.assertFalse(self.state._rooms)


if __name__ == "__main__":
    unittest.main()
