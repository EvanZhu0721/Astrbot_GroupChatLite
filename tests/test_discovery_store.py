"""Metadata-only discovery persistence; no runtime or network dependencies."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "gcl_discovery_store", Path(__file__).resolve().parents[1] / "store.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Store = module.Store


class DiscoveryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "history.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def observe(self, umo="bot:GroupMessage:-100#2", **kwargs):
        return self.store.observe_group(
            umo,
            kwargs.get("platform_id", "bot"),
            kwargs.get("group_id", "-100#2"),
            kwargs.get("display_name", "示例群 / 话题二"),
            kwargs.get("observed_at", 100),
        )

    def test_discovery_survives_database_reopen_with_explicit_fields(self):
        expected = self.observe()
        self.store.close()
        self.store = Store(self.path)
        actual = self.store.get_discovered_group(expected["umo"])
        self.assertEqual(actual, expected)
        self.assertEqual(
            set(actual),
            {
                "umo",
                "platform_id",
                "group_id",
                "display_name",
                "first_seen",
                "last_seen",
            },
        )
        self.assertIsNone(self.store.get_discovered_group("missing:GroupMessage:-1"))

    def test_duplicate_keeps_first_seen_and_monotonic_last_seen(self):
        original = self.observe(observed_at=100)
        renamed = self.observe(observed_at=200, display_name="新群名")
        replayed = self.observe(observed_at=50, display_name="")
        self.assertEqual(renamed["display_name"], "新群名")
        self.assertEqual(replayed["display_name"], "新群名")
        self.assertEqual(replayed["first_seen"], original["first_seen"])
        self.assertEqual(replayed["last_seen"], 200)
        self.assertEqual(len(self.store.list_discovered_groups()), 1)

    def test_existing_database_adds_discovery_without_changing_history(self):
        saved = self.store.add_human(
            "bot:GroupMessage:-100#2", "1", "old message", 100, 100, 60
        )
        self.store.connection.execute("DROP TABLE discovered_groups")
        self.store.connection.commit()
        self.store.close()
        self.store = Store(self.path)
        self.assertEqual(self.store.list_discovered_groups(), [])
        self.observe()
        self.assertEqual(
            self.store.read_messages(saved["window"]["umo"], [saved["message"]["id"]]),
            [saved["message"]],
        )
        self.assertEqual(
            self.store.get_window(saved["window"]["umo"], saved["window"]["id"]),
            saved["window"],
        )

    def test_empty_name_does_not_erase_existing_name(self):
        expected = self.observe()["display_name"]
        for name in ("", "   ", None):
            with self.subTest(name=name):
                self.assertEqual(
                    self.observe(display_name=name)["display_name"], expected
                )

    def test_platform_instances_and_topics_have_distinct_candidates(self):
        candidates = [
            self.observe("one:GroupMessage:-100#2", platform_id="one"),
            self.observe("two:GroupMessage:-100#2", platform_id="two"),
            self.observe(
                "one:GroupMessage:-100#3", platform_id="one", group_id="-100#3"
            ),
            self.observe("one:GroupMessage:-100", platform_id="one", group_id="-100"),
        ]
        self.assertEqual(len(self.store.list_discovered_groups()), 4)
        for row in candidates:
            self.assertEqual(self.store.get_discovered_group(row["umo"]), row)

    def test_discovery_writes_no_conversation_data_or_activation_flags(self):
        self.observe()
        self.observe(observed_at=500)
        for table in ("windows", "messages", "summaries"):
            count = self.store.connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            self.assertEqual(count, 0)
        columns = [
            row[1]
            for row in self.store.connection.execute(
                "PRAGMA table_info(discovered_groups)"
            )
        ]
        self.assertEqual(
            set(columns),
            {
                "umo",
                "platform_id",
                "group_id",
                "display_name",
                "first_seen",
                "last_seen",
            },
        )
        self.assertNotIn("enabled", self.observe())

    def test_identifiers_rejected_without_truncating_and_names_bounded(self):
        row = self.observe(display_name="名" * 10000)
        self.assertEqual(row["display_name"], "名" * Store.MAX_DISCOVERED_NAME)
        invalid = [
            {"umo": "x" * (Store.MAX_DISCOVERED_UMO + 1)},
            {"platform_id": "x" * (Store.MAX_DISCOVERED_ID + 1)},
            {"group_id": "x" * (Store.MAX_DISCOVERED_ID + 1)},
            {"umo": ""},
            {"platform_id": " "},
            {"group_id": None},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.observe(**kwargs)
        self.assertEqual(len(self.store.list_discovered_groups()), 1)

    def test_conflicting_identity_and_nonfinite_times_leave_old_row_unchanged(self):
        original = self.observe()
        for kwargs in (
            {"platform_id": "another"},
            {"group_id": "-999"},
            {"observed_at": float("nan")},
            {"observed_at": float("inf")},
            {"display_name": {"text": "not a group name"}},
        ):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.observe(**kwargs)
            self.assertEqual(self.store.get_discovered_group(original["umo"]), original)

    def test_list_is_recent_first_and_result_limit_is_bounded(self):
        with self.store.connection:
            self.store.connection.executemany(
                "INSERT INTO discovered_groups VALUES(?,?,?,?,?,?)",
                [
                    (f"bot:GroupMessage:{i}", "bot", str(i), "group", i, i)
                    for i in range(205)
                ],
            )
        rows = self.store.list_discovered_groups(limit=10000)
        self.assertEqual(len(rows), Store.MAX_RESULTS)
        self.assertEqual(rows[0]["last_seen"], 204)
        self.assertEqual(len(self.store.list_discovered_groups(limit=2)), 2)
