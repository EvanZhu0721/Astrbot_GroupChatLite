import importlib.util
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "lite_store", Path(__file__).resolve().parents[1] / "store.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Store = module.Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "history.sqlite3")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_relationships_keep_telegram_ids_separate_from_local_ids(self):
        saved = self.human(
            "900",
            100,
            sender_id="alice",
            reply_to_message_id="899",
            reply_to_sender_id="bob",
        )
        self.assertNotEqual(str(saved["message"]["id"]), "900")
        self.assertEqual(saved["message"]["reply_to_message_id"], "899")
        bot = self.store.add_bot(
            "tg:group:1:topic:1",
            saved["window"]["id"],
            "gcl:900",
            "answer",
            101,
            101,
            sender_id="bot",
            response_to_message_id="900",
            response_to_sender_id="alice",
        )
        self.assertEqual(bot["message"]["response_to_message_id"], "900")
        self.assertEqual(bot["message"]["reply_to_message_id"], "")

    def test_legacy_database_migrates_without_inventing_relationships(self):
        saved = self.human("900", 100, sender_id="legacy", sender_name="Unknown")
        self.store.close()
        path = Path(self.tmp.name) / "history.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            for name in (
                "reply_to_message_id",
                "reply_to_sender_id",
                "response_to_message_id",
                "response_to_sender_id",
            ):
                connection.execute(f"ALTER TABLE messages DROP COLUMN {name}")
        self.store = Store(path)
        record = self.store.window_messages(
            "tg:group:1:topic:1", saved["window"]["id"]
        )[0]
        self.assertEqual(record["sender_id"], "legacy")
        self.assertEqual(record["source_message_id"], "900")
        self.assertEqual(record["response_to_sender_id"], "")
        self.assertEqual(record["reply_to_message_id"], "")

    def human(self, source, stamp, umo="tg:group:1:topic:1", **kwargs):
        return self.store.add_human(
            umo,
            source,
            kwargs.pop("text", source),
            stamp,
            kwargs.pop("observed_at", stamp),
            10,
            **kwargs,
        )

    def test_idle_boundary_and_bot_does_not_extend_window(self):
        first = self.human("1", 100)
        umo, wid = first["window"]["umo"], first["window"]["id"]
        self.store.add_bot(umo, wid, "bot:1", "answer", 108, 108)
        self.assertEqual(self.store.get_window(umo, wid)["last_human_at"], 100)
        second = self.human("2", 110)
        self.assertNotEqual(wid, second["window"]["id"])
        self.assertEqual(self.store.get_window(umo, wid)["status"], "closed")
        self.assertEqual(
            self.store.latest_previous_window(umo, second["window"]["id"])["id"], wid
        )

    def test_dedup_preserves_original_window_and_version(self):
        first = self.human("1", 100)
        self.human("2", 200)
        duplicate = self.human("1", 300, text="changed")
        self.assertFalse(duplicate["inserted"])
        self.assertEqual(
            first["window"], duplicate["window"] | {"closed_at": None, "status": "open"}
        )
        self.assertEqual(duplicate["message"]["text"], "1")
        self.assertEqual(duplicate["window"]["version"], 1)

    def test_full_session_topic_isolation(self):
        a = self.human("1", 100, umo="tg:group:1:topic:1", text="needle")
        b = self.human("1", 100, umo="tg:group:1:topic:2", text="needle")
        self.assertNotEqual(a["window"]["id"], b["window"]["id"])
        self.assertEqual(
            self.store.window_messages(a["window"]["umo"], b["window"]["id"]), []
        )
        self.assertEqual(
            self.store.read_messages(a["window"]["umo"], [b["message"]["id"]]), []
        )
        self.assertEqual(
            len(self.store.search_messages(a["window"]["umo"], "needle")), 1
        )
        with self.assertRaises(ValueError):
            self.store.add_bot(
                a["window"]["umo"], b["window"]["id"], "b", "wrong", 101, 101
            )

    def test_late_bot_invalidates_summary_and_stale_writer_rejected(self):
        first = self.human("1", 100)
        umo, wid = first["window"]["umo"], first["window"]["id"]
        self.human("2", 120)
        self.assertTrue(self.store.save_summary(umo, wid, "summary", 1))
        self.store.add_bot(umo, wid, "bot:late", "late answer", 130, 130)
        self.assertIsNone(self.store.get_summary(umo, wid))
        self.assertFalse(self.store.save_summary(umo, wid, "stale", 1))
        self.assertTrue(self.store.save_summary(umo, wid, "fresh", 2))
        self.assertEqual(len(self.store.window_messages(umo, wid)), 2)
        self.assertEqual(self.store.get_summary(umo, wid)["source_version"], 2)
        self.assertEqual(self.store.active_window(umo)["last_human_at"], 120)

    def test_idle_close_excludes_inflight_and_lists_unsummarized(self):
        self.human("a", 100, umo="a")
        self.human("b", 100, umo="b")
        closed = self.store.idle_close(110, 10, exclude_umos=("a",))
        self.assertEqual([w["umo"] for w in closed], ["b"])
        self.assertEqual(closed[0]["status"], "closed")
        self.assertIsNotNone(self.store.active_window("a"))
        self.assertEqual(
            [w["umo"] for w in self.store.list_unsummarized_closed()], ["b"]
        )
        b = closed[0]
        self.store.save_summary("b", b["id"], "brief", b["version"])
        self.assertEqual(self.store.list_unsummarized_closed(), [])

    def test_search_literal_wildcards_dates_and_bound(self):
        for i, text in enumerate(("literal %_\\", "plain abc", "literal %_\\")):
            self.human(str(i), 100 + i, text=text)
        umo = "tg:group:1:topic:1"
        self.assertEqual(len(self.store.search_messages(umo, "%_\\")), 2)
        self.assertEqual(
            len(self.store.search_messages(umo, "%_\\", after=101, before=103)), 1
        )
        self.assertEqual(len(self.store.search_messages(umo, "literal", limit=1)), 1)
        self.assertEqual(self.store.search_messages(umo, "' OR 1=1 --"), [])

    def test_late_human_keeps_original_window_and_current_clock(self):
        a = self.human("a", 100)
        self.human("b", 105)
        current = self.human("c", 120)
        late = self.human("late", 103, observed_at=150)
        self.assertEqual(late["window"]["id"], a["window"]["id"])
        self.assertEqual(late["window"]["last_human_at"], 105)
        self.assertEqual(
            self.store.active_window(a["window"]["umo"])["id"], current["window"]["id"]
        )
        self.assertEqual(
            self.store.active_window(a["window"]["umo"])["last_human_at"], 120
        )
        self.assertEqual(late["message"]["event_at"], 103)
        self.assertEqual(late["message"]["observed_at"], 150)

    def test_ancient_unmatched_message_gets_closed_historical_window(self):
        current = self.human("current", 200)
        ancient = self.human("ancient", 50, observed_at=201)
        self.assertEqual(ancient["window"]["status"], "closed")
        self.assertNotEqual(current["window"]["id"], ancient["window"]["id"])
        self.assertEqual(
            self.store.active_window(current["window"]["umo"])["id"],
            current["window"]["id"],
        )

    def test_latest_messages_order_and_sender_fields(self):
        a = self.human("a", 100, sender_id="123", sender_name="User")
        self.human("c", 104)
        self.human("b", 102, observed_at=106)
        messages = self.store.window_messages(
            a["window"]["umo"], a["window"]["id"], limit=2
        )
        self.assertEqual([m["text"] for m in messages], ["b", "c"])
        self.assertEqual(a["message"]["sender_id"], "123")
        self.assertEqual(a["message"]["sender_name"], "User")

    def test_failed_transaction_rolls_back_window_change(self):
        a = self.human("a", 100)
        self.store.connection.execute(
            "CREATE TRIGGER fail_insert BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
        )
        self.store.connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.human("b", 120)
        self.assertEqual(
            self.store.active_window(a["window"]["umo"])["id"], a["window"]["id"]
        )
        self.assertEqual(self.store.active_window(a["window"]["umo"])["version"], 1)

    def test_persistence_without_pruning(self):
        a = self.human("0", 100)
        for i in range(1, 205):
            self.human(str(i), 100 + i)
        rows = self.store.window_messages(
            a["window"]["umo"], a["window"]["id"], limit=9999
        )
        self.assertEqual(len(rows), 200)
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[
                0
            ],
            205,
        )
        self.store.close()
        self.store = Store(Path(self.tmp.name) / "history.sqlite3")
        self.assertEqual(
            self.store.get_window(a["window"]["umo"], a["window"]["id"])["version"], 205
        )


if __name__ == "__main__":
    unittest.main()
