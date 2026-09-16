"""Local, bounded-query conversation storage without external history sources."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path


class Store:
    MAX_RESULTS = 200
    MAX_DISCOVERED_UMO = 1024
    MAX_DISCOVERED_ID = 256
    MAX_DISCOVERED_NAME = 512
    DISCOVERED_COLUMNS = "umo,platform_id,group_id,display_name,first_seen,last_seen"

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS windows (
                id INTEGER PRIMARY KEY, umo TEXT NOT NULL,
                start_at REAL NOT NULL, last_human_at REAL NOT NULL,
                closed_at REAL, version INTEGER NOT NULL DEFAULT 0
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_window
                ON windows(umo) WHERE closed_at IS NULL;
            CREATE INDEX IF NOT EXISTS window_time ON windows(umo, start_at);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, umo TEXT NOT NULL,
                window_id INTEGER NOT NULL REFERENCES windows(id),
                source_message_id TEXT NOT NULL, role TEXT NOT NULL,
                text TEXT NOT NULL, event_at REAL NOT NULL, observed_at REAL NOT NULL,
                sender_id TEXT NOT NULL DEFAULT '', sender_name TEXT NOT NULL DEFAULT '',
                UNIQUE(umo, source_message_id)
            );
            CREATE INDEX IF NOT EXISTS message_window ON messages(umo,window_id,event_at,id);
            CREATE TABLE IF NOT EXISTS summaries (
                window_id INTEGER PRIMARY KEY REFERENCES windows(id),
                umo TEXT NOT NULL, text TEXT NOT NULL, source_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS discovered_groups (
                umo TEXT PRIMARY KEY, platform_id TEXT NOT NULL,
                group_id TEXT NOT NULL, display_name TEXT NOT NULL DEFAULT '',
                first_seen REAL NOT NULL, last_seen REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS discovered_group_recency
                ON discovered_groups(last_seen DESC,umo);
        """)
        # Older databases retain every row; absent relationship evidence stays empty.
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(messages)")
        }
        with self.connection:
            for name in (
                "reply_to_message_id",
                "reply_to_sender_id",
                "response_to_message_id",
                "response_to_sender_id",
            ):
                if name not in columns:
                    self.connection.execute(
                        f"ALTER TABLE messages ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                    )

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("A finite timestamp is required")
        return value

    @classmethod
    def _limit(cls, value):
        return max(1, min(int(value), cls.MAX_RESULTS))

    def _one(self, sql, args):
        row = self.connection.execute(sql, args).fetchone()
        return self._record(row) if row else None

    @staticmethod
    def _record(row):
        result = dict(row)
        if "last_human_at" in result:
            result["status"] = "open" if result["closed_at"] is None else "closed"
        return result

    def _many(self, sql, args):
        return [
            self._record(row) for row in self.connection.execute(sql, args).fetchall()
        ]

    @staticmethod
    def _discovered_identifier(value, maximum):
        # Never truncate identity fields: two long IDs must not become one group.
        if not isinstance(value, str) or len(value) > maximum or not value.strip():
            raise ValueError(
                "Discovered group identifier is empty or exceeds its limit"
            )
        return value

    def observe_group(self, umo, platform_id, group_id, display_name, observed_at):
        """Remember metadata only; discovery never enables a group or opens a window."""
        umo = self._discovered_identifier(umo, self.MAX_DISCOVERED_UMO)
        platform_id = self._discovered_identifier(platform_id, self.MAX_DISCOVERED_ID)
        group_id = self._discovered_identifier(group_id, self.MAX_DISCOVERED_ID)
        if display_name is None:
            display_name = ""
        if not isinstance(display_name, str):
            raise ValueError("Discovered group display name must be text")
        display_name = display_name[: self.MAX_DISCOVERED_NAME].strip()
        observed_at = self._number(observed_at)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            previous = self.get_discovered_group(umo)
            if previous and (
                previous["platform_id"] != platform_id
                or previous["group_id"] != group_id
            ):
                raise ValueError("Discovered group identity conflicts with this UMO")
            self.connection.execute(
                "INSERT INTO discovered_groups(umo,platform_id,group_id,display_name,first_seen,last_seen) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(umo) DO UPDATE SET "
                "display_name=CASE WHEN excluded.display_name<>'' THEN excluded.display_name "
                "ELSE discovered_groups.display_name END, "
                "last_seen=MAX(discovered_groups.last_seen,excluded.last_seen)",
                (umo, platform_id, group_id, display_name, observed_at, observed_at),
            )
            return self.get_discovered_group(umo)

    def list_discovered_groups(self, limit=200):
        """Return at most MAX_RESULTS candidates, newest observations first."""
        return self._many(
            f"SELECT {self.DISCOVERED_COLUMNS} FROM discovered_groups "
            "ORDER BY last_seen DESC,umo LIMIT ?",
            (self._limit(limit),),
        )

    def get_discovered_group(self, umo):
        umo = self._discovered_identifier(umo, self.MAX_DISCOVERED_UMO)
        return self._one(
            f"SELECT {self.DISCOVERED_COLUMNS} FROM discovered_groups WHERE umo=?",
            (umo,),
        )

    def get_window(self, umo, window_id):
        return self._one("SELECT * FROM windows WHERE umo=? AND id=?", (umo, window_id))

    def active_window(self, umo):
        return self._one(
            "SELECT * FROM windows WHERE umo=? AND closed_at IS NULL", (umo,)
        )

    def latest_previous_window(self, umo, window_id=None):
        current = (
            self.get_window(umo, window_id)
            if window_id is not None
            else self.active_window(umo)
        )
        if window_id is not None and current is None:
            return None
        if current:
            return self._one(
                "SELECT * FROM windows WHERE umo=? AND (start_at<? OR (start_at=? AND id<?)) ORDER BY start_at DESC,id DESC LIMIT 1",
                (umo, current["start_at"], current["start_at"], current["id"]),
            )
        return self._one(
            "SELECT * FROM windows WHERE umo=? AND closed_at IS NOT NULL ORDER BY start_at DESC,id DESC LIMIT 1",
            (umo,),
        )

    def _duplicate(self, umo, source_message_id):
        old = self._one(
            "SELECT * FROM messages WHERE umo=? AND source_message_id=?",
            (umo, source_message_id),
        )
        if old:
            return {
                "inserted": False,
                "message": old,
                "window": self.get_window(umo, old["window_id"]),
            }
        return None

    def _insert(
        self,
        umo,
        window_id,
        source_message_id,
        role,
        text,
        event_at,
        observed_at,
        sender_id="",
        sender_name="",
        reply_to_message_id="",
        reply_to_sender_id="",
        response_to_message_id="",
        response_to_sender_id="",
    ):
        cur = self.connection.execute(
            "INSERT INTO messages(umo,window_id,source_message_id,role,text,event_at,observed_at,sender_id,sender_name,reply_to_message_id,reply_to_sender_id,response_to_message_id,response_to_sender_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                umo,
                window_id,
                str(source_message_id),
                role,
                text,
                event_at,
                observed_at,
                str(sender_id),
                str(sender_name),
                str(reply_to_message_id or ""),
                str(reply_to_sender_id or ""),
                str(response_to_message_id or ""),
                str(response_to_sender_id or ""),
            ),
        )
        self.connection.execute(
            "UPDATE windows SET version=version+1 WHERE id=? AND umo=?",
            (window_id, umo),
        )
        self.connection.execute(
            "DELETE FROM summaries WHERE window_id=? AND umo=?", (window_id, umo)
        )
        return {
            "inserted": True,
            "message": self._one("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)),
            "window": self.get_window(umo, window_id),
        }

    def add_human(
        self,
        umo,
        source_message_id,
        text,
        event_at,
        observed_at,
        idle_seconds,
        *,
        sender_id="",
        sender_name="",
        reply_to_message_id="",
        reply_to_sender_id="",
    ):
        """Assign by source time; delayed events cannot advance a newer window."""
        event_at, observed_at = self._number(event_at), self._number(observed_at)
        idle_seconds = self._number(idle_seconds)
        if (
            not umo
            or source_message_id is None
            or not isinstance(text, str)
            or idle_seconds <= 0
        ):
            raise ValueError("Invalid message metadata")
        source_message_id = str(source_message_id)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            duplicate = self._duplicate(umo, source_message_id)
            if duplicate:
                return duplicate
            active = self.active_window(umo)
            target = None
            if active and event_at >= active["last_human_at"]:
                if event_at - active["last_human_at"] < idle_seconds:
                    target = active
                else:
                    self.connection.execute(
                        "UPDATE windows SET closed_at=? WHERE id=?",
                        (active["last_human_at"] + idle_seconds, active["id"]),
                    )
            else:
                # Half-open inactivity interval: a message exactly at the boundary starts a new window.
                target = self._one(
                    "SELECT * FROM windows WHERE umo=? AND start_at<=? AND last_human_at+?>? ORDER BY start_at DESC,id DESC LIMIT 1",
                    (umo, event_at, idle_seconds, event_at),
                )
            if target is None:
                # Backfilled history gets its own closed window, never the currently active one.
                historical = (
                    bool(active and event_at < active["start_at"])
                    or observed_at - event_at >= idle_seconds
                )
                cur = self.connection.execute(
                    "INSERT INTO windows(umo,start_at,last_human_at,closed_at) VALUES(?,?,?,?)",
                    (
                        umo,
                        event_at,
                        event_at,
                        event_at + idle_seconds if historical else None,
                    ),
                )
                window_id = cur.lastrowid
            else:
                window_id = target["id"]
                self.connection.execute(
                    "UPDATE windows SET last_human_at=MAX(last_human_at,?),closed_at=CASE WHEN closed_at IS NULL THEN NULL ELSE MAX(closed_at,?+?) END WHERE id=?",
                    (event_at, event_at, idle_seconds, window_id),
                )
            return self._insert(
                umo,
                window_id,
                source_message_id,
                "user",
                text,
                event_at,
                observed_at,
                sender_id,
                sender_name,
                reply_to_message_id,
                reply_to_sender_id,
            )

    def add_bot(
        self,
        umo,
        window_id,
        source_message_id,
        text,
        event_at,
        observed_at,
        *,
        sender_id="",
        sender_name="",
        reply_to_message_id="",
        reply_to_sender_id="",
        response_to_message_id="",
        response_to_sender_id="",
    ):
        """Persist final output in its originating window, even after that window closes."""
        event_at, observed_at = self._number(event_at), self._number(observed_at)
        if not umo or source_message_id is None or not isinstance(text, str):
            raise ValueError("Invalid message metadata")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if not self.get_window(umo, window_id):
                raise ValueError("Window does not belong to this session")
            duplicate = self._duplicate(umo, str(source_message_id))
            if duplicate:
                return duplicate
            return self._insert(
                umo,
                window_id,
                source_message_id,
                "assistant",
                text,
                event_at,
                observed_at,
                sender_id,
                sender_name,
                reply_to_message_id,
                reply_to_sender_id,
                response_to_message_id,
                response_to_sender_id,
            )

    def list_open_windows(self, limit=200):
        return self._many(
            "SELECT * FROM windows WHERE closed_at IS NULL ORDER BY last_human_at,id LIMIT ?",
            (self._limit(limit),),
        )

    def idle_close(self, now, idle_seconds, exclude_umos=(), target_umos=None):
        now, idle_seconds = self._number(now), self._number(idle_seconds)
        if idle_seconds <= 0:
            raise ValueError("Idle duration must be positive")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            # Bound each maintenance batch; callers may drain additional batches when needed.
            excluded = tuple(exclude_umos)
            clause = (
                " AND umo NOT IN (" + ",".join("?" for _ in excluded) + ")"
                if excluded
                else ""
            )
            targets = tuple(target_umos) if target_umos is not None else ()
            if target_umos is not None and not targets:
                return []
            if targets:
                clause += " AND umo IN (" + ",".join("?" for _ in targets) + ")"
            rows = self._many(
                "SELECT * FROM windows WHERE closed_at IS NULL AND last_human_at+?<=?"
                + clause
                + " ORDER BY last_human_at LIMIT ?",
                (idle_seconds, now, *excluded, *targets, self.MAX_RESULTS),
            )
            for row in rows:
                row["closed_at"] = row["last_human_at"] + idle_seconds
                row["status"] = "closed"
                self.connection.execute(
                    "UPDATE windows SET closed_at=? WHERE id=?",
                    (row["closed_at"], row["id"]),
                )
            return rows

    def window_messages(self, umo, window_id, limit=50, before_id=None):
        """Return latest matching records chronologically; before_id is an insertion cursor."""
        sql = "SELECT * FROM messages WHERE umo=? AND window_id=?"
        args = [umo, window_id]
        if before_id is not None:
            sql += " AND id<?"
            args.append(before_id)
        args.append(self._limit(limit))
        rows = self._many(sql + " ORDER BY event_at DESC,id DESC LIMIT ?", args)
        return list(reversed(rows))

    def save_summary(self, umo, window_id, text, source_version):
        if not isinstance(text, str):
            raise ValueError("Summary must be text")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            window = self.get_window(umo, window_id)
            if not window or window["version"] != source_version:
                return False
            self.connection.execute(
                "INSERT INTO summaries(window_id,umo,text,source_version) VALUES(?,?,?,?) ON CONFLICT(window_id) DO UPDATE SET text=excluded.text,source_version=excluded.source_version",
                (window_id, umo, text, source_version),
            )
            return True

    def get_summary(self, umo, window_id):
        return self._one(
            "SELECT s.* FROM summaries s JOIN windows w ON w.id=s.window_id AND w.umo=s.umo WHERE s.umo=? AND s.window_id=? AND s.source_version=w.version",
            (umo, window_id),
        )

    def list_unsummarized_closed(self, limit=20):
        return self._many(
            "SELECT w.* FROM windows w LEFT JOIN summaries s ON s.window_id=w.id AND s.umo=w.umo AND s.source_version=w.version WHERE w.closed_at IS NOT NULL AND s.window_id IS NULL ORDER BY w.closed_at DESC,w.id DESC LIMIT ?",
            (self._limit(limit),),
        )

    def search_messages(self, umo, query, limit=20, *, before=None, after=None):
        """Literal substring search: SQL wildcard characters have no special meaning."""
        if not isinstance(query, str) or not query:
            return []
        sql = "SELECT * FROM messages WHERE umo=? AND instr(text,?)>0"
        args = [umo, query]
        for bound, operator in ((before, "<"), (after, ">=")):
            if bound is not None:
                sql += " AND event_at" + operator + "?"
                args.append(self._number(bound))
        return self._many(
            sql + " ORDER BY event_at DESC,id DESC LIMIT ?", [*args, self._limit(limit)]
        )

    def read_messages(self, umo, ids, limit=50):
        ids = list(dict.fromkeys(ids))[: self._limit(limit)]
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        return self._many(
            f"SELECT * FROM messages WHERE umo=? AND id IN ({marks}) ORDER BY event_at,id",
            [umo, *ids],
        )
