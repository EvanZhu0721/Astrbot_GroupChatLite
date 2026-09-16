"""Bounded, process-local Telegram metadata used by the reply score.

No message text, network access, persistence, or synthetic reply edges live here.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
import time


def _identifier(value):
    if value is None or isinstance(value, bool):
        return ""
    value = str(value)
    return value if 0 < len(value) <= 1024 else ""


def _username(value):
    return _identifier(value).lstrip("@").casefold()


def _is_current_bot(sender_id, sender_username, bot_id, bot_username):
    sender_id, bot_id = _identifier(sender_id), _identifier(bot_id)
    if sender_id and bot_id and sender_id == bot_id:
        return True
    names = {_username(bot_username)}
    if bot_id and not bot_id.lstrip("-").isdigit():
        names.add(_username(bot_id))
    names.discard("")
    return bool(_username(sender_username) and _username(sender_username) in names)


@dataclass(frozen=True)
class StateFeatures:
    reply_hops: int
    recent_target: bool
    arrivals_60s: int
    pending: int

    @property
    def arrival_count(self):
        """Count over the requested frequency window (60 seconds by default)."""
        return self.arrivals_60s


@dataclass(frozen=True)
class _Node:
    at: float
    sender_id: str
    sender_username: str = ""
    parent_id: str = ""
    parent_sender_id: str = ""
    parent_sender_username: str = ""
    bot_id: str = ""
    bot_username: str = ""


class _Room:
    def __init__(self, now):
        self.touched = now
        self.nodes = OrderedDict()
        self.arrivals = OrderedDict()
        self.recent_sender = ""
        self.recent_at = float("-inf")


class ScoringState:
    def __init__(
        self,
        clock=time.monotonic,
        *,
        ttl_seconds=3600,
        max_umos=128,
        max_messages=512,
        max_arrivals=2048,
    ):
        self.clock = clock
        self.ttl_seconds = max(1.0, min(float(ttl_seconds), 3600.0))
        self.max_umos = max(1, min(int(max_umos), 128))
        self.max_messages = max(1, min(int(max_messages), 512))
        self.max_arrivals = max(1, min(int(max_arrivals), 2048))
        self._rooms = OrderedDict()

    def _room(self, umo, now, create=False):
        umo = _identifier(umo)
        if not umo:
            return None
        for key, room in list(self._rooms.items()):
            if now - room.touched >= self.ttl_seconds:
                del self._rooms[key]
        room = self._rooms.get(umo)
        if room is None:
            if not create:
                return None
            room = _Room(now)
            self._rooms[umo] = room
        room.touched = now
        self._rooms.move_to_end(umo)
        while len(self._rooms) > self.max_umos:
            self._rooms.popitem(last=False)
        for key, node in list(room.nodes.items()):
            if now - node.at >= self.ttl_seconds:
                del room.nodes[key]
        while room.arrivals:
            first = next(iter(room.arrivals))
            if now - room.arrivals[first] < self.ttl_seconds:
                break
            del room.arrivals[first]
        return room

    def _put_node(self, room, message_id, node):
        room.nodes[message_id] = node
        room.nodes.move_to_end(message_id)
        while len(room.nodes) > self.max_messages:
            room.nodes.popitem(last=False)

    def observe_message(
        self,
        umo,
        message_id,
        sender_id,
        *,
        reply_to_message_id=None,
        reply_to_sender_id=None,
        reply_to_sender_username=None,
        bot_id=None,
        bot_username=None,
        is_bot=False,
    ):
        """Count one real human arrival and retain only its actual reply edge."""
        message_id, sender_id = _identifier(message_id), _identifier(sender_id)
        if is_bot or not message_id or not sender_id:
            return False
        now = self.clock()
        room = self._room(umo, now, create=True)
        if room is None or message_id in room.arrivals:
            return False
        room.arrivals[message_id] = now
        while len(room.arrivals) > self.max_arrivals:
            room.arrivals.popitem(last=False)
        self._put_node(
            room,
            message_id,
            _Node(
                now,
                sender_id,
                parent_id=_identifier(reply_to_message_id),
                parent_sender_id=_identifier(reply_to_sender_id),
                parent_sender_username=_username(reply_to_sender_username),
                bot_id=_identifier(bot_id),
                bot_username=_username(bot_username),
            ),
        )
        return True

    def record_delivery(
        self,
        umo,
        message_id,
        *,
        target_message_id,
        target_sender_id,
        bot_id=None,
        bot_username=None,
    ):
        """Record a verified bot message; edits refresh recency, never arrivals.

        The response's target is metadata about who was answered, not proof that
        Telegram sent a reply_to edge. Therefore no target edge is synthesized.
        """
        message_id = _identifier(message_id)
        target_sender_id = _identifier(target_sender_id)
        if not message_id or not target_sender_id:
            return
        room = self._room(umo, self.clock(), create=True)
        if room is None:
            return
        now = self.clock()
        self._put_node(
            room, message_id, _Node(now, _identifier(bot_id), _username(bot_username))
        )
        room.recent_sender, room.recent_at = target_sender_id, now

    def snapshot(
        self,
        umo,
        message_id,
        sender_id,
        *,
        pending=1,
        frequency_seconds=60,
        recent_seconds=120,
    ):
        now = self.clock()
        room = self._room(umo, now)
        pending = max(0, min(int(pending), 1_000_000))
        if room is None:
            return StateFeatures(0, False, 0, pending)
        frequency_seconds = self._seconds(frequency_seconds)
        recent_seconds = self._seconds(recent_seconds)
        count = sum(now - at < frequency_seconds for at in room.arrivals.values())
        recent = bool(
            _identifier(sender_id)
            and room.recent_sender == _identifier(sender_id)
            and 0 <= now - room.recent_at < recent_seconds
        )
        hops = 0
        node = room.nodes.get(_identifier(message_id))
        if node is not None:
            bot_id, bot_username = node.bot_id, node.bot_username
            visited = {_identifier(message_id)}
            for depth in range(1, 9):
                if not node.parent_id or node.parent_id in visited:
                    break
                visited.add(node.parent_id)
                if _is_current_bot(
                    node.parent_sender_id,
                    node.parent_sender_username,
                    bot_id,
                    bot_username,
                ):
                    hops = depth
                    break
                parent = room.nodes.get(node.parent_id)
                if parent is None:
                    break
                if _is_current_bot(
                    parent.sender_id, parent.sender_username, bot_id, bot_username
                ):
                    hops = depth
                    break
                node = parent
        return StateFeatures(hops, recent, count, pending)

    def _seconds(self, value):
        value = float(value)
        return max(0.0, min(value, self.ttl_seconds)) if math.isfinite(value) else 0.0

    def clear(self):
        self._rooms.clear()
