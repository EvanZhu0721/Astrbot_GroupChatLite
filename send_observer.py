"""Observe successful Telegram text delivery without replacing client objects."""

import asyncio
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import inspect
import logging
import weakref

logger = logging.getLogger(__name__)
_SCOPE = ContextVar("groupchat_lite_send_scope", default=None)
_IN_CLIENT_CALL = ContextVar("groupchat_lite_in_client_call", default=False)
_CLIENT_PATCHES = {}


def _base_client(client):
    """Recognize BetterTG's existing wrapper without changing its identity."""
    seen = set()
    for _ in range(4):
        if id(client) in seen:
            break
        seen.add(id(client))
        wrapped = getattr(client, "original_client", None)
        if wrapped is None or not callable(getattr(wrapped, "send_message", None)):
            break
        client = wrapped
    return client


def _method_wrapper(original):
    @wraps(original)
    async def observed(client, *args, **kwargs):
        scope = _SCOPE.get()
        if (
            scope is None
            or asyncio.current_task() is not scope[1]
            or client is not scope[2]
            or _IN_CLIENT_CALL.get()
        ):
            return await original(client, *args, **kwargs)
        token = _IN_CLIENT_CALL.set(True)
        try:
            response = await original(client, *args, **kwargs)
        finally:
            _IN_CLIENT_CALL.reset(token)
        await scope[0].record(response, is_streaming=scope[3])
        return response

    return observed


def _acquire_client(client_type):
    entry = _CLIENT_PATCHES.get(client_type)
    if entry is not None:
        entry["users"] += 1
        return
    entry = {"users": 1, "methods": {}}
    try:
        for name in ("send_message", "edit_message_text"):
            original = getattr(client_type, name, None)
            if not callable(original):
                continue
            owned = name in vars(client_type)
            descriptor = vars(client_type).get(name)
            wrapper = _method_wrapper(original)
            setattr(client_type, name, wrapper)
            entry["methods"][name] = (owned, descriptor, wrapper)
        if "send_message" not in entry["methods"]:
            raise TypeError("Telegram client has no class send_message method")
    except Exception:
        _restore_methods(client_type, entry)
        raise
    _CLIENT_PATCHES[client_type] = entry


def _restore_methods(client_type, entry):
    for name, (owned, descriptor, wrapper) in entry["methods"].items():
        if vars(client_type).get(name) is wrapper:
            if owned:
                setattr(client_type, name, descriptor)
            else:
                delattr(client_type, name)


def _release_client(client_type):
    entry = _CLIENT_PATCHES.get(client_type)
    if entry is None:
        return
    entry["users"] -= 1
    if entry["users"] == 0:
        _restore_methods(client_type, entry)
        del _CLIENT_PATCHES[client_type]


async def _callback(callback, *args):
    try:
        result = callback(*args)
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        logger.warning("Telegram observation callback failed (CancelledError)")
    except Exception as exc:
        logger.warning("Telegram observation callback failed (%s)", type(exc).__name__)


class SendObserver:
    """Scope one event's sends; shared class wrappers never retain the event.

    Ordinary external history uses on_success/should_record. Optional on_delivery
    receives successful ordinary and streaming text receipts, including edits,
    behind its separate live should_observe predicate.
    """

    def __init__(
        self, event, on_success, should_record, *, on_delivery=None, should_observe=None
    ):
        self.event = event
        self.on_success = on_success
        self.should_record = should_record
        self.on_delivery = on_delivery
        self.should_observe = should_observe or should_record
        self.active = True
        self._seen = OrderedDict()
        self._original_send = event.send
        self._original_streaming = getattr(event, "send_streaming", None)
        self._event_wrappers = {}
        self._client = _base_client(event.client)
        group = str(event.get_group_id())
        self._chat, _, thread = group.partition("#")
        self._thread = thread or None
        client_type = type(self._client)
        _acquire_client(client_type)
        self._finalizer = weakref.finalize(self, _release_client, client_type)
        try:
            self._wrap_event("send", "_original_send", False)
            if callable(self._original_streaming):
                self._wrap_event("send_streaming", "_original_streaming", True)
        except Exception:
            self.detach()
            raise

    def _wrap_event(self, name, original_name, is_streaming):
        event = self.event
        owned, original = name in vars(event), vars(event).get(name)

        async def scoped_send(*args, **kwargs):
            if not self.allowed():
                return await getattr(self, original_name)(*args, **kwargs)
            token = _SCOPE.set(
                (self, asyncio.current_task(), self._client, is_streaming)
            )
            try:
                return await getattr(self, original_name)(*args, **kwargs)
            finally:
                _SCOPE.reset(token)

        setattr(event, name, scoped_send)
        self._event_wrappers[name] = (owned, original, scoped_send)

    def allowed(self):
        try:
            return self.active and bool(self.should_observe())
        except Exception as exc:
            logger.warning("Telegram send observation failed (%s)", type(exc).__name__)
            return False

    async def record(self, response, *, is_streaming=False):
        try:
            if not self.allowed():
                return
            chat = getattr(response, "chat", None)
            if str(getattr(chat, "id", None)) != self._chat:
                return
            thread = (
                getattr(response, "message_thread_id", None)
                if getattr(response, "is_topic_message", False)
                else None
            )
            if getattr(chat, "is_forum", False) is True and thread == 1:
                thread = None
            if (str(thread) if thread else None) != self._thread:
                return
            message_id = getattr(response, "message_id", None)
            text = getattr(response, "text", None)
            if type(message_id) is not int or message_id <= 0:
                return
            if not isinstance(text, str) or not text.strip():
                return
            key = f"telegram:{self._chat}:{self._thread or 'general'}:{message_id}"
            if self.on_delivery is not None:
                await _callback(self.on_delivery, response, key, is_streaming)
            if (
                not self.allowed()
                or is_streaming
                or key in self._seen
                or not self.should_record()
            ):
                return
            await _callback(self.on_success, text, key)
            self._seen[key] = None
            while len(self._seen) > 512:
                self._seen.popitem(last=False)
        except Exception as exc:
            logger.warning("Telegram send observation failed (%s)", type(exc).__name__)

    def detach(self):
        """Stop observing and restore only methods still owned by this handle."""
        self.active = False
        for name, (owned, original, wrapper) in self._event_wrappers.items():
            if getattr(self.event, name, None) is wrapper:
                if owned:
                    setattr(self.event, name, original)
                else:
                    delattr(self.event, name)
        self._finalizer()


def install_send_observer(
    event, on_success, should_record, *, on_delivery=None, should_observe=None
):
    return SendObserver(
        event,
        on_success,
        should_record,
        on_delivery=on_delivery,
        should_observe=should_observe,
    )
