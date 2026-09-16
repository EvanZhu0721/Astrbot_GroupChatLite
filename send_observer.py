"""Observe confirmed ordinary Telegram text sends on one event only."""

import asyncio
import logging

logger = logging.getLogger(__name__)


class _ClientProxy:
    def __init__(self, client, observer, task):
        self._client = client
        self._observer = observer
        self._task = task

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def send_message(self, *args, **kwargs):
        response = await self._client.send_message(*args, **kwargs)
        if asyncio.current_task() is self._task:
            self._observer.record(response)
        return response


class SendObserver:
    """Wrap an event's ordinary send without changing the shared Telegram bot.

    The caller supplies a live scope/ownership predicate and synchronous callback.
    No original message chain is recorded; only successful Telegram response text.
    Streaming, media captions, proactive calls and other tasks are outside scope.
    """

    def __init__(self, event, on_success, should_record):
        self.event = event
        self.on_success = on_success
        self.should_record = should_record
        self.active = True
        self._lock = asyncio.Lock()
        self._seen = set()
        self._original_send = event.send
        self._had_instance_send = "send" in vars(event)
        self._instance_send = vars(event).get("send")
        group = str(event.get_group_id())
        self._chat, _, thread = group.partition("#")
        self._thread = thread or None

        async def observed_send(*args, **kwargs):
            async with self._lock:
                if not self.allowed():
                    return await self._original_send(*args, **kwargs)
                client = event.client
                proxy = _ClientProxy(client, self, asyncio.current_task())
                event.client = proxy
                try:
                    return await self._original_send(*args, **kwargs)
                finally:
                    if event.client is proxy:
                        event.client = client

        self._wrapper = observed_send
        event.send = observed_send

    def allowed(self):
        try:
            return self.active and bool(self.should_record())
        except Exception as exc:
            logger.warning("Telegram send observation failed (%s)", type(exc).__name__)
            return False

    def record(self, response):
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
            if key in self._seen:
                return
            self.on_success(text, key)
            self._seen.add(key)
        except Exception as exc:
            logger.warning("Telegram send observation failed (%s)", type(exc).__name__)

    def detach(self):
        """Stop collecting and restore only the wrapper owned by this observer."""
        self.active = False
        if self.event.send is self._wrapper:
            if self._had_instance_send:
                self.event.send = self._instance_send
            else:
                del self.event.send


def install_send_observer(event, on_success, should_record):
    """Install one event-local observer; the caller owns its cleanup lifecycle."""
    return SendObserver(event, on_success, should_record)
