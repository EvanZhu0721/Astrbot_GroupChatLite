"""Bounded, in-memory image references; this module never opens media files."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice
import math
import time


@dataclass(frozen=True)
class MediaRef:
    message_id: int
    image_url: str


@dataclass(frozen=True)
class _CachedImage:
    reference: MediaRef
    expires_at: float


class MediaCache:
    """Keep references only, isolated by full UMO and local SQL message ID.

    Limits count individual images. Capacity eviction follows insertion order;
    reading a reference does not extend its TTL or alter its eviction priority.
    Calls are synchronous and intended for the plugin's event-loop thread.
    """

    MAX_IMAGES = 128
    MAX_IMAGES_PER_UMO = 16
    MAX_IMAGES_PER_MESSAGE = 8
    MAX_IMAGE_URL_CHARS = 8192
    MAX_UMO_CHARS = 1024

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._images: OrderedDict[tuple[str, int, int], _CachedImage] = OrderedDict()

    @classmethod
    def _validate_umo(cls, umo):
        if not isinstance(umo, str) or not umo.strip() or len(umo) > cls.MAX_UMO_CHARS:
            raise ValueError("UMO must be nonempty text within its length limit")

    @staticmethod
    def _validate_message_id(message_id):
        if (
            not isinstance(message_id, int)
            or isinstance(message_id, bool)
            or message_id <= 0
        ):
            raise ValueError("Message ID must be a positive integer")

    def _expire(self, now):
        for key in tuple(self._images):
            if self._images[key].expires_at <= now:
                del self._images[key]

    def put(
        self,
        umo: str,
        message_id: int,
        images: Iterable[str],
        ttl_seconds: float = 1200,
    ) -> None:
        """Replace a message's references; TTL zero explicitly forgets it.

        Inspect at most the first eight supplied references. Empty, non-string,
        or oversized references are ignored; paths and URLs remain opaque text.
        No existence check, download, copy, deletion, or persistence takes place.
        """
        self._validate_umo(umo)
        self._validate_message_id(message_id)
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("TTL must be a finite nonnegative number") from exc
        if isinstance(ttl_seconds, bool) or not math.isfinite(ttl) or ttl < 0:
            raise ValueError("TTL must be a finite nonnegative number")
        now = self._clock()
        expires_at = now + ttl
        if not math.isfinite(expires_at):
            raise ValueError("Expiration must be finite")

        refs = []
        if ttl > 0:
            if isinstance(images, (str, bytes)):
                raise ValueError("Images must be an iterable of image references")
            for reference in islice(images, self.MAX_IMAGES_PER_MESSAGE):
                if (
                    isinstance(reference, str)
                    and reference.strip()
                    and len(reference) <= self.MAX_IMAGE_URL_CHARS
                    and reference not in refs
                ):
                    refs.append(reference)

        self._expire(now)
        for key in tuple(self._images):
            if key[:2] == (umo, message_id):
                del self._images[key]
        for index, reference in enumerate(refs):
            self._images[umo, message_id, index] = _CachedImage(
                MediaRef(message_id, reference), expires_at
            )

        group_keys = [key for key in self._images if key[0] == umo]
        excess = len(group_keys) - self.MAX_IMAGES_PER_UMO
        for key in group_keys[: max(0, excess)]:
            del self._images[key]
        while len(self._images) > self.MAX_IMAGES:
            self._images.popitem(last=False)

    def select(
        self, umo: str, message_ids: Iterable[int], limit: int
    ) -> list[MediaRef]:
        """Return the newest requested images in supplied message order.

        The input order defines recency, independent of put order or SQL IDs.
        Repeated paths belong to their latest requested message. Take the last
        ``limit`` unique images, then preserve their message and image order.
        """
        self._validate_umo(umo)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("Selection limit must be a nonnegative integer")
        self._expire(self._clock())
        if limit == 0:
            return []

        available_ids = {key[1] for key in self._images if key[0] == umo}
        positions = {}
        for position, message_id in enumerate(message_ids):
            self._validate_message_id(message_id)
            if message_id in available_ids:
                # Only cached IDs occupy this map, even for a large input list.
                positions[message_id] = position

        ordered = sorted(
            (
                (positions[message_id], image_index, cached.reference)
                for (scope, message_id, image_index), cached in self._images.items()
                if scope == umo and message_id in positions
            ),
            key=lambda item: item[:2],
        )
        newest_by_path = {}
        for position, image_index, reference in ordered:
            newest_by_path[reference.image_url] = (position, image_index, reference)
        unique = sorted(newest_by_path.values(), key=lambda item: item[:2])
        return [item[2] for item in unique[-min(limit, self.MAX_IMAGES_PER_UMO) :]]

    def clear(self) -> None:
        """Forget every reference without touching the referenced files."""
        self._images.clear()
