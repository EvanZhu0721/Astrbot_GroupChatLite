"""Event-independent image copies in an exclusively owned plugin directory."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
import tempfile
import time

from .media_cache import MediaCache


logger = logging.getLogger(__name__)
_OWNED_NAME = re.compile(r"gcl-image-[a-f0-9]{64}\.bin|gcl-copy-[\w-]+\.tmp")


class OwnedMediaCache(MediaCache):
    """Copy files before event cleanup; one live owner per cache directory.

    Limits and TTL apply to references. Identical bytes share one owned file.
    The directory is private to this class, never AstrBot's shared temp directory.
    """

    MAX_FILE_BYTES = 64 * 1024 * 1024

    def __init__(self, cache_dir: Path, clock=time.monotonic):
        super().__init__(clock)
        directory = Path(cache_dir)
        if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
            raise ValueError("Media cache directory must not be a symbolic link")
        directory.mkdir(parents=True, exist_ok=True)
        self.cache_dir = directory.resolve()
        lock_path = self.cache_dir / ".owner.lock"
        if lock_path.is_symlink():
            raise ValueError("Media cache lock must not be a symbolic link")
        self._lock = lock_path.open("a+b")
        try:
            self._lock.seek(0, 2)
            if not self._lock.tell():
                self._lock.write(b"0")
                self._lock.flush()
            self._lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise RuntimeError("Image cache already has an active owner") from None
        self._closed = False
        self._owned: set[Path] = set()
        self._staging = False
        try:
            for path in self.cache_dir.iterdir():
                if _OWNED_NAME.fullmatch(path.name) and not path.is_dir():
                    self._owned.add(path)
            self._collect()
        except BaseException:
            self._closed = True
            self._lock.close()
            raise

    def _collect(self):
        live = {item.reference.image_url for item in self._images.values()}
        for path in tuple(self._owned):
            if str(path) in live:
                continue
            try:
                # Unlink only direct children we created or recognized at startup.
                if path.parent != self.cache_dir or not _OWNED_NAME.fullmatch(
                    path.name
                ):
                    raise ValueError("Invalid owned media path")
                path.unlink(missing_ok=True)
                self._owned.discard(path)
            except OSError as exc:
                logger.warning("Image cache cleanup failed (%s)", type(exc).__name__)

    def _copy(self, source):
        temporary = None
        try:
            path = Path(source)
            if not path.is_file() or path.stat().st_size > self.MAX_FILE_BYTES:
                return None
            digest = hashlib.sha256()
            total = 0
            with (
                path.open("rb") as incoming,
                tempfile.NamedTemporaryFile(
                    dir=self.cache_dir, prefix="gcl-copy-", suffix=".tmp", delete=False
                ) as outgoing,
            ):
                temporary = Path(outgoing.name)
                self._owned.add(temporary)
                while chunk := incoming.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.MAX_FILE_BYTES:
                        raise ValueError("Image exceeds cache byte limit")
                    digest.update(chunk)
                    outgoing.write(chunk)
            target = self.cache_dir / f"gcl-image-{digest.hexdigest()}.bin"
            # Replace rather than trusting a preexisting file or symlink.
            temporary.replace(target)
            self._owned.discard(temporary)
            self._owned.add(target)
            return str(target)
        except (OSError, ValueError) as exc:
            logger.warning("Image cache copy failed (%s)", type(exc).__name__)
            return None

    def _expire(self, now):
        super()._expire(now)
        if not self._staging:
            self._collect()

    def put(self, umo, message_id, images, ttl_seconds=1200):
        if self._closed:
            raise RuntimeError("Image cache is closed")
        # Validate using the base implementation before performing filesystem work.
        validator = MediaCache(self._clock)
        validator.put(umo, message_id, images, ttl_seconds)
        self._staging = True
        try:
            copies = []
            for item in validator._images.values():
                copied = self._copy(item.reference.image_url)
                if copied and copied not in copies:
                    copies.append(copied)
            super().put(umo, message_id, copies, ttl_seconds)
        finally:
            self._staging = False
            self._collect()

    def clear(self):
        super().clear()
        self._collect()

    def close(self):
        """Drop owned files and release the OS lock after active users finish."""
        if not self._closed:
            self.clear()
            self._closed = True
            self._lock.close()
