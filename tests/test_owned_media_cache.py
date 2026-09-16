"""Offline ownership and lifecycle tests for retained image copies."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


package = types.ModuleType("owned_cache_test_package")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    package.__name__ + ".owned_media_cache",
    Path(package.__path__[0]) / "owned_media_cache.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
OwnedMediaCache = module.OwnedMediaCache


class OwnedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 0
        self.cache = OwnedMediaCache(self.root / "owned", clock=lambda: self.now)
        self.addCleanup(self.cache.close)
        self.source = self.root / "source.png"
        self.source.write_bytes(b"synthetic-image")

    def add(self, message_id=1, ttl=60, umo="a"):
        self.cache.put(umo, message_id, [str(self.source)], ttl)
        return Path(self.cache.select(umo, [message_id], 1)[0].image_url)

    def test_core_removes_source_but_next_turn_keeps_copy(self):
        owned = self.add()
        self.source.unlink()
        self.now = 51
        self.assertEqual(self.cache.select("a", [1], 4)[0].image_url, str(owned))
        self.assertEqual(owned.read_bytes(), b"synthetic-image")

    def test_ttl_deletes_owned_only(self):
        owned = self.add()
        self.now = 60
        self.assertEqual(self.cache.select("a", [1], 4), [])
        self.assertFalse(owned.exists())
        self.assertTrue(self.source.exists())

    def test_shared_content_lives_until_last_reference_expires(self):
        owned = self.add(ttl=20)
        self.assertEqual(self.add(2, ttl=50, umo="b"), owned)
        self.now = 20
        self.assertEqual(self.cache.select("a", [1], 4), [])
        self.assertTrue(owned.exists())
        self.now = 50
        self.cache.select("b", [2], 4)
        self.assertFalse(owned.exists())

    def test_capacity_eviction_collects_only_unreferenced_files(self):
        self.cache.MAX_IMAGES_PER_UMO = 1
        old = self.add()
        self.source.write_bytes(b"different")
        new = self.add(2)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(self.source.exists())

    def test_same_message_replacement_and_zero_ttl(self):
        old = self.add()
        self.assertEqual(self.add(), old)
        self.cache.put("a", 1, [], 0)
        self.assertFalse(old.exists())

    def test_clear_preserves_source_and_unrelated_files(self):
        owned = self.add()
        unrelated = self.cache.cache_dir / "user-file.txt"
        unrelated.write_text("keep", encoding="utf-8")
        self.cache.clear()
        self.assertFalse(owned.exists())
        self.assertTrue(unrelated.exists())
        self.assertTrue(self.source.exists())

    def test_reload_cleans_recognized_orphan_but_not_foreign_file(self):
        self.cache.close()
        orphan = self.root / "owned" / ("gcl-image-" + "a" * 64 + ".bin")
        orphan.write_bytes(b"orphan")
        foreign = orphan.parent / "foreign.bin"
        foreign.write_bytes(b"keep")
        cache = OwnedMediaCache(orphan.parent)
        self.addCleanup(cache.close)
        self.assertFalse(orphan.exists())
        self.assertTrue(foreign.exists())

    def test_overlapping_owner_cannot_delete_live_copy(self):
        owned = self.add()
        with self.assertRaises(RuntimeError):
            OwnedMediaCache(self.cache.cache_dir)
        self.assertTrue(owned.exists())

    def test_missing_and_oversized_files_never_fall_back(self):
        self.cache.MAX_FILE_BYTES = 2
        self.cache.put("a", 1, [str(self.source), str(self.root / "missing")])
        self.assertEqual(self.cache.select("a", [1], 4), [])
        self.assertTrue(self.source.exists())

    def test_partial_copy_failure_removes_temporary_file(self):
        with patch.object(Path, "replace", side_effect=OSError("private payload")):
            with self.assertLogs(module.logger, level="WARNING") as logs:
                self.cache.put("a", 1, [str(self.source)])
        self.assertNotIn("private payload", " ".join(logs.output))
        self.assertEqual(self.cache.select("a", [1], 4), [])
        self.assertEqual(list(self.cache.cache_dir.glob("gcl-*")), [])

    def test_invalid_input_does_not_mutate_existing_entries(self):
        owned = self.add()
        with self.assertRaises(ValueError):
            self.cache.put("a", 1, [str(self.source)], float("nan"))
        self.assertTrue(owned.exists())
        self.assertEqual(len(self.cache.select("a", [1], 1)), 1)

    def test_close_releases_lock_and_rejects_new_put(self):
        self.add()
        self.cache.close()
        with self.assertRaises(RuntimeError):
            self.add()
        other = OwnedMediaCache(self.cache.cache_dir)
        other.close()

    def test_startup_failure_releases_lock_with_traceback_alive(self):
        self.cache.close()
        captured = None
        with patch.object(Path, "iterdir", side_effect=OSError("scan failed")):
            try:
                OwnedMediaCache(self.cache.cache_dir)
            except OSError as exc:
                captured = exc
        self.assertIsNotNone(captured)
        self.assertIsNotNone(captured.__traceback__)
        other = OwnedMediaCache(self.cache.cache_dir)
        other.close()


if __name__ == "__main__":
    unittest.main()
