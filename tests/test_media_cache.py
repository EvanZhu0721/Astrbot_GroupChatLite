"""Pure offline media-reference retention and scope tests."""

from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "lite_media_cache", Path(__file__).resolve().parents[1] / "media_cache.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
MediaCache, MediaRef = module.MediaCache, module.MediaRef


class MediaCacheTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.cache = MediaCache(clock=lambda: self.now)
        self.umo = "bot:GroupMessage:-100#1"

    def test_selection_uses_supplied_message_order_and_keeps_image_order(self):
        self.cache.put(self.umo, 20, ["new-a", "new-b"], 60)
        self.cache.put(self.umo, 10, ["old-a", "old-b"], 60)
        self.assertEqual(
            self.cache.select(self.umo, [10, 20], 3),
            [MediaRef(10, "old-b"), MediaRef(20, "new-a"), MediaRef(20, "new-b")],
        )
        self.assertEqual(
            self.cache.select(self.umo, [20, 10], 2),
            [MediaRef(10, "old-a"), MediaRef(10, "old-b")],
        )

    def test_duplicate_path_belongs_to_latest_requested_message(self):
        self.cache.put(self.umo, 1, ["shared", "shared", "older"], 60)
        self.cache.put(self.umo, 2, ["shared", "newer"], 60)
        self.assertEqual(
            self.cache.select(self.umo, [1, 2], 8),
            [MediaRef(1, "older"), MediaRef(2, "shared"), MediaRef(2, "newer")],
        )
        self.assertEqual(
            self.cache.select(self.umo, [2, 1], 2),
            [MediaRef(1, "shared"), MediaRef(1, "older")],
        )

    def test_scope_and_message_ids_are_both_required_for_lookup(self):
        other = "otherbot:GroupMessage:-100#2"
        self.cache.put(self.umo, 1, ["same-path"], 60)
        self.cache.put(other, 1, ["same-path", "other-only"], 60)
        self.assertEqual(
            self.cache.select(self.umo, [1], 8), [MediaRef(1, "same-path")]
        )
        self.assertEqual(self.cache.select(self.umo, [2], 8), [])
        self.assertEqual(self.cache.select("third:GroupMessage:-100", [1], 8), [])
        self.assertEqual(len(self.cache.select(other, [1], 8)), 2)

    def test_expiration_boundary_and_reads_do_not_extend_ttl(self):
        self.cache.put(self.umo, 1, ["image"], 10)
        self.now = 109.999
        self.assertEqual(self.cache.select(self.umo, [1], 1), [MediaRef(1, "image")])
        self.now = 110
        self.assertEqual(self.cache.select(self.umo, [1], 1), [])
        self.assertFalse(self.cache._images)

    def test_zero_ttl_removes_message_and_never_reads_image_iterable(self):
        self.cache.put(self.umo, 1, ["old"], 60)
        self.cache.put(self.umo, 2, ["keep"], 60)

        def images():
            raise AssertionError("Zero TTL must not read references")
            yield "never"

        self.cache.put(self.umo, 1, images(), 0)
        self.assertEqual(self.cache.select(self.umo, [1, 2], 8), [MediaRef(2, "keep")])
        self.cache.put(self.umo, 3, ["new"], 0)
        self.assertEqual(self.cache.select(self.umo, [3], 1), [])

    def test_put_replaces_message_and_empty_images_remove_it(self):
        self.cache.put(self.umo, 1, ["old-a", "old-b"], 60)
        self.cache.put(self.umo, 1, ["new"], 60)
        self.assertEqual(self.cache.select(self.umo, [1], 8), [MediaRef(1, "new")])
        self.cache.put(self.umo, 1, [], 60)
        self.assertEqual(self.cache.select(self.umo, [1], 8), [])

    def test_per_message_cap_bounds_iterable_consumption(self):
        def references():
            for index in range(8):
                yield f"image-{index}"
            raise AssertionError("Should inspect only eight input references")

        self.cache.put(self.umo, 1, references(), 60)
        self.assertEqual(len(self.cache.select(self.umo, [1], 1000)), 8)

    def test_per_scope_capacity_evicts_only_oldest_excess_images(self):
        self.cache.put(self.umo, 1, [f"old-{i}" for i in range(8)], 60)
        self.cache.put(self.umo, 2, [f"middle-{i}" for i in range(8)], 60)
        self.cache.put(self.umo, 3, ["new"], 60)
        remaining = self.cache.select(self.umo, [1, 2, 3], 1000)
        self.assertEqual(len(remaining), 16)
        self.assertEqual(remaining[0], MediaRef(1, "old-1"))
        self.assertEqual(remaining[-1], MediaRef(3, "new"))

    def test_global_capacity_counts_images_and_evicts_oldest_across_scopes(self):
        for group in range(8):
            for message in (1, 2):
                self.cache.put(
                    f"scope-{group}",
                    message,
                    [f"image-{message}-{i}" for i in range(8)],
                    60,
                )
        self.assertEqual(len(self.cache._images), 128)
        self.cache.put("new-scope", 1, ["new"], 60)
        self.assertEqual(len(self.cache._images), 128)
        oldest = self.cache.select("scope-0", [1, 2], 1000)
        self.assertEqual(len(oldest), 15)
        self.assertEqual(oldest[0], MediaRef(1, "image-1-1"))

    def test_expired_entries_are_removed_before_capacity_eviction(self):
        self.cache.put(self.umo, 1, [f"keep-{i}" for i in range(8)], 100)
        self.cache.put(self.umo, 2, [f"expire-{i}" for i in range(8)], 1)
        self.now += 2
        self.cache.put(self.umo, 3, [f"new-{i}" for i in range(8)], 60)
        selected = self.cache.select(self.umo, [1, 2, 3], 1000)
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0], MediaRef(1, "keep-0"))

    def test_reference_bounds_skip_invalid_values_without_touching_paths(self):
        boundary = "x" * 8192
        self.cache.put(self.umo, 1, [None, "", " ", 123, "x" * 8193, boundary], 60)
        self.assertEqual(self.cache.select(self.umo, [1], 8), [MediaRef(1, boundary)])

    def test_invalid_identity_or_ttl_rejects_without_overwriting_existing_refs(self):
        self.cache.put(self.umo, 1, ["keep"], 60)
        for invalid in ("", " ", "x" * 1025, None):
            with self.subTest(umo=repr(invalid)[:30]), self.assertRaises(ValueError):
                self.cache.put(invalid, 1, ["bad"], 60)
        for invalid in (0, -1, True, "1", 1.0):
            with self.subTest(message_id=invalid), self.assertRaises(ValueError):
                self.cache.put(self.umo, invalid, ["bad"], 60)
        for invalid in (-1, float("nan"), float("inf"), None, True):
            with self.subTest(ttl=invalid), self.assertRaises(ValueError):
                self.cache.put(self.umo, 1, ["bad"], invalid)
        self.assertEqual(self.cache.select(self.umo, [1], 8), [MediaRef(1, "keep")])

    def test_zero_limit_clear_and_frozen_references(self):
        self.cache.put(self.umo, 1, ["image"], 60)
        self.assertEqual(self.cache.select(self.umo, [1], 0), [])
        reference = self.cache.select(self.umo, [1], 1)[0]
        with self.assertRaises(FrozenInstanceError):
            reference.image_url = "changed"
        self.cache.clear()
        self.assertEqual(self.cache.select(self.umo, [1], 1), [])

    def test_references_are_opaque_and_do_not_use_filesystem_or_network(self):
        fail = AssertionError("Media cache must not perform I/O")
        with (
            patch("builtins.open", side_effect=fail),
            patch("os.stat", side_effect=fail),
            patch("os.remove", side_effect=fail),
            patch("shutil.copyfile", side_effect=fail),
            patch.object(socket.socket, "connect", side_effect=fail),
        ):
            references = ["C:/missing/image.png", "https://invalid.example/image.png"]
            self.cache.put(self.umo, 1, references, 60)
            self.assertEqual(
                self.cache.select(self.umo, [1], 8),
                [MediaRef(1, value) for value in references],
            )
            self.cache.clear()


if __name__ == "__main__":
    unittest.main()
