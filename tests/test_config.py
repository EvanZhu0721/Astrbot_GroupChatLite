import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gcl_config_test", ROOT / "config.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Settings = module.Settings


def defaults(items):
    return {
        k: defaults(v.get("items", {})) if v["type"] == "object" else v.get("default")
        for k, v in items.items()
    }


class ConfigTests(unittest.TestCase):
    def test_decision_prompt_default_and_schema(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        field = schema["advanced"]["items"]["decision_prompt"]
        self.assertEqual(field["type"], "text")
        self.assertEqual(field["default"], module.DEFAULT_DECISION_PROMPT)
        for config in ({}, defaults(schema), {"advanced": {"decision_prompt": " \n"}}):
            self.assertEqual(
                Settings.from_mapping(config).decision_prompt,
                module.DEFAULT_DECISION_PROMPT,
            )

    def test_decision_prompt_group_inheritance_and_exact_override(self):
        settings = Settings.from_mapping(
            {
                "advanced": {"decision_prompt": " 全局规则 "},
                "group_overrides": [
                    {"group_id": "room", "decision_prompt": " \n"},
                    {
                        "group_id": "tg:GroupMessage:room#2",
                        "decision_prompt": " 本话题规则 ",
                    },
                ],
            }
        )
        self.assertEqual(settings.decision_prompt, "全局规则")
        self.assertEqual(
            settings.effective("tg:GroupMessage:room", "room").decision_prompt,
            "全局规则",
        )
        self.assertEqual(
            settings.effective("tg:GroupMessage:room#2", "room").decision_prompt,
            "本话题规则",
        )

    def test_decision_prompt_strict_type_and_length(self):
        for value in (None, False, 1, [], "x" * 8001):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    Settings.from_mapping({"advanced": {"decision_prompt": value}})
                with self.assertRaises(ValueError):
                    Settings.from_mapping(
                        {
                            "group_overrides": [
                                {"group_id": "room", "decision_prompt": value}
                            ]
                        }
                    )
        self.assertEqual(
            len(
                Settings.from_mapping(
                    {"advanced": {"decision_prompt": "x" * 8000}}
                ).decision_prompt
            ),
            8000,
        )

    def test_image_defaults_and_group_overrides(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        settings = Settings.from_mapping(defaults(schema))
        self.assertTrue(settings.image_input_enabled)
        self.assertEqual(settings.max_context_images, 4)
        self.assertEqual(settings.image_retention_minutes, 20)
        settings = Settings.from_mapping(
            {
                "group_overrides": [
                    {
                        "group_id": "a",
                        "image_input_enabled": "disabled",
                        "max_context_images": 0,
                        "image_retention_minutes": 1,
                    },
                    {
                        "group_id": "b",
                        "image_input_enabled": "inherit",
                        "max_context_images": -1,
                    },
                ]
            }
        )
        self.assertFalse(settings.effective("a", "a").image_input_enabled)
        self.assertEqual(settings.effective("a", "a").max_context_images, 0)
        self.assertEqual(settings.effective("a", "a").image_retention_minutes, 1)
        self.assertTrue(settings.effective("b", "b").image_input_enabled)
        self.assertEqual(settings.effective("b", "b").max_context_images, 4)

    def test_image_options_reject_invalid_ranges(self):
        for key, value in (
            ("image_input_enabled", "false"),
            ("max_context_images", 9),
            ("max_context_images", 1.5),
            ("image_retention_minutes", 0),
            ("image_retention_minutes", 1441),
            ("image_retention_minutes", float("inf")),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                Settings.from_mapping({"advanced": {key: value}})

    def test_discovery_defaults_do_not_select_or_enable_any_group(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        for config in ({}, defaults(schema)):
            settings = Settings.from_mapping(config)
            self.assertTrue(settings.group_discovery_enabled)
            self.assertEqual(settings.selected_groups, ())
            self.assertEqual(settings.group_ids, ())
        self.assertEqual(schema["selected_groups"]["options"], [])
        self.assertEqual(schema["selected_groups"]["labels"], [])

    def test_selected_groups_union_is_deduplicated_and_selection_preserved(self):
        umo = "telegram-instance:GroupMessage:-123#5"
        second = "another-instance:GroupMessage:-123#5"
        settings = Settings.from_mapping(
            {
                "group_ids": ["-123", umo],
                "selected_groups": [umo, " " + second + " ", umo],
                "group_discovery_enabled": False,
                "group_overrides": [{"group_id": umo, "idle_minutes": 3}],
            }
        )
        self.assertEqual(settings.group_ids, ("-123", umo, second))
        self.assertEqual(settings.selected_groups, (umo, second))
        effective = settings.effective(umo, "-123#5")
        self.assertEqual(effective.idle_minutes, 3)
        self.assertEqual(effective.selected_groups, (umo, second))
        self.assertFalse(effective.group_discovery_enabled)

    def test_discovery_and_selected_groups_require_strict_types(self):
        for value in ("false", 0, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings.from_mapping({"group_discovery_enabled": value})
        for value in (
            "a:GroupMessage:b",
            (),
            None,
            [1],
            [""],
            ["*"],
            ["-123"],
            ["a:FriendMessage:b"],
            [":GroupMessage:b"],
            ["a:GroupMessage:"],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings.from_mapping({"selected_groups": value})

    def test_selected_group_umo_preserves_entire_session_suffix(self):
        umo = "instance:GroupMessage:group:topic"
        settings = Settings.from_mapping({"selected_groups": [umo], "enabled": False})
        self.assertEqual(settings.selected_groups, (umo,))
        self.assertFalse(settings.enabled)

    def test_decision_reasoning_log_defaults_off(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertFalse(Settings.from_mapping({}).decision_log_reasoning)
        self.assertFalse(Settings.from_mapping(defaults(schema)).decision_log_reasoning)

    def test_decision_reasoning_log_global_setting_survives_group_profile(self):
        for enabled in (False, True):
            settings = Settings.from_mapping(
                {
                    "advanced": {"decision_log_reasoning": enabled},
                    "group_overrides": [{"group_id": "a", "preset": "busy"}],
                }
            )
            self.assertIs(
                settings.effective("umo", "a").decision_log_reasoning, enabled
            )

    def test_decision_reasoning_log_requires_strict_global_bool(self):
        for value in ("true", "false", "inherit", 0, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings.from_mapping({"advanced": {"decision_log_reasoning": value}})
        with self.assertRaises(ValueError):
            Settings.from_mapping(
                {"group_overrides": [{"group_id": "a", "decision_log_reasoning": True}]}
            )

    def test_schema_defaults_preserve_all_profiles(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        config = defaults(schema)
        for preset in module.PRESETS:
            config["preset"] = preset
            settings = Settings.from_mapping(config)
            self.assertEqual(settings.group_ids, ())
            for key, value in module.PRESETS[preset].items():
                self.assertEqual(getattr(settings, key), value)

    def test_exact_umo_overrides_group_and_preserves_allowlist(self):
        settings = Settings.from_mapping(
            {
                "group_ids": ["-123"],
                "preset": "small",
                "group_overrides": [
                    {"group_id": "-123", "preset": "busy", "context_max_chars": 3000},
                    {
                        "group_id": "tg:group:-123:topic:4",
                        "preset": "balanced",
                        "context_max_chars": 4000,
                    },
                ],
            }
        )
        self.assertEqual(settings.effective("other", "-123").context_max_chars, 3000)
        self.assertEqual(
            settings.effective("tg:group:-123:topic:4", "-123").context_max_chars, 4000
        )
        self.assertEqual(
            settings.effective("tg:group:-123:topic:4", "-123").idle_minutes, 10
        )
        self.assertEqual(settings.effective("other", "-999").group_ids, ("-123",))

    def test_explicit_global_and_group_precedence(self):
        s = Settings.from_mapping(
            {
                "preset": "small",
                "advanced": {"idle_minutes": 7, "bridge_messages": 0},
                "group_overrides": [
                    {"group_id": "a", "preset": "busy", "idle_minutes": -1},
                    {"group_id": "b", "preset": "busy", "idle_minutes": 3},
                ],
            }
        )
        self.assertEqual(s.effective("a", "a").idle_minutes, 7)
        self.assertEqual(s.effective("b", "b").idle_minutes, 3)
        self.assertEqual(s.effective("a", "a").bridge_messages, 0)

    def test_zero_is_explicit_not_inherit(self):
        s = Settings.from_mapping(
            {
                "advanced": {
                    "merge_wait_seconds": 0,
                    "reply_cooldown_seconds": 0,
                    "summary_retries": 0,
                    "reply_mode": "mention_only",
                }
            }
        )
        self.assertEqual(s.merge_wait_seconds, 0)
        self.assertEqual(s.reply_cooldown_seconds, 0)
        self.assertEqual(s.summary_attempts, 1)
        self.assertFalse(s.auto_reply)

    def test_invalid_configuration_never_silently_broadens(self):
        for config in (
            {"enabled": "false"},
            {"group_ids": "*"},
            {"group_ids": ["*"]},
            {"preset": "unknown"},
            {"advanced": {"idle_minutes": float("nan")}},
            {"advanced": {"context_max_chars": 1999}},
            {"advanced": {"summary_retries": 3}},
            {"enabeld": False},
            {"group_overrides": [{"group_id": "a"}, {"group_id": "a"}]},
            {"auto_reply": False, "reply_mode": "smart"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                Settings.from_mapping(config)

    def test_template_editor_defaults_are_valid_after_target_supplied(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        rule = defaults(schema["group_overrides"]["templates"]["group"]["items"])
        rule.update(group_id="a", __template_key="group")
        config = defaults(schema)
        config["group_overrides"] = [rule]
        settings = Settings.from_mapping(config)
        self.assertEqual(settings.effective("a", "a").context_max_chars, 20000)
        self.assertEqual(settings.group_ids, ())

    def test_legacy_attribute_compatibility(self):
        s = Settings.from_mapping(
            {
                "auto_reply": False,
                "idle_minutes": 3,
                "summary_attempts": 3,
                "summary_max_input_chars": 5000,
            }
        )
        self.assertFalse(s.auto_reply)
        self.assertEqual(s.idle_seconds, 180)
        self.assertEqual(s.summary_attempts, 3)
        self.assertEqual(s.summary_max_chars, 5000)


if __name__ == "__main__":
    unittest.main()
