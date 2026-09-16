"""Strict presets and exact-session configuration. Numeric -1 means inherit."""

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

LEGACY_DECISION_PROMPT = (
    "你是群聊读空气助手，只判断此刻是否适合参与，不回答群消息。"
    "以本轮触发消息和待处理消息为重点，结合当前窗口、上一窗口摘要及少量原文理解承接关系。"
    "历史已回答的问题不是新的请求，不要因为旧话题仍在上下文中就再次参与。"
    "有人明确提问、请求帮助、提到AI/LLM相关话题时, 延续与你的对话，或你能提供具体帮助时，可以回应；"
    "他人之间的对话、通知、重复内容或已经解决的问题，可以选择静默."
)
DEFAULT_DECISION_PROMPT = (
    "你是群聊读空气评分助手，只评估此刻参与当前对话的语义适宜程度，不回答群消息。"
    "以本轮触发消息和待处理消息为重点，结合当前窗口、上一窗口摘要及少量原文理解承接关系。"
    "历史已回答的问题不是新的请求，不要因为旧话题仍在上下文中就再次参与。"
    "有人明确提问、请求帮助、提到AI/LLM相关话题、延续与你的对话，或你能提供具体帮助时，提高基础分；"
    "他人之间的对话、通知、重复内容或已经解决的问题，降低基础分。"
    "不要计算消息频率、夜间时段、最近成功回复对象奖励或引用链奖励，这些由程序另行加减，避免重复计分。"
    "输出JSON对象，score为0到1的基础分，reason为最多一句简短理由。"
)

SCORE_LIMITS = {
    "score_threshold": (0, 1, False),
    "score_quote_bonus": (0, 1, False),
    "score_quote_decay": (0, 1, False),
    "score_recent_bonus": (0, 1, False),
    "score_recent_seconds": (1, 3600, False),
    "score_high_penalty": (0, 1, False),
    "score_high_count": (2, 1000, True),
    "score_low_bonus": (0, 1, False),
    "score_low_count": (0, 999, True),
    "score_frequency_seconds": (1, 3600, False),
    "score_night_bonus": (0, 1, False),
}
SCORE_STRINGS = {"score_timezone", "score_night_start", "score_night_end"}

PRESETS = {
    "small": dict(
        idle_minutes=20,
        merge_wait_seconds=1,
        decision_cooldown_seconds=8,
        reply_cooldown_seconds=15,
        history_max_messages=60,
        context_max_chars=16000,
    ),
    "balanced": dict(
        idle_minutes=10,
        merge_wait_seconds=1.5,
        decision_cooldown_seconds=15,
        reply_cooldown_seconds=30,
        history_max_messages=80,
        context_max_chars=20000,
    ),
    "busy": dict(
        idle_minutes=5,
        merge_wait_seconds=3,
        decision_cooldown_seconds=30,
        reply_cooldown_seconds=60,
        history_max_messages=100,
        context_max_chars=24000,
    ),
}
LIMITS = {
    **SCORE_LIMITS,
    "max_context_images": (0, 8, True),
    "image_retention_minutes": (1, 1440, False),
    "max_merge_wait_seconds": (0, 30, False),
    "idle_minutes": (1, 1440, False),
    "merge_wait_seconds": (0, 10, False),
    "decision_cooldown_seconds": (0, 3600, False),
    "reply_cooldown_seconds": (0, 3600, False),
    "history_max_messages": (1, 200, True),
    "context_max_chars": (2000, 64000, True),
    "bridge_messages": (0, 8, True),
    "summary_min_messages": (1, 200, True),
    "summary_max_chars": (2000, 64000, True),
    "summary_output_chars": (100, 4000, True),
    "summary_timeout": (1, 180, False),
    "summary_retries": (0, 2, True),
    "decision_max_chars": (512, 16000, True),
    "decision_timeout": (1, 120, False),
    "history_tool_limit": (1, 200, True),
    "history_tool_max_chars": (512, 32000, True),
    "stale_message_seconds": (10, 86400, False),
}
ALIASES = {
    "current_max_chars": "context_max_chars",
    "summary_max_input_chars": "summary_max_chars",
    "summary_input_chars": "summary_max_chars",
    "decision_timeout_seconds": "decision_timeout",
    "summary_timeout_seconds": "summary_timeout",
}
GROUP_FIELDS = {
    *SCORE_LIMITS,
    *SCORE_STRINGS,
    "scoring",
    "decision_use_persona",
    "decision_prompt",
    "image_input_enabled",
    "max_context_images",
    "image_retention_minutes",
    "idle_minutes",
    "merge_wait_seconds",
    "decision_cooldown_seconds",
    "reply_cooldown_seconds",
    "history_max_messages",
    "context_max_chars",
    "bridge_messages",
    "auto_reply",
    "reply_mode",
    "summary_enabled",
}


def _overrides(raw, group=False):
    if not isinstance(raw, Mapping):
        raise ValueError("Overrides must be an object")
    result = {}
    for name, value in raw.items():
        key = ALIASES.get(name, name)
        if group and key in {"group_id", "preset", "__template_key"}:
            continue
        if group and key not in GROUP_FIELDS:
            raise ValueError("Unsupported group override")
        if key == "scoring":
            if not isinstance(value, Mapping) or any(
                k not in {*SCORE_LIMITS, *SCORE_STRINGS} for k in value
            ):
                raise ValueError("scoring must contain only score fields")
            result.update(_overrides(value, group))
        elif key in SCORE_STRINGS:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be text")
            value = value.strip()
            if not value:
                continue
            if key == "score_timezone":
                try:
                    ZoneInfo(value)
                except (ZoneInfoNotFoundError, ValueError) as exc:
                    raise ValueError(
                        "Invalid score_timezone or missing IANA tzdata"
                    ) from exc
            elif not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                raise ValueError(f"{key} must use HH:MM")
            result[key] = value
        elif key in LIMITS:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
            if value == -1:
                continue
            low, high, integer = LIMITS[key]
            if not low <= value <= high or integer and int(value) != value:
                raise ValueError(f"{key} outside allowed range")
            result[key] = int(value) if integer else float(value)
        elif key == "decision_prompt":
            if not isinstance(value, str) or len(value) > 8000:
                raise ValueError(
                    "decision_prompt must be text of at most 8000 characters"
                )
            value = value.strip()
            if value == LEGACY_DECISION_PROMPT:
                value = DEFAULT_DECISION_PROMPT
            if value:
                result[key] = value
            elif not group:
                result[key] = DEFAULT_DECISION_PROMPT
        elif key == "decision_log_reasoning":
            if type(value) is not bool:
                raise ValueError("decision_log_reasoning must be boolean")
            result[key] = value
        elif key in {
            "auto_reply",
            "summary_enabled",
            "image_input_enabled",
            "decision_use_persona",
        }:
            if value == "inherit":
                continue
            if (
                group
                and key in {"image_input_enabled", "decision_use_persona"}
                and value in ("enabled", "disabled")
            ):
                value = value == "enabled"
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            result[key] = value
        elif key == "reply_mode":
            if value not in ("inherit", "smart", "mention_only"):
                raise ValueError("Invalid reply_mode")
            if value != "inherit":
                result["auto_reply"] = value == "smart"
        elif key in {"decision_provider_id", "summary_provider_id"}:
            if not isinstance(value, str):
                raise ValueError("Provider id must be text")
            result[key] = value.strip()
        else:
            raise ValueError("Unknown advanced field")
    if (
        "auto_reply" in raw
        and raw.get("reply_mode", "inherit") != "inherit"
        and raw["auto_reply"] != (raw["reply_mode"] == "smart")
    ):
        raise ValueError("Conflicting reply settings")
    return result


@dataclass(frozen=True)
class Settings:
    score_threshold: float = 0.7
    score_quote_bonus: float = 1.0
    score_quote_decay: float = 0.5
    score_recent_bonus: float = 0.2
    score_recent_seconds: float = 120.0
    score_high_penalty: float = 0.1
    score_high_count: int = 6
    score_low_bonus: float = 0.2
    score_low_count: int = 1
    score_frequency_seconds: float = 60.0
    score_night_bonus: float = 0.1
    score_timezone: str = "Asia/Hong_Kong"
    score_night_start: str = "00:00"
    score_night_end: str = "07:00"
    enabled: bool = True
    image_input_enabled: bool = True
    max_context_images: int = 4
    image_retention_minutes: float = 20
    group_ids: tuple[str, ...] = ()
    group_discovery_enabled: bool = True
    selected_groups: tuple[str, ...] = ()
    preset: str = "balanced"
    idle_minutes: float = 10
    auto_reply: bool = True
    decision_provider_id: str = ""
    decision_prompt: str = DEFAULT_DECISION_PROMPT
    decision_use_persona: bool = True
    decision_log_reasoning: bool = False
    summary_provider_id: str = ""
    context_max_chars: int = 20000
    merge_wait_seconds: float = 1.5
    max_merge_wait_seconds: float = 8
    decision_cooldown_seconds: float = 15
    reply_cooldown_seconds: float = 30
    history_max_messages: int = 80
    bridge_messages: int = 4
    summary_enabled: bool = True
    summary_min_messages: int = 4
    summary_max_chars: int = 12000
    summary_output_chars: int = 500
    summary_timeout: float = 45
    summary_retries: int = 1
    decision_max_chars: int = 4000
    decision_timeout: float = 20
    history_tool_limit: int = 20
    history_tool_max_chars: int = 8000
    stale_message_seconds: float = 300
    _global: dict = field(default_factory=dict, repr=False, compare=False)
    _groups: tuple = field(default_factory=tuple, repr=False, compare=False)

    def __post_init__(self):
        if self.score_low_count >= self.score_high_count:
            raise ValueError("score_low_count must be less than score_high_count")
        if not self.enabled:
            return
        try:
            ZoneInfo(self.score_timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Invalid score_timezone or missing IANA tzdata") from exc

    @property
    def idle_seconds(self):
        return self.idle_minutes * 60

    @property
    def summary_attempts(self):
        return self.summary_retries + 1

    def effective(self, umo, group_id):
        selected = next((e for e in self._groups if e[0] == umo), None)
        if selected is None:
            selected = next((e for e in self._groups if e[0] == group_id), None)
        if selected is None:
            return self
        _, preset, overrides = selected
        values = dict(PRESETS[self.preset if preset == "inherit" else preset])
        values.update(self._global)
        values.update(overrides)
        return replace(self, **values)

    @classmethod
    def from_mapping(cls, config):
        if not isinstance(config, Mapping):
            raise ValueError("Configuration must be an object")
        allowed = (
            {
                "enabled",
                "group_ids",
                "group_discovery_enabled",
                "selected_groups",
                "preset",
                "advanced",
                "scoring",
                "group_overrides",
                "auto_reply",
                "reply_mode",
                "decision_provider_id",
                "decision_prompt",
                "summary_provider_id",
                "summary_enabled",
                "summary_attempts",
            }
            | LIMITS.keys()
            | ALIASES.keys()
            | SCORE_STRINGS
        )
        if any(key not in allowed for key in config):
            raise ValueError("Unknown configuration field")
        enabled = config.get("enabled", True)
        groups = config.get("group_ids", [])
        discovery = config.get("group_discovery_enabled", True)
        selected = config.get("selected_groups", [])
        preset = config.get("preset", "balanced")
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(discovery) is not bool:
            raise ValueError("group_discovery_enabled must be boolean")
        if not isinstance(selected, list) or any(
            not isinstance(g, str) or not g.strip() or g.strip() == "*"
            for g in selected
        ):
            raise ValueError("selected_groups must be an exact nonempty-string list")
        for group in selected:
            parts = group.strip().split(":", 2)
            if (
                len(parts) != 3
                or not parts[0]
                or parts[1] != "GroupMessage"
                or not parts[2]
            ):
                raise ValueError(
                    "selected_groups entries must be complete group UMO identifiers"
                )
        if not isinstance(groups, (list, tuple)) or any(
            not isinstance(g, str) or not g.strip() or g.strip() == "*" for g in groups
        ):
            raise ValueError(
                "group_ids must be an exact nonempty-string list; no wildcard"
            )
        if not isinstance(preset, str) or preset not in PRESETS:
            raise ValueError("Invalid preset")
        values = _overrides(config.get("advanced", {}))
        values.update(_overrides({"scoring": config.get("scoring", {})}))
        legacy = {
            k: v
            for k, v in config.items()
            if k in LIMITS
            or k in SCORE_STRINGS
            or k in ALIASES
            or k
            in {
                "auto_reply",
                "reply_mode",
                "decision_provider_id",
                "decision_prompt",
                "summary_provider_id",
                "summary_enabled",
            }
        }
        if "summary_attempts" in config:
            n = config["summary_attempts"]
            if type(n) is not int or not 1 <= n <= 3:
                raise ValueError("summary_attempts must be 1 to 3")
            legacy["summary_retries"] = n - 1
        values.update(_overrides(legacy))
        raw = config.get("group_overrides", [])
        if not isinstance(raw, list):
            raise ValueError("group_overrides must be a template list")
        entries, seen = [], set()
        for e in raw:
            if not isinstance(e, Mapping):
                raise ValueError("Group override must be an object")
            target, profile = e.get("group_id"), e.get("preset", "inherit")
            if (
                not isinstance(target, str)
                or not target.strip()
                or target.strip() == "*"
                or target.strip() in seen
            ):
                raise ValueError("Group override requires unique exact identifier")
            if profile not in ("inherit", *PRESETS):
                raise ValueError("Invalid group preset")
            seen.add(target.strip())
            entries.append((target.strip(), profile, _overrides(e, True)))
        resolved = dict(PRESETS[preset])
        resolved.update(values)
        settings = cls(
            enabled=enabled,
            group_ids=tuple(dict.fromkeys(g.strip() for g in [*groups, *selected])),
            group_discovery_enabled=discovery,
            selected_groups=tuple(dict.fromkeys(g.strip() for g in selected)),
            preset=preset,
            _global=values,
            _groups=tuple(entries),
            **resolved,
        )
        for target, _, _ in entries:
            settings.effective(target, target)
        return settings
