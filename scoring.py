"""Pure local participation scoring; callers own TG source-ID state."""

from datetime import datetime
import json
import math
from zoneinfo import ZoneInfo


def _finite(value, low, high, name):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"Invalid {name}")
    return float(value)


def parse_model_score(text):
    """Accept one JSON object, with no prose, code fences or nonfinite scores."""
    if not isinstance(text, str) or len(text) > 16000:
        return None
    try:

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate field")
                result[key] = value
            return result

        value = json.loads(text, object_pairs_hook=unique_pairs)
        if not isinstance(value, dict) or set(value) - {"score", "reason"}:
            return None
        score = _finite(value.get("score"), 0, 1, "model score")
        reason = value.get("reason", "")
        if not isinstance(reason, str) or len(reason) > 200:
            return None
        return {"score": score, "reason": reason.strip()}
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


def _minute(value):
    try:
        hour, minute = value.split(":")
        if len(hour) != 2 or len(minute) != 2:
            raise ValueError
        hour, minute = int(hour), int(minute)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError
        return hour * 60 + minute
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid night clock time") from exc


def score_decision(
    base_score,
    *,
    reply_hops,
    recent_reply_target,
    recent_message_count,
    pending_count,
    now,
    cfg,
):
    """Add independent bonuses once, then clamp the final total to [0, 1].

    reply_hops counts original Telegram reply edges: direct bot reply is one.
    None means unknown/no chain. Counts and recent-target eligibility must come
    from the caller's isolated UMO state, never from model guesses or SQL IDs.
    """
    base = _finite(base_score, 0, 1, "base score")
    for name, value in (
        ("recent count", recent_message_count),
        ("pending count", pending_count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {name}")
    if reply_hops is not None and (
        type(reply_hops) is not int or not 1 <= reply_hops <= 8
    ):
        raise ValueError("Invalid reply hop count")
    if type(recent_reply_target) is not bool:
        raise ValueError("Invalid recent reply target")
    stamp = _finite(now, -62135596800, 253402214400, "timestamp")
    weights = {
        name: _finite(getattr(cfg, name), 0, 1, name)
        for name in (
            "score_quote_bonus",
            "score_quote_decay",
            "score_recent_bonus",
            "score_high_penalty",
            "score_low_bonus",
            "score_night_bonus",
            "score_threshold",
        )
    }
    local = datetime.fromtimestamp(stamp, ZoneInfo(cfg.score_timezone))
    minute = local.hour * 60 + local.minute
    start, end = _minute(cfg.score_night_start), _minute(cfg.score_night_end)
    night = (
        (start <= minute < end)
        if start < end
        else (minute >= start or minute < end)
        if start > end
        else False
    )
    bonuses = {
        "quote": weights["score_quote_bonus"]
        * weights["score_quote_decay"] ** (reply_hops - 1)
        if reply_hops
        else 0.0,
        "recent": weights["score_recent_bonus"] if recent_reply_target else 0.0,
        "high_frequency": -weights["score_high_penalty"]
        if recent_message_count >= cfg.score_high_count
        else 0.0,
        "low_or_singleton": weights["score_low_bonus"]
        if recent_message_count <= cfg.score_low_count or pending_count == 1
        else 0.0,
        "night": weights["score_night_bonus"] if night else 0.0,
    }
    total = min(1.0, max(0.0, math.fsum([base, *bonuses.values()])))
    threshold = weights["score_threshold"]
    return {
        "base_score": base,
        "bonuses": bonuses,
        "total": total,
        "threshold": threshold,
        "should_reply": total >= threshold,
    }
