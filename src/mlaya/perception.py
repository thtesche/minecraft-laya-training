from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# view_status -> structured_content shape:
#   response.data.result.situation.{vitals,clock,position,inventory,tools,nearby,lastDeath,...}
# read_recent_events -> response.data.result.events[] (type/observedAt/summary/...)


def get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def coerce(data: Any, text: str | None) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(text, str) and text.lstrip().startswith("{"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _iter_items(container: Any):
    """Yield (item_name, count) from inventory stacks / dicts."""
    if isinstance(container, list):
        for e in container:
            if isinstance(e, dict):
                name = e.get("item_name") or e.get("item") or e.get("name")
                cnt = e.get("count") or e.get("quantity") or 1
                if name:
                    yield str(name), int(cnt)
    elif isinstance(container, dict):
        for key, value in container.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                yield str(key), int(value)


@dataclass
class GameState:
    health: float | None = None
    food: float | None = None
    saturation: float | None = None
    phase: str | None = None
    raining: bool | None = None
    pos: tuple[float, float, float] | None = None
    on_ground: bool | None = None
    in_water: bool | None = None
    in_lava: bool | None = None
    dimension: str | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)
    hostiles: list[tuple[str, int, float | None]] = field(default_factory=list)
    nearest_hostile: float | None = None
    players: list[str] = field(default_factory=list)
    last_death_cause: str | None = None
    last_death_age_s: float | None = None
    activity_owner: str | None = None
    active_action: str | None = None
    events: list[str] = field(default_factory=list)
    unread: int = 0
    survivors: list[str] = field(default_factory=list)

    def as_state(self) -> dict:
        """Structured view for logging and System-2 prompts (never sent raw to Laya)."""
        return {
            "health": self.health,
            "food": self.food,
            "phase": self.phase,
            "raining": self.raining,
            "position": self.pos,
            "on_ground": self.on_ground,
            "in_water": self.in_water,
            "dimension": self.dimension,
            "inventory": self.inventory,
            "tools": self.tools,
            "hostiles": self.hostiles,
            "players": self.players,
            "last_death": self.last_death_cause,
            "events": self.events,
        }

    def narrative(self) -> str:
        """Compact natural-language state. Must stay small: Laya truncates long states."""
        parts: list[str] = []
        if self.dimension:
            parts.append(f"in the {self.dimension}")
        if self.health is not None:
            parts.append("health is full" if self.health >= 20 else f"health is {self.health:g}/20")
        if self.food is not None:
            parts.append("hunger is full" if self.food >= 20 else f"hunger is {self.food:g}/20")
        if self.phase:
            weather = " and raining" if self.raining else ""
            label = "daytime" if self.phase == "day" else self.phase
            parts.append(f"it is {label}{weather}")
        if self.pos:
            x, y, z = self.pos
            where = "on the ground" if self.on_ground else "in the air"
            if self.in_water:
                where = "in water"
            parts.append(f"at x={x:.0f}, y={y:.0f}, z={z:.0f} ({where})")
        inv = ", ".join(f"{c}x {n}" for n, c in sorted(self.inventory.items())) or "empty"
        parts.append(f"inventory: {inv}")
        tools = ", ".join(f"{cls} {tier}".strip() for cls, tier in self.tools.items()
                          if tier and tier != "none") or "no tools"
        parts.append(f"tools: {tools}")
        close = [(n, c, d) for n, c, d in self.hostiles if d is None or d <= 16]
        if close:
            desc = ", ".join(
                f"{n} x{c}" + (f" at {d:.0f}m" if d is not None else "")
                for n, c, d in close[:3]
            )
            parts.append(f"hostile mobs within 16 blocks: {desc}")
        else:
            parts.append("no hostile mobs within 16 blocks")
        if self.players:
            parts.append(f"players nearby: {', '.join(self.players[:3])}")
        if self.last_death_cause:
            age = f" {self.last_death_age_s:.0f}s ago" if self.last_death_age_s is not None else ""
            parts.append(f"last death:{age} {self.last_death_cause}")
        parts[0] = parts[0][:1].upper() + parts[0][1:]
        return "You are a Minecraft bot. " + ". ".join(parts) + "."


def _summarize_events(data: dict, limit: int = 3) -> tuple[list[str], int]:
    result = get(data, "response", "data", "result", default={}) or {}
    raw = result.get("events") if isinstance(result, dict) else None
    out: list[str] = []
    for e in (raw or [])[-limit:]:
        if isinstance(e, dict):
            summary = " ".join(str(e.get("summary") or e.get("type") or "").split())
            out.append(summary[:80])
    unread = get(data, "notifications", "unreadCount", default=0)
    return out, int(unread) if isinstance(unread, int) else 0


def parse_status(data: dict) -> GameState:
    gs = GameState()
    sit = get(data, "response", "data", "result", "situation", default={}) or {}
    if not isinstance(sit, dict):
        return gs

    vitals = sit.get("vitals") or {}
    gs.health = _num(vitals.get("health"))
    gs.food = _num(vitals.get("food"))
    gs.saturation = _num(vitals.get("saturation"))

    clock = sit.get("clock") or {}
    gs.phase = clock.get("phase")
    raining = clock.get("raining")
    gs.raining = bool(raining) if isinstance(raining, bool) else None

    pos = sit.get("position") or {}
    if all(k in pos for k in ("x", "y", "z")):
        gs.pos = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
    gs.on_ground = pos.get("onGround") if isinstance(pos.get("onGround"), bool) else None
    gs.in_water = pos.get("inWater") if isinstance(pos.get("inWater"), bool) else None
    gs.in_lava = pos.get("inLava") if isinstance(pos.get("inLava"), bool) else None

    gs.dimension = sit.get("dimension")

    inv = sit.get("inventory") or {}
    for name, cnt in _iter_items(inv.get("stacks")):
        gs.inventory[name] = gs.inventory.get(name, 0) + cnt

    tools = (sit.get("tools") or {}).get("tools") or []
    for t in tools:
        if isinstance(t, dict) and t.get("class"):
            gs.tools[str(t["class"])] = str(t.get("tier") or "none")

    nearby = sit.get("nearby") or {}
    hostile_rows = nearby.get("hostiles")
    if not isinstance(hostile_rows, list) or not hostile_rows:
        hostile_rows = [m for m in (nearby.get("mobs") or [])
                        if isinstance(m, dict) and m.get("kind") == "hostile"]
    for m in hostile_rows:
        if not isinstance(m, dict):
            continue
        nearest = m.get("nearest") or {}
        dist = _num(nearest.get("distance")) if isinstance(nearest, dict) else None
        gs.hostiles.append((str(m.get("name") or "?"), int(m.get("count") or 1), dist))
    dists = [d for _, _, d in gs.hostiles if d is not None]
    gs.nearest_hostile = min(dists) if dists else None

    for p in nearby.get("players") or []:
        if isinstance(p, dict) and p.get("username"):
            gs.players.append(str(p["username"]))

    death = sit.get("lastDeath") or {}
    if isinstance(death, dict) and death.get("cause"):
        gs.last_death_cause = str(death["cause"])
        gs.last_death_age_s = _age_seconds(sit.get("observedAt"), death.get("observedAt"))

    activity = sit.get("activity") or {}
    gs.activity_owner = activity.get("owner")
    gs.active_action = activity.get("activeAction")

    surv = get(data, "response", "data", "survival", default={}) or {}
    summary = surv.get("summary") if isinstance(surv, dict) else None
    if isinstance(summary, str) and summary not in ("unknown", ""):
        gs.survivors.append(summary)
    return gs


def _age_seconds(now_iso: Any, then_iso: Any) -> float | None:
    if not (isinstance(now_iso, str) and isinstance(then_iso, str)):
        return None
    from datetime import datetime, timezone

    def parse(s: str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    now, then = parse(now_iso), parse(then_iso)
    if now and then:
        return max(0.0, (now - then).total_seconds())
    return None


async def build(executor) -> GameState:
    """Perceive through the MCP read tools (ungated, cheap)."""
    status = await executor.read("view_status", {}, "read live bot vitals and position")
    events = await executor.read(
        "read_recent_events", {"limit": 5}, "read the newest world events for context"
    )
    gs = parse_status(coerce(status.data, status.text))
    gs.events, gs.unread = _summarize_events(coerce(events.data, events.text))
    return gs
