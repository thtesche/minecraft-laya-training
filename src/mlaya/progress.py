"""Where the run stands: one object, one precedence.

"Fortschritt" was never an object in this loop. It is three sources with different half-lives
sitting side by side:

* the **ledger** -- the MCP server's own SQLite (`action_executions`, `events`,
  `frontier_chunk_materials`, `stronghold_throws`, `notes`). It survives a death, a dimension
  change and a restart, and it answers from 300 blocks away, where no block scan reaches;
* the **scan** -- `view_status` and `view_blocks`, true right now but only for loaded chunks;
* the **latch** -- the ~40 flags in `Agent.__init__`, which every cold start resets to
  `False`/`[]`.

`Progress` is the single answer those three give, and the order is fixed everywhere:

    durable ledger fact  >  world scan  >  run latch  >  False

A latch may *set* a milestone, never refute one the world answered. The one exception is the
frame, which the scan may still undo: a plane that names its own obsidian as air is the frame
talking about itself (`graph.py`'s `_frame_short_at_base` rule, live run103).

The predicates in `actions/predicates.py` stay as they are -- they are the adapters this object
is built from, and rewriting them would rewrite the tests that pin them. What changes is that
`_inject` hands every tick one `Progress`, and the milestone sentence, the checkpoint note and the
cold-start bootstrap all read that one object instead of three ledgers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import perception
from .actions import (BLAZE_POWDER_ITEM, END_PORTAL_SOCKETS, ENDER_EYE_ITEM, EYE_THROW_BUFFER,
                      _all_nether_prep_steps, _bed_at_base, _frame_complete_at_base,
                      _frame_short_at_base, _food_reserve_met, _gear_ready,
                      _nether_bank_carried, _overworld, _portal_lit_scan)

# How close to the gate a chest has to stand to count as *the base's* chest when the run has not
# adopted a base yet. The loop's own network radius is 4; a gate chest sits at base+(0,1,1), so
# anything further is somebody else's container and its powder is not this run's milestone.
GATE_CHEST_RADIUS = 6


@dataclass
class DurableFacts:
    """What the server's own record says this world already contains.

    Read once per run (`bootstrap_progress`), never per tick: the ledger does not change shape
    during a run in a way the loop could act on, and paying five queries every tick would be
    five reads for a fact that is already in the latches.
    """

    gate_lit: bool = False
    frame_complete: bool = False
    end_portal_active: bool = False
    frame_eyes: int = 0
    stronghold_estimated: bool = False
    nether_entered: bool = False
    in_end: bool = False
    # Cells a bed was laid in. Whether one of them is *at the base* is a base-side question the
    # ledger cannot answer, so the cells are kept and the milestone is decided against `base`.
    bed_cells: tuple[tuple[int, int, int], ...] = ()
    # The Overworld cell the last `activate_portal` was aimed at: the gate's own position, which
    # is what the powder milestone measures a chest's distance from.
    gate_cell: tuple[int, int, int] | None = None
    # Counters only this run knows, restored from the loop's own checkpoint note.
    pearl_wave_trips: int = 0
    barter_attempts: int = 0
    checkpoint: dict[str, Any] = field(default_factory=dict)


def _cell(args: dict) -> tuple[int, int, int] | None:
    x, y, z = (args.get("x"), args.get("y"), args.get("z"))
    if all(isinstance(v, int) for v in (x, y, z)):
        return (x, y, z)
    return None


def _num(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


async def bootstrap_progress(executor) -> DurableFacts:
    """Read the server's durable record of this world, once, at the start of a run.

    Every read is a `query_bot_data` the server already answers, and each is wrapped on its own:
    a world with no ledger rows, a schema the server has not created yet, or a transport blip
    must not stop the run from deciding anything. An unread fact is simply an absent one, and
    the milestone then falls back to the scan and the latch exactly as it did before.
    """
    facts = DurableFacts()

    async def read(sql: str, why: str) -> Any:
        res = await executor.read("query_bot_data", {"sql": sql}, why)
        return perception.coerce(res.data, res.text)

    # a) A lit portal block anywhere the server has ever scanned a chunk: the gate burns, even
    #    if its chunk is unloaded now.
    try:
        chunks = perception.parse_gate_chunks(await read(perception.GATE_CHUNK_SQL,
                                                   "find the chunks this world's lit portal was seen in"))
        if chunks:
            facts.gate_lit = True
    except Exception:
        pass

    # b) Every portal this bot has tried to light. Newest first, so the first Nether hit is the
    #    gate the run is standing next to, and the end rows add up the frame.
    try:
        for entry in perception.parse_portal_ledger(await read(perception.PORTAL_LEDGER_SQL,
                                                         "read the portals this bot has already lit")):
            portal = entry["result"].get("portal") or {}
            kind = portal.get("kind")
            if kind == "nether":
                if _num(portal.get("portalAfter")):
                    facts.gate_lit = True
                    # A frame that lit is a frame that stands: `activate_portal` refuses
                    # `[PORTAL_NO_FRAME]` before it ever reaches the flint and steel.
                    facts.frame_complete = True
                    if facts.gate_cell is None:
                        facts.gate_cell = _cell(entry["args"])
            elif kind == "end":
                # `eyesAfter` is the server counting the sockets in the world, not the eyes this
                # run happened to carry - the number the eye chain has been sizing on wrong.
                eyes = _num(portal.get("eyesAfter"))
                if eyes is not None:
                    facts.frame_eyes = max(facts.frame_eyes, min(END_PORTAL_SOCKETS, eyes))
                if _num(portal.get("portalAfter")) >= 9 or portal.get("activated"):
                    facts.end_portal_active = True
    except Exception:
        pass

    # c) A bed the world still holds.
    try:
        beds = perception.parse_bed_ledger(await read(perception.BED_LEDGER_SQL,
                                                "read which bed this bot laid at its base"))
        facts.bed_cells = tuple(beds[:4])
    except Exception:
        pass

    # d) Which dimensions this bot has been carried into.
    try:
        dims = perception.parse_dimension_ledger(await read(perception.DIMENSION_LEDGER_SQL,
                                                      "read which dimensions this bot has crossed into"))
        facts.nether_entered = any("nether" in d for d in dims)
        facts.in_end = any("end" in d for d in dims)
    except Exception:
        pass

    # e) A saved stronghold bearing.
    try:
        facts.stronghold_estimated = perception.parse_bearing(await read(perception.BEARING_SQL,
                                                                   "read whether a stronghold bearing is saved"))
    except Exception:
        pass

    # f) The loop's own checkpoint: the counters no ledger can prove.
    try:
        payload = await read(perception.CHECKPOINT_READ_SQL,
                             "read the last checkpoint this loop wrote for this world")
        facts.checkpoint = perception.parse_checkpoint(payload)
        if facts.checkpoint:
            facts.pearl_wave_trips = int(facts.checkpoint.get("pearl_wave_trips", 0) or 0)
            facts.barter_attempts = int(facts.checkpoint.get("barter_attempts", 0) or 0)
    except Exception:
        pass

    return facts


def progress_of(gs, durable: DurableFacts | None = None) -> Progress:
    """The milestones, resolved in the fixed order, with where each one came from."""
    d = durable or DurableFacts()
    out = Progress()

    def mark(name: str, value: bool, source: str) -> None:
        setattr(out, name, value)
        out.source[name] = source

    # --- The gate ---
    if d.gate_lit:
        mark("gate_lit", True, "ledger")
    elif _portal_lit_scan(gs):
        mark("gate_lit", True, "scan")
    else:
        mark("gate_lit", bool(gs.portal_lit), "latch" if gs.portal_lit else "default")

    # --- The frame. The scan may refute the latch here and nowhere else: a plane that names its
    #     own obsidian is the frame itself, a blank plane names nothing (see `_inject`). ---
    if _frame_short_at_base(gs):
        mark("frame_complete", False, "scan")
    elif d.frame_complete:
        mark("frame_complete", True, "ledger")
    elif _frame_complete_at_base(gs):
        mark("frame_complete", True, "scan")
    else:
        mark("frame_complete", bool(gs.frame_complete), "latch" if gs.frame_complete else "default")

    # --- The bed at the base. The ledger knows a bed cell, not which base it belongs to, so the
    #     anchor is the base this run adopted or, while there is none, the gate the ledger lit. ---
    anchor = gs.base or d.gate_cell
    bed_durable = anchor is not None and any(
        abs(x - (anchor[0] + 0.5)) <= GATE_CHEST_RADIUS and abs(z - (anchor[2] + 0.5)) <= GATE_CHEST_RADIUS
        for x, _, z in d.bed_cells)
    if bed_durable:
        mark("bed_at_base", True, "ledger")
    elif _bed_at_base(gs):
        mark("bed_at_base", True, "scan")
    else:
        mark("bed_at_base", gs.bed_pos is not None, "latch" if gs.bed_pos is not None else "default")

    # --- The kit and the prep: only the pack can answer these, and it answers every tick. ---
    mark("kit_worn", _gear_ready(gs), "scan")
    mark("prep_done", not _all_nether_prep_steps(gs) and _nether_bank_carried(gs)
                     and _food_reserve_met(gs), "scan")

    # --- The powder bank. The durable read of it is the container ledger `_note_stocked_chest`
    #     runs, which lands in the latch; a pack of powder in the Overworld is the live read. ---
    if gs.powder_stowed:
        mark("powder_banked", True, "latch")
    elif _overworld(gs) and int(gs.inventory.get(BLAZE_POWDER_ITEM, 0)) > 0:
        mark("powder_banked", True, "scan")
    else:
        mark("powder_banked", False, "default")

    # --- The end portal ---
    if d.end_portal_active or gs.end_portal_active:
        mark("end_portal_active", True, "ledger" if d.end_portal_active else "latch")
    else:
        mark("end_portal_active", False, "default")

    if d.frame_eyes:
        out.frame_eyes = max(d.frame_eyes, int(gs.frame_eyes))
        out.source["frame_eyes"] = "ledger"
    else:
        out.frame_eyes = max(0, min(END_PORTAL_SOCKETS, int(gs.frame_eyes)))
        out.source["frame_eyes"] = "latch" if out.frame_eyes else "default"

    if d.stronghold_estimated or gs.stronghold_estimated:
        mark("stronghold_estimated", True, "ledger" if d.stronghold_estimated else "latch")
    else:
        mark("stronghold_estimated", False, "default")

    dim = str(gs.dimension or "").lower()
    if "end" in dim:
        mark("in_end", True, "scan")
    elif d.in_end:
        mark("in_end", True, "ledger")
    else:
        mark("in_end", False, "default")

    if gs.nether_entered or d.nether_entered:
        mark("nether_entered", True, "latch" if gs.nether_entered else "ledger")
    else:
        mark("nether_entered", False, "default")

    empty = 0 if out.end_portal_active else max(0, END_PORTAL_SOCKETS - out.frame_eyes)
    buffer = 0 if out.stronghold_estimated else EYE_THROW_BUFFER
    out.eyes_owed = max(0, empty + buffer - int(gs.inventory.get(ENDER_EYE_ITEM, 0)))
    out.source["eyes_owed"] = "derived"
    return out


@dataclass
class Progress:
    """The run's milestones, and where each one came from."""

    kit_worn: bool = False
    bed_at_base: bool = False
    frame_complete: bool = False
    gate_lit: bool = False
    prep_done: bool = False
    powder_banked: bool = False
    frame_eyes: int = 0
    eyes_owed: int = END_PORTAL_SOCKETS
    stronghold_estimated: bool = False
    end_portal_active: bool = False
    in_end: bool = False
    nether_entered: bool = False
    source: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kit_worn": self.kit_worn,
            "bed_at_base": self.bed_at_base,
            "frame_complete": self.frame_complete,
            "gate_lit": self.gate_lit,
            "prep_done": self.prep_done,
            "powder_banked": self.powder_banked,
            "frame_eyes": self.frame_eyes,
            "eyes_owed": self.eyes_owed,
            "stronghold_estimated": self.stronghold_estimated,
            "end_portal_active": self.end_portal_active,
            "in_end": self.in_end,
            "nether_entered": self.nether_entered,
            "source": dict(self.source),
        }

    def sentence(self) -> str:
        """The one short line that says where the run is.

        Laya truncates a long state to its first window (`docs/architecture.md`), so this has to
        sit at the front of the narrative and stay one clause per milestone. No trailing period:
        `narrative` joins the parts with ". " and closes the sentence itself.
        """
        bits = ["the gate burns" if self.gate_lit else "no gate burns yet"]
        if self.frame_complete:
            # Only said when something actually named the frame: a lit gate whose plane was never
            # scanned says nothing about the frame, and "incomplete" would send the bot to mine
            # obsidian for a frame that stands.
            bits.append("its frame stands")
        bits.append("the bed is laid" if self.bed_at_base else "no bed is laid")
        bits.append("the blaze powder is banked" if self.powder_banked
                   else "no blaze powder is banked")
        if self.end_portal_active:
            bits.append("the end portal is open")
        else:
            bits.append(f"the end portal frame holds {self.frame_eyes} of {END_PORTAL_SOCKETS} eyes")
        return "Milestones: " + ", ".join(bits)
