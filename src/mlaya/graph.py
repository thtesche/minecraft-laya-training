from __future__ import annotations

import asyncio
import math
import re
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from . import perception
from . import survival
from . import verify as vmod
from .actions import (candidates, inventory_relief, is_hostile_hunt, resolve,
                      tech_tree_complete, _at_base, _at_surface, _cell, _gear_ready,
                      _frame_complete_at_base, _frame_short_at_base,
                      nether_expedition_complete, stronghold_complete, BLAZE_POWDER_ITEM,
                      FORTRESS_MIN_M, BLAZE_ROD_ITEM)
from .brain import Brain, escalate_to_system2
from .mcp import _find
from .memory import MemoryLog
from .config import Settings
from .targets import bed_frame_site, flat_build_site, ground_targets, step_away_cell, _clear

# Safe default when a chosen label cannot be resolved (should be rare; candidates are filtered).
FALLBACK_LABEL = "get_wood"

# Labels whose step needs a destination; the M4 parameter layer picks it from `view_blocks`.
TARGET_LABELS = frozenset({"explore"})

# A placed bed within this range of the bot means the bot is standing at a home base a previous
# run (or a manual build) left behind: that bed's ground cell is re-adopted as `base`, so the run
# continues there instead of raising a second base somewhere else.
ADOPT_BED_RADIUS = 12.0
# How far a chest may sit from the re-adopted base and still belong to it (one block of slack
# over the base radius, plus the vertical step a chest is placed at).
ADOPT_CHEST_RADIUS = 4
# How long an *intended* portal crossing keeps the `_in_portal` rescue switched off, in ticks.
# The crossing action blocks until the server settles it and the teleport needs a moment after
# that, so a couple of decisions of slack; a gate the bot merely wandered into is rescued again
# right after (Agent `_note_crossing`).
PORTAL_INTENT_TICKS = 3

# Tool-result codes that mean "the bot's surroundings block the action". `craft_item` and
# `smelt_item` place a carried workstation beside the bot; in a 1x1 excavation shaft there
# is no replaceable cell, so the tool answers `[WORKSTATION_PLACEMENT_FAILED] ... no_cell`.
# `sleep` places the carried bed the same way and needs two supported cells, so the same
# shaft answers `[BED_PLACEMENT_FAILED] ... no ... flat ground` (live, armour run: three
# wasted ticks). `place_block` names its cell, and a block standing in it answers
# `[PLACE_TARGET_OCCUPIED]` (live, from-zero run: the cell beside the base held `grass_block`,
# so `place_chest` answered it on every tick). Retrying in place loops forever, so the fix is
# to change the surroundings and try again.
# `repeated_movement_failure` is the pathfinder saying the body did not actually move on a
# route it kept re-issuing: the bot is boxed in (live, run10: wedged in a 1-wide flooded shaft,
# every `navigate`, `collect_block` and `build_structure` answered it and the forced
# `escape_water` rescue spun on it forever with no Laya choice and no way out). Treating it as a
# surroundings block lets the same unblock ladder fire -- dig a pocket, then walk to open ground
# -- instead of an infinite forced loop.
# `NO_REACHABLE_MATCHING_TARGETS` is a gather that found its blocks but could reach none of them
# (`contract.ts`: "Every loaded X block was found but none could be reached"), e.g. a tree the
# server refuses to mine for liquid isolation. Retrying in place loops forever (live, run12: the
# bot stood among 56 trees and answered `get_wood` with it every tick); the unblock ladder digs a
# pocket and then walks to open ground, which is exactly the change such a refusal needs.
BLOCKED_CODES = ("WORKSTATION_PLACEMENT_FAILED", "BED_PLACEMENT_FAILED", "PLACE_BLOCK_FAILED",
                 "PLACE_TARGET_OCCUPIED", "repeated_movement_failure",
                 "NO_REACHABLE_MATCHING_TARGETS")
# Tool-result codes from a death-recovery walk (`pick_up_items` with `recover_death_items`) that
# mean the dropped stack will never be recovered: the server reached the site and its 16 m scan
# found nothing (`NO_MATCHING_ITEMS`), or it has no usable death record at all. Retrying in place
# is what stalls the loop -- `recover_drops` owns the tick for the whole recovery window -- so the
# site is written off and the tech tree resumes (see `Agent._note_recover`).
RECOVER_FINAL_CODES = ("NO_MATCHING_ITEMS", "NO_RECORDED_DEATH",
                       "DEATH_DIMENSION_MISMATCH", "DEATH_RECORD_STALE")
# A walk that cannot reach the site is final only once it repeats (`Agent._note_recover`).
RECOVER_UNREACHABLE_CODE = "RECOVERY_SITE_UNREACHABLE"
RECOVER_UNREACHABLE_REPEATS = 3

# A walk refused `no path found` after at most `BOXED_NODES` visited nodes, its closest node
# within `BOXED_CLOSEST_M` of the bot, means the pathfinder cannot leave the pocket the bot
# stands in. The pathfinder's two kinds of failure lie far apart: run90's shut-in bot answered
# 70-78 nodes, a search that really ran out answers some 100,000. `BOXED_REPEATS` such refusals
# from one cell hand the tick to `dig_out` (`Agent._note_boxed`).
BOXED_NODES = 200
BOXED_CLOSEST_M = 5.0
BOXED_REPEATS = 3
# `dig_out` rotates through the four sides (`planner`), one per try. A refused dig turns to the
# next side too, not only one that went through, so one unreachable wall is not dug at forever;
# after this many refusals from the one cell every side has been tried round and the pocket is
# judged undiggable from here, so the latch drops and the tick is freed for the rest of the
# ladder (live run101 it6-181: 176 refused digs at one east wall, the budget spent on a tunnel
# the build reach would never give).
DIG_OUT_MAX_TRIES = 8
_NO_PATH = re.compile(r"no path found after \d+ ms compute; visited (\d+) nodes"
                      r"(?:, generated \d+)?; closest node was (-?\d+),(-?\d+),(-?\d+)")


def shut_in(text: str | None, pos) -> bool:
    """True when a refusal says the pathfinder cannot leave the cells around `pos`."""
    m = _NO_PATH.search(text or "")
    if m is None or pos is None:
        return False
    closest = (int(m.group(2)) + 0.5, int(m.group(3)), int(m.group(4)) + 0.5)
    return int(m.group(1)) <= BOXED_NODES and math.dist(closest, pos) <= BOXED_CLOSEST_M
# A gathering action is refused with `[INVENTORY_FULL]` once all 36 carried slots are taken; the
# unblock there is not terrain but pack space (stow at the base, else drop), so it routes through
# a separate relief step.
INVENTORY_CODES = ("INVENTORY_FULL",)

# How many identical refusals in a row (same label, same tool-result code, no `progressed`) a
# step may answer before the loop withholds it from the offer for `STUCK_COOLDOWN` ticks. A
# single-candidate step the server refuses forever burns the whole run: live, run16 answered
# `mine_obsidian` -> `[NAVIGATION_TARGET_UNSUPPORTED]` on 215 ticks while it was the only
# candidate, so the last fallback (`explore`/`wait`) never got its turn to move the bot; run11
# (`go_home`, 230 ticks) and run14 (`recover_drops`, 117 ticks) stalled the same way and each
# needed a bespoke latch. This is the generic net under those. The label returns on its own, so a
# refusal that was merely transient is not lost, and the exploring done in between can change the
# world that caused it.
STUCK_REPEATS = 4
STUCK_COOLDOWN = 15

# The walking guard, in the same spirit and on the same window: an action that reports walking
# this far and arriving this close to where it started has not been working. `_track_stuck` only
# sees refusals, so a `collect_block` that *succeeds* while shambling was never caught: live, run43
# and run45 each spent an action walking 459 blocks to end 17 from the start -- 27 walked for every
# block gained -- and the loop's own positions looked fine, because the bot really was moving. Two
# such actions in a row and the label is withheld, so the next tick is offered something that goes
# somewhere. The Nether march passes this on merit (609 walked, 396 net).
WANDER_WALK = 80.0
WANDER_RATIO = 6.0
WANDER_REPEATS = 2

# How long the run may spend walking back to the chest it last kept before it stops pretending
# that cell is its base. A long walk is the normal case (the live one was 300 m), so this is
# generous, but it is not unbounded: an anchor the bot cannot reach would stand between it and
# every base step, and the run would look for a portal around a site it never chose.
HOME_ANCHOR_TICKS = 60

# The idle guard: a label that succeeds at nothing, `IDLE_REPEATS` ticks running. See
# `_track_idle`. `PASS_TIME_LABELS` are the steps whose job is to change nothing - a lull, a night
# sleep - so exempting them keeps the guard from ending a wait with no work to do instead.
IDLE_REPEATS = 3
PASS_TIME_LABELS = ("wait", "sleep")

# The walk off a junk dropped on open ground is a 3-block `navigate`. A refusal keeps the latch
# set (the junk is still at the bot's feet), so a step that keeps failing - rough Nether terrain
# the 3-block navigate cannot cross - would otherwise wedge the bot on that one candidate forever
# (live, run73: a fortress in view, `step_away_drops` refused on every tick). After this many
# consecutive refusals the walk off is given up and the bot moves on; the junk stays dropped,
# because the bot marches away from it, not over it, so the auto-pickup never reaches it.
STEP_AWAY_GIVE_UP = 3
# Foreground calls in a row that ran out of `[mcp] action_wait_ms` with the body still held
# (`[BODY_HELD]`) before the run stops with `"stop": "body_held"`. run89's hostile reflex held the
# body in a `fight`/`hold` against a zombie 1.6 m off for nearly three hours - full health, no
# damage, no event - and the loop could only wait; nothing it can send moves a body the reflex
# owns. Two budgets (ten minutes by default) say the server will not hand it back, and an ended
# run with that reason is something the operator sees, where a silent wait is not. run95 spent
# three such pairs of ten minutes each (it268/269, it279/280, it354/355) - both calls of a pair
# naming the *same* action id - and never reached the third in a row, so the guard never fired
# and 18% of the run went into waiting on a body nobody was going to release.
BODY_HELD_STOP = 2

# A death first seen older than this was not this run's to answer (`Agent._reapply_after_death`).
DEATH_REAPPLY_AGE_S = 90.0


def _hostile_count(gs) -> int:
    """How many hostiles the perception sees - the sum of the per-kind counts, not the kinds.

    `GameState.hostiles` is `(name, count, distance)`, so two packs of blazes read as their total,
    and a held fight that thins the crowd by one is progress even while new blazes keep the kinds.
    """
    return sum(c for _, c, _ in (gs.hostiles or []))

# The summaries the server's survival reflexes write when they act ("hostile: target died; ...",
# "fire: dry footing; ..."): a defer that sees one of these handed the tick to someone. Eating
# ("hunger: ate; ...") is no answer to the danger, so it does not count.
SURVIVAL_EVENT_PREFIXES = ("hostile:", "fire:", "recover footing:", "lava:",
                           "water:", "drowning:")

_CODE_RE = re.compile(r"\[([A-Z][A-Z_]+)\]")

# The two lines the server prints into every settled result's final progress
# (`formatProgress` in mine-ai-mcp): "**Travel:** 458.9 blocks, including ..." and
# "**Distance from start:** 16.6 blocks." or "... unavailable".
_WALKED_RE = re.compile(r"\*\*Travel:\*\*\s*([\d.]+)\s*blocks")
_NET_RE = re.compile(r"\*\*Distance from start:\*\*\s*([\d.]+)\s*blocks")

# The block the server found where a container was asked for:
# "[CONTAINER_NOT_PRESENT] The loaded target block is air, not a supported ...".
_FOUND_BLOCK_RE = re.compile(r"target block is (?:minecraft:)?([a-z_]+)")
_CHEST_BLOCKS = frozenset({"chest", "trapped_chest", "barrel"})


def _walk_progress(res) -> tuple[float | None, float | None]:
    """How far an action walked and how far it ended from its start, or None for either.

    The payload first (`_find` walks it, so nesting does not matter), then the result text, which
    carries the same two numbers in every settled answer. Anything unreadable is None: a guard that
    cannot see the walk has no evidence, and must not withhold on a guess.
    """
    data = getattr(res, "data", None)
    walked = _find(data, "distanceTravelledBlocks")
    net = _find(data, "distanceFromStartBlocks")
    if isinstance(walked, (int, float)) and isinstance(net, (int, float)):
        return float(walked), float(net)
    text = getattr(res, "text", None) or ""
    m, n = _WALKED_RE.search(text), _NET_RE.search(text)
    if m is None or n is None:
        return None, None
    return float(m.group(1)), float(n.group(1))


def _fail_code(text: str | None) -> str | None:
    """The tool-result code that identifies a refusal, for the stuck-offer guard.

    A bracketed server code (`[NAVIGATION_TARGET_UNSUPPORTED]`, `[NO_MATCHING_ITEMS]`, ...) names
    the failure exactly; when the answer carries none, its first line stands in, so a plain
    repeated answer is still recognised as the same failure.
    """
    match = _CODE_RE.search(text or "")
    if match:
        return match.group(1)
    first = (text or "").strip().splitlines()
    return first[0][:80] if first and first[0] else None


def blocked_by_surroundings(text: str | None) -> bool:
    """True when a failed tool result says the surroundings prevent the action."""
    return any(code in (text or "") for code in BLOCKED_CODES)


def blocked_by_inventory(text: str | None) -> bool:
    """True when a failed tool result says the pack is full."""
    return any(code in (text or "") for code in INVENTORY_CODES)


def unblock_action(gs, attempt: int = 0):
    """The step that clears a surroundings block, or None when nothing can.

    First choice is to dig a small pocket where the bot stands (`make_room`): it is cheap
    and keeps the bot next to what it was working on. Only if that pocket did not help
    does the second attempt fall back to walking to open ground (`explore`, M4 target pool).
    """
    if attempt < 1:
        action = resolve("make_room", gs)
        if action is not None:
            return action
    targets = ground_targets(gs)
    return resolve("explore", gs, targets[0]) if targets else None


def _chosen(answers: dict) -> dict:
    """Compact record of Laya's own answers for the log: {qid: choice or score}.

    Kept small on purpose (probabilities are dropped): the M3 dataset needs the `in_danger`
    gate label and the `intent` pick, both of which are here.
    """
    out: dict = {}
    for qid, ans in (answers or {}).items():
        if not isinstance(ans, dict):
            continue
        if "choice" in ans:
            out[qid] = ans["choice"]
        elif "score" in ans:
            out[qid] = ans["score"]
    return out


class AgentState(TypedDict, total=False):
    iteration: int
    before: Any
    decision: Any
    action: Any
    result: Any
    escalated: bool
    deferred: bool
    note: str
    done: bool
    stop: str  # why the loop ended: "goal" (tech tree complete) or "limit"


class Agent:
    def __init__(self, settings: Settings, executor):
        self.settings = settings
        self.mcp = executor
        self.brain = Brain(settings.laya)
        self.log = MemoryLog(settings.log_path())
        self.history: list[dict] = []
        # The server restores default policy on death; re-apply once per death. The death the
        # policy was last re-applied for (or found already old), by its server timestamp.
        self._death_seen: str | None = None
        # Label whose last attempt failed because the surroundings blocked it; the next
        # tick changes the surroundings before retrying it (`blocked_by_surroundings`).
        self._blocked_label: str | None = None
        # What kind of block that was: "surroundings" (dig a pocket / walk to open ground) or
        # "inventory" (make pack room). It chooses the unblock step and when the label clears.
        self._blocked_kind = "surroundings"
        # A deposit into the base chest failed (the chest is full): the inventory relief then
        # drops instead of retrying a deposit that cannot free anything.
        self._stow_blocked = False
        # How many unblock steps have been spent on that label: pocket first, then explore.
        self._unblock_attempts = 0
        # Set right after an unblock step: the next time the blocked label is chosen it runs
        # instead of being intercepted again. Without it the guard answers every retry with a
        # fresh unblock step, so the label can never succeed and stays blocked for the rest of
        # the run (live: `smelt_iron` on grassy ground, the guard answered `explore` forever
        # while the furnace never got its retry). It is only set for a terrain fix; the
        # inventory relief removes the cause itself, so it frees the label immediately.
        self._unblock_done = False
        # Home base: the surface anchor the loop chooses once (see `establish_base`) and the
        # placed chest there. Both are per-run only and injected into each GameState.
        self.base: tuple[int, int, int] | None = None
        self.chest_pos: tuple[int, int, int] | None = None
        self.banked: dict[str, int] = {}
        # The storage network grown at the base: the overflow chests placed after the first one
        # (`chest_pos` keeps the main chest), the cells the server called full and the candidate
        # cells whose ground refused a chest. Per-run state like `chest_pos` itself: a restart
        # forgets them until the server's container database or the next placement names them.
        self._extra_chests: list[tuple[int, int, int]] = []
        self._chests_full: set[tuple[int, int, int]] = set()
        self._overflow_no_support: set[tuple[int, int, int]] = set()
        # What each chest of the network last held, per cell, as `(cell, item, count)` triples --
        # the same read that aggregates `banked`, kept per chest (`_note_stocked_chest`) because
        # `take_stored_food` has to reach into the chest that actually holds the provisions.
        self._banked_stores: tuple = ()
        self._stock_read: tuple[float, float, float] | None = None
        self._stock_warned = False
        # The furnace stood at the base (`place_furnace`), remembered so `cook_food`/`smelt_gold`
        # reuse its cell instead of placing a temporary workstation each call. Re-adopted from a
        # scan after a restart, like the chest.
        self.furnace_pos: tuple[int, int, int] | None = None
        self.smoker_pos: tuple[int, int, int] | None = None
        # Whether `base` came from the world rather than from an `establish_base` this run: a lit
        # portal a previous run left standing (`_adopt_portal_base`). While the bot has not reached
        # it yet, `base_adopted` rides in the GameState and `go_home` walks it back, so a restart
        # finishes at its own portal instead of drifting off to explore.
        self._base_adopted = False
        # Position of the last "is there a lit portal anywhere?" scan, so the extra read runs
        # while the base is still unset but not on every tick (`_adopt_portal_base`).
        self._portal_base_scanned: tuple[float, float] | None = None
        # Level ground the base scan proposed for the portal frame (`targets.flat_build_site`).
        # Per-run like `base`; `_build_site_scanned` remembers where the last box read was taken
        # so standing still does not pay for the same scan every tick.
        self.build_site: tuple[int, int, int] | None = None
        self._build_site_scanned: tuple[float, float] | None = None
        # The base a standing bed anchors, but only once `_scan_bed_site` has proven its ground
        # fits the frame (`targets.bed_frame_site`). `_bed_base_scanned` remembers which bed cell
        # that read was taken for, so the two box reads are paid once per bed, not every tick.
        self._bed_base: tuple[int, int, int] | None = None
        self._bed_base_scanned: tuple[int, int, int] | None = None
        # The surface cell `(x, feet_y, z)` the loop last learned for the bot's column, or None
        # while unknown. `surface_up` asks the server for it (a `navigate` without a height) and
        # `_verify` reads the answer back into here; it is what keeps `establish_base` off a cave
        # floor, since a mined tunnel's `air` reads exactly like open sky. Per-run, like `base`.
        self.surface: tuple[int, int, int] | None = None
        # Lava cells a pour already failed on with `[BUCKET_NO_FACE]`. That answer is
        # structural -- no solid block beside that cell faces the bot -- so retrying it can
        # only fail again; remembering it lets the next tick pour elsewhere (live: the
        # surface pond holds one usable cell and the obsidian would never be made there).
        self._pour_skip: set[tuple[int, int, int]] = set()
        # Stances a `mine_obsidian` walk was refused at (`[NAVIGATION_TARGET_UNSUPPORTED]`): the
        # cell beside that lava has no footing (live, run16: the stance sat over a deep lava lake
        # and the same walk was answered on 215 ticks). The next tick stands on another neighbour
        # of the same source, and once every stance of a source is refused the step moves to the
        # next source (`_lava_pour_target`), exactly as `pour_skip` does for a refused pour.
        self._stance_skip: set[tuple[int, int, int]] = set()
        # The stuck-offer guard, one run per label: the last refusal code and the count of
        # identical failures for that label, and the ticks left in the cooling-off window that
        # withholds it (see `STUCK_REPEATS`/`STUCK_COOLDOWN` and `_track_stuck`). Per label, not a
        # single slot: while one label sat out its window the loop returned early and never counted
        # a *different* label failing on every tick it got (live, run17: `build_portal` was withheld
        # for 15 ticks, during which `light_portal` answered `[PORTAL_NO_FRAME]` 150 times without
        # ever reaching the threshold).
        self._stuck_reps: dict[str, int] = {}
        self._stuck_code: dict[str, str | None] = {}
        self._stuck_off: dict[str, int] = {}
        # The walking guard's own count per label (`_track_wander`), separate from `_stuck_reps`:
        # a wandering action is not a refusal and must not reset or advance the refusal run.
        self._wander_reps: dict[str, int] = {}
        # The remembered chest a lost base is walked back to, and the decisions left to get there
        # (see `_walk_home_to_memory`/`_settle_home_anchor` and `HOME_ANCHOR_TICKS`).
        self._home_anchor: tuple[int, int, int] | None = None
        self._home_ticks: int = 0
        # The idle guard's own count per label (`_track_idle`), separate again: a step that
        # succeeds at nothing is neither a refusal nor a wander, and must not disturb either run.
        self._idle_reps: dict[str, int] = {}
        # Death cells whose recovery walk already reached them and found nothing
        # (`[NO_MATCHING_ITEMS]` and the other final codes): `_can_recover` stops offering the
        # walk there, so the bot does not stand at an emptied death site until the window runs
        # out (live, run14: two loops, 117 ticks in all, after the kit was already lost).
        self._recover_written_off: set[tuple[int, int, int]] = set()
        # The death cell the last recovery walks could not reach, and how many in a row
        # (`[RECOVERY_SITE_UNREACHABLE]`): `RECOVER_UNREACHABLE_REPEATS` of them write it off.
        self._recover_unreachable: tuple[tuple[int, int, int] | None, int] = (None, 0)
        # The cell the last walks were refused from as shut in, and how many in a row
        # (`_note_boxed`); and how many tunnels `dig_out` has dug from that cell.
        self._boxed: tuple[tuple[int, int, int] | None, int] = (None, 0)
        self._dig_out_tries = 0
        # Decisions spent off the feet in water (`GameState.water_ticks`): the rescue waits out
        # the first ones so a crossing the server can swim is not torn up on sight (live, run15).
        self._water_ticks = 0
        # Defers in a row that met no damage and no survival reflex (`_note_defer`).
        self._defer_quiet = 0
        # Foreground calls in a row that ended `[BODY_HELD]` (`_note_held`).
        self._held_ticks = 0
        # A `drop_junk` whose disposal pit the terrain refused (`[DROP_HOLE_UNAVAILABLE]`): from
        # then on the relief drops the surplus on the open ground instead. That still frees the
        # slots it is for, and unlike the pit dig it cannot fail, so the loop cannot stall on it.
        self._drop_hole_blocked = False
        # After dropping junk on open ground (no pit), step away so items aren't re-collected.
        self._drop_open_pending = False
        self._drop_pos: tuple[float,float,float]|None = None
        # The way the bot last walked (`_note_heading`), and its value when the drop was made.
        self._heading: tuple[float, float] | None = None
        self._drop_heading: tuple[float, float] | None = None
        # Consecutive refused `step_away_drops` since the last drop; `STEP_AWAY_GIVE_UP` of them
        # releases the walk off so a step that cannot be made cannot wedge the bot.
        self._step_away_fails = 0
        # The ground read around that drop (`_scan_drop_ground`), the drop it was read for, and
        # the cells a walk off it was already refused at.
        self._drop_grid: dict = {}
        self._drop_grid_at: tuple[float, float, float] | None = None
        self._drop_refused: set[tuple[int, int, int]] = set()
        # A `place_chest` the base's ground refused (`[PLACE_NO_SUPPORT]`): no solid block
        # touches the cell beside the base, so the chest cannot stand there. Retrying can only
        # fail again, so the chain is dropped and the relief falls back to `drop_junk` (live,
        # Nether run: the answer came on every tick and the run never reached the portal).
        self._chest_blocked = False
        # A `place_furnace` the base's ground refused (`[PLACE_NO_SUPPORT]`): same as the chest,
        # the cell beside the base cannot hold it. The chain is dropped so `cook_food` falls back
        # to the temporary workstation `smelt_item` places itself.
        self._furnace_blocked = False
        # What the base furnace was last seen to hold, from a `use_container inspect`. Empty until
        # `collect_smelted` looks; `_note_furnace` clears it once the furnace has been emptied, so
        # the step stops interrupting cooks.
        self._furnace_contents: dict[str, int] = {}
        # A `smelt_item` refused this furnace as occupied (`[FURNACE_NOT_EMPTY]`). That is the
        # signal to drain it, not a reason to give the cell up.
        self._furnace_occupied = False
        # The exact cell a placement was refused at because a block stood in it
        # (`[PLACE_TARGET_OCCUPIED]`): `make_room` digs that cell clear, since a pocket around
        # the bot need not even touch it (the chest cell sits beside the base, not the bot).
        self._blocked_cell: tuple[int, int, int] | None = None
        # The opaque block the server found sitting on the base chest's lid (`[CHEST_BLOCKED]`,
        # `perception.parse_block_above`): `stow_surplus` digs that exact cell before opening the
        # chest again, since the lid cannot swing under a full cube (live, run26: the buried base
        # put a cobblestone on the chest and the full pack could never be unloaded).
        self._chest_above: tuple[int, int, int] | None = None
        # Cells of the portal opening (`perception.portal_interior`) that hold a stray block,
        # read while a frame build is pending. `clear_portal` digs them, so a block left above
        # the support layer cannot stall the gate (live: a bed in the opening left
        # `light_portal` answering `[PORTAL_NO_FRAME]` on every tick).
        self._portal_blocked: list[tuple[int, int, int]] = []
        # A `build_portal` the server answered `ok` to while it spent nothing (`_note_build`):
        # the frame already stands complete. The step's "resume a partial frame" clause would
        # otherwise re-offer the build on a finished frame for as long as any obsidian is left
        # in the pack, and a policy that keeps choosing it freezes the run at the base (live,
        # run25: 163 identical `build_portal` ticks on a finished frame, the bot at its base).
        self._frame_complete = False
        # Frame cells a `build_structure` refused to dig (`_note_frame_cells`): a container the
        # movement policy will not break unasked. `clear_frame` digs each with the explicit
        # `collect_block` the policy allows, and the build resumes once the cell is clear.
        self._refused_frame: set[tuple[int, int, int]] = set()
        # The block name of each cell of the frame plane, from the frame scan
        # (`perception.scan_frame_volume`). `clear_frame` needs the name to hand `collect_block`.
        self._frame_names: dict[tuple[int, int, int], str] = {}
        # The crossing the loop is *intending* (`enter_portal` / `leave_nether`), so the
        # `_in_portal` rescue stands down while the body is in the gate. Counted down per observe
        # (`_note_dimension`) so a gate the bot merely walked into can never keep it switched off.
        self._enter_intent_ticks = 0
        # The Nether has been entered at least once, and where the bot landed doing it. The
        # Overworld gate is then closed (`enter_portal` stops being offered) and `leave_nether`
        # has a cell to walk back to when no exit portal block is loaded in the scan. The
        # arrival is recorded off an observed crossing only: a first tick that reads Nether
        # (`_last_dimension` is None) saw no crossing, and its position is wherever the last
        # run left the bot, not a gate - which is exactly what run74 took for the way home.
        # The server's own memory of the last crossing (`_nether_gate`) is what stands in for
        # those runs instead, read once (`_note_nether_gate`).
        self._nether_entered = False
        self._nether_arrival: tuple[int, int, int] | None = None
        self._last_dimension: str | None = None
        self._nether_gate: tuple[int, int, int] | None = None
        self._gate_read = False
        self._nether_ticks = 0
        self._nether_travel = 0.0
        self._nether_last_pos: tuple[float, float] | None = None   # (x, z) of the last tick
        # The blaze powder reached the chest at the Overworld portal (`_note_stow`): the phase-3
        # goal. Latched on the deposit's own success, the same shape as `_frame_complete`.
        self._powder_stowed = False
        # The stronghold phase's latches (docs/stronghold.md), each set on its action's own
        # result: the open end portal (`_note_portal`, the phase goal), the saved eye-of-ender
        # bearing (`_note_locate`) and a gate chest out of powder (`_note_take_powder`).
        self._end_portal_active = False
        self._stronghold_estimated = False
        self._powder_exhausted = False
        # Barter calls the run has made (`_note_barter`). The gold reserve is dug only while the
        # cap is not reached, so a route that keeps missing pearls does not turn the lull into a
        # permanent gold farm.
        self._barter_attempts = 0
        # The cells a powder stow actually deposited into (`_note_stow`): the network's roomy
        # chest, not necessarily `chest_pos`. `take_blaze_powder` withdraws from these, and the
        # exhaustion latch only fires once every remembered cell has come up empty.
        self._powder_cells: list[tuple[int, int, int]] = []
        # Fortress landmark cells the bot has already stood on with nothing to show for it
        # (`_note_landmark`); `_fortress_hit` refuses them, so the Nether search moves on instead
        # of re-walking a cage it is already on.
        self._nether_landmarks: list[tuple[int, int, int]] = []
        self.graph = self._build()

    def _inject(self, gs):
        """Carry the loop-owned facts perception cannot see into the GameState."""
        self._adopt_homebase(gs)
        gs.base = self.base
        gs.surface = self.surface
        gs.chest_pos = self.chest_pos
        gs.banked = dict(self.banked)
        gs.banked_stores = self._banked_stores
        gs.extra_chests = tuple(self._extra_chests)
        gs.chests_full = frozenset(self._chests_full)
        gs.overflow_skip = frozenset(self._overflow_no_support)
        gs.nether_travel = self._nether_travel
        gs.barter_attempts = self._barter_attempts
        gs.furnace_pos = self.furnace_pos
        gs.smoker_pos = self.smoker_pos
        gs.base_adopted = self._base_adopted
        gs.build_site = self.build_site
        gs.pour_skip = frozenset(self._pour_skip)
        gs.stance_skip = frozenset(self._stance_skip)
        gs.stuck_labels = frozenset(self._stuck_off)
        gs.recover_written_off = frozenset(self._recover_written_off)
        gs.boxed_in = self._boxed[1] >= BOXED_REPEATS
        gs.dig_out_try = self._dig_out_tries
        # One decision off the feet in water, counted here so `candidates` can hold the rescue
        # back for its first moments (live, run15).
        self._water_ticks = self._water_ticks + 1 if (gs.in_water and not gs.on_ground) else 0
        gs.water_ticks = self._water_ticks
        gs.defer_quiet = self._defer_quiet
        gs.drop_hole_blocked = self._drop_hole_blocked
        gs.drop_open_pending = self._drop_open_pending
        gs.drop_pos = self._drop_pos
        gs.drop_heading = self._drop_heading
        gs.drop_away = (step_away_cell(self._drop_grid, self._drop_pos, gs.pos,
                                       frozenset(self._drop_refused), self._drop_heading)
                        if self._drop_open_pending and self._drop_pos is not None else None)
        gs.chest_blocked = self._chest_blocked
        gs.furnace_blocked = self._furnace_blocked
        gs.furnace_contents = dict(self._furnace_contents)
        gs.furnace_occupied = self._furnace_occupied
        gs.blocked_cell = self._blocked_cell
        gs.chest_above = self._chest_above
        gs.portal_blocked = list(self._portal_blocked)
        gs.refused_frame = tuple(sorted(self._refused_frame))
        gs.frame_names = dict(self._frame_names)
        # The frame-plane read only exists with the base chunks loaded, so a bot that finished the
        # frame and walked off reads it back as blank, `_frame_needs_obsidian` sees "no frame", and
        # the obsidian chain reopens for a gate already built (live run97 it142-145: 290 m out, a
        # blank scan, a second ten obsidian mined). Latch completion the moment the plane shows it -
        # the same one-way flag `_note_build` sets from a no-op build - so it holds wherever the bot
        # roams and a later blank scan can never read the frame back to unbuilt.
        if _frame_short_at_base(gs):
            # The latch is one-way, and a frame can lose blocks after it fires: live, run103 -
            # ten obsidian spent at it114, the plane read latched it complete, two of those cells
            # were air again by it191, and the bot held the two blocks in its pack and `wait`ed at
            # its own gate for fifty ticks while `build_portal` stayed stepped over. The read that
            # names the frame's own obsidian is the frame talking about itself, so it may undo the
            # latch; a blank read names nothing and leaves it standing (run97).
            self._frame_complete = False
        elif not self._frame_complete and _frame_complete_at_base(gs):
            self._frame_complete = True
        gs.frame_complete = self._frame_complete
        gs.enter_intent = self._enter_intent_ticks > 0
        gs.nether_entered = self._nether_entered
        gs.nether_arrival = self._nether_arrival
        gs.nether_gate = self._nether_gate
        gs.nether_ticks = self._nether_ticks
        gs.powder_stowed = self._powder_stowed
        gs.end_portal_active = self._end_portal_active
        gs.stronghold_estimated = self._stronghold_estimated
        gs.powder_exhausted = self._powder_exhausted
        gs.powder_cells = tuple(self._powder_cells)
        gs.nether_landmarks = frozenset(self._nether_landmarks)
        return gs

    async def _scan_build_site(self, gs) -> None:
        """Look for level ground to raise the base on, while the base is still unset.

        The portal frame only fits where its whole bottom row stands on level ground, so the
        base scan reads two boxes around the bot -- its own level and four above
        (`perception.scan_ground` + `targets.flat_build_site`). Skipped once a site is found or
        a base is adopted, before the diamond kit exists (the base is not raised yet) and while
        the bot is still below the surface (`_at_surface` -- a cave floor is not a base site);
        re-run only after the bot has moved a few blocks, so it does not pay for the reads
        every tick.
        """
        if self.base is not None or self.build_site is not None or gs.pos is None:
            return
        if not _gear_ready(gs):
            return
        if not _at_surface(gs):
            # No surface learned yet, or the bot is still on a cave floor: a scan here would
            # propose the cave floor itself, whose mined `air` overhead reads exactly like sky
            # (`flat_build_site` cannot tell). Wait until `surface_up` has brought the bot out.
            return
        px, _, pz = gs.pos
        at = self._build_site_scanned
        if at is not None and abs(at[0] - px) + abs(at[1] - pz) < 4.0:
            return
        self._build_site_scanned = (px, pz)
        grid: dict = {}
        # Two reads: the bot's own level, and four above it. The single box reaches only ±3,
        # and a frame site needs four clear cells above its ground, so one read alone can only
        # confirm ground at or below the bot's feet -- useless to a bot that stands in a dip
        # with the level ground overhead.
        for dy in (0, 4):
            grid.update(perception.parse_box(await perception.scan_ground(self.mcp, gs.pos, dy)))
        self.build_site = flat_build_site(grid, gs.pos)

    async def _scan_bed_site(self, gs) -> None:
        """Prove the site a nearby standing bed anchors can carry the portal frame, once.

        `_adopt_homebase` re-anchors the base on a bed a restart (or an early `sleep`) left in the
        world, and that path never met `establish_base`'s `flat_build_site` clearance. A bed the
        run laid underground or on a slope then pins the base where the frame cannot be raised
        (live, run17: the base sat nine blocks under the surface and the frame's top and west cells
        stayed stone). This reads the ground around the bed -- its own level and four above, as
        `_scan_build_site` does for the bot -- and records the base only when
        `targets.bed_frame_site` accepts it, so a bad site falls through to `establish_base`. The
        read is kept per bed cell, so it is paid once.
        """
        if self.base is not None or gs.pos is None:
            return
        bed = gs.resource_nearest("white_bed")
        if bed is None or bed.distance > ADOPT_BED_RADIUS:
            self._bed_base = None
            self._bed_base_scanned = None
            return
        at = (bed.x, bed.y, bed.z)
        if self._bed_base_scanned == at:
            return
        self._bed_base_scanned = at
        grid: dict = {}
        for dy in (0, 4):
            grid.update(perception.parse_box(
                await perception.scan_ground(self.mcp, (bed.x, bed.y, bed.z), dy)))
        self._bed_base = (bed.x, bed.y - 1, bed.z) if bed_frame_site(grid, bed) else None

    async def _scan_drop_ground(self, gs) -> None:
        """Read the ground around an open drop once, so the walk off it aims at real footing.

        One `view_blocks box` per drop, paid only while the walk off it is still owed; the
        cell itself is picked on every tick from this read (`targets.step_away_cell`), passing
        over the cells a walk was refused at.
        """
        if not self._drop_open_pending or self._drop_pos is None:
            self._drop_grid, self._drop_grid_at = {}, None
            return
        if self._drop_grid_at == self._drop_pos:
            return
        self._drop_grid_at = self._drop_pos
        self._drop_refused.clear()
        try:
            self._drop_grid = perception.parse_box(await perception.scan_ground(
                self.mcp, self._drop_pos, 0,
                "read the ground around the junk just dropped to find footing to step away to"))
        except Exception:
            self._drop_grid = {}

    async def _observe(self):
        """One perception read, with the loop-owned facts and the base-site scan folded in."""
        gs = await perception.build(self.mcp)
        await self._adopt_portal_base(gs)
        await self._scan_bed_site(gs)
        await self._scan_drop_ground(gs)
        await self._walk_home_to_memory(gs)
        await self._note_nether_gate(gs)
        await self._note_stocked_chest(gs)
        self._note_dimension(gs)
        gs = self._inject(gs)
        await self._scan_build_site(gs)
        await self._scan_portal_opening(gs)
        if self._base_adopted and _at_base(gs):
            # The re-adopted base has been reached: from here behave as if the run raised it.
            self._base_adopted = False
        self._settle_home_anchor(gs)
        return gs

    async def _walk_home_to_memory(self, gs) -> None:
        """Stand a provisional base on the chest this bot last kept, so a run can walk home again.

        `base` is per-run state, so a restart loses it, and both ways back are bounded by what the
        world has loaded: the lit portal and the bed a previous run left are found only within view
        distance, which after a death in the Nether is nowhere near the base. That is not a rare
        edge - the bot died 300 m from home, the bed at the base was gone, and the run answered by
        raising a second base on open ground and pouring a second portal with obsidian it had to
        mine again (live, run47, `establish_base` at (19,73,476) beside 4 fresh obsidian).

        The chest is different. The server keeps every container the bot has opened in its own
        database, and a read of that database is not a read of the world: the entry outlives the
        death, the dimension change and the restart, and it answers from any distance. So the
        standing cell beside it becomes a provisional base and `go_home` walks the bot there. On
        arrival `_settle_home_anchor` drops the guess, because by then the real anchor is in view
        distance and the portal and bed scans can name it properly. The chest alone cannot: it is
        placed at base+(0,1,1), but `_adopt_homebase` also accepts any chest within four blocks, so
        its position does not determine the base and must not be treated as if it did.

        One read, once per run, and only while the run has no base at all. This is the rescue, not
        the normal path: a run that has never kept a chest finds nothing and carries on.
        """
        if self.base is not None or self.chest_pos is not None or gs.pos is None:
            return
        if "nether" in str(gs.dimension or "").lower():
            # The remembered chest is in the Overworld. Guessing a base from a Nether position is
            # how a run ends up building where it cannot come back from.
            return
        try:
            home = await perception.remembered_home(self.mcp)
        except Exception:
            return
        if home is None:
            return
        self._home_anchor = home
        self._home_ticks = HOME_ANCHOR_TICKS
        self.chest_pos = home
        self.base = (home[0], home[1] - 1, home[2] + 1)

    async def _note_nether_gate(self, gs) -> None:
        """Read the server's own memory of the last crossing, once, while the bot is in the Nether.

        A run that starts with the bot in the Nether has no crossing to record the exit cell
        from: the one that put the bot down there happened while another process was running,
        and the graph resets with every restart. It is not without the fact though, because
        the server is not: it writes a `player_dimension_change` event with the position it
        delivered the bot at, in the database that outlives the death, the dimension change and
        the restart (`perception.REMEMBERED_GATE_SQL`). That is the way home out of the Nether,
        and the alternative is what run74 did with a "arrival" of its own making: 250 blocks of
        walking to the cell it started on, then standing at bare basalt for a gate that was
        never there, the bot dying down where no drop is ever coming back.

        The same read of memory `_walk_home_to_memory` makes, once per run and paid only when
        the bot stands where the answer is used: the overworld tick does not owe it.
        """
        if self._gate_read or "nether" not in str(gs.dimension or "").lower():
            return
        self._gate_read = True
        try:
            gate = await perception.remembered_gate(self.mcp)
        except Exception:
            return
        self._nether_gate = gate

    async def _note_stocked_chest(self, gs) -> None:
        """Read what every Overworld chest the network owns last held; keep the edibles.

        The same database read `_walk_home_to_memory` makes, one column wider. It is a read of the
        server's memory rather than of the world, so it answers from the Nether, from 300 blocks
        away and after a restart, and it costs no `use_container` inspection on the tick.

        Every chest of the network, not the newest one: a storage network grows past its first
        chest, and a read bound to one cell would lose the provisions standing in the overflow --
        the total that stops hunts would come up one container short and the run would hunt
        beside its own base again. The database knows no network (it lists every container the
        bot has ever opened, in any base, in any world), so the loop filters the rows to the
        cells it placed or adopted: what it owns is what it banks.

        Edibles only: the count exists to answer "is there food banked?", and every other item in
        the chest would be carried into every state, the narrative and every logged tick for
        nothing. A snapshot from the last observation is the right shape - the loop is the only
        writer, so what the chest held when the bot last touched it is what the bot put there.
        """
        if self._stock_read is not None and gs.pos is not None:
            dx = abs(self._stock_read[0] - gs.pos[0]) + abs(self._stock_read[2] - gs.pos[2])
            if dx < 16.0:
                return
        try:
            res = await self.mcp.read("query_bot_data",
                                    {"sql": perception.REMEMBERED_STOCKS_SQL},
                                    "read what the chests at the base last held")
            rows = perception.parse_remembered_stocks(perception.coerce(res.data, res.text))
        except Exception as exc:
            # Loudly, once, and only while it has never worked. Swallowed, a wrong argument name
            # and a genuinely empty chest look identical - both leave `banked` empty - and the read
            # would then quietly fail to prevent the hunt it exists to prevent. Logged rather than
            # raised: a chest that cannot be read is not a reason to stop the run.
            if not self._stock_warned:
                self._stock_warned = True
                self.log.append({"stage": "policy", "reason": "chest_stock_unreadable",
                                 "error": str(exc)[:160]})
            return
        self._stock_read = gs.pos
        network = {tuple(self.chest_pos)} if self.chest_pos is not None else set()
        network |= {tuple(cell) for cell in self._extra_chests}
        edible: dict[str, int] = {}
        stores: list[tuple] = []
        powder_cells: list[tuple] = []
        for cell, counts in rows:
            if cell not in network:
                continue
            if int(counts.get(BLAZE_POWDER_ITEM, 0)) > 0:
                powder_cells.append(cell)
            for name, count in counts.items():
                if name in perception.EDIBLE_NAMES and int(count) > 0:
                    edible[name] = edible.get(name, 0) + int(count)
                    stores.append((cell, name, int(count)))
        self.banked = edible
        self._banked_stores = tuple(stores)
        self._adopt_powder_stow(powder_cells)

    def _adopt_powder_stow(self, powder_cells: list[tuple]) -> None:
        """Re-derive the powder-stowed milestone from the chest a previous run filled.

        `_powder_stowed` (= `nether_expedition_complete`) is the loop latch set when a stow
        deposit succeeds (`_note_stow`); a cold start forgets it, and `nether_expedition_open`
        then reads True again and marches the bot back through the gate for nine rods it already
        banked - the most dangerous stretch of the run, re-run for powder sitting in its own chest.
        The powder is a real deposit, though, and the server's container database answers it across
        the restart (the read `_note_stocked_chest` just made, filtered to the cells this base
        owns). So blaze powder standing in a network chest latches the milestone from the world,
        the same fact the deposit would have set: the first Nether haul is behind the bot.

        One-way and only ever set True: a chest drained of its powder (withdrawn and spent) leaves
        the latch untouched, and `_note_take_powder`/`_nether_topup_wanted` own that reopening. The
        cells are remembered too (`_powder_cells`), so a withdraw after a restart looks in the same
        chest the stow used rather than reading an empty main chest and re-hauling a top-up it does
        not owe.
        """
        if not powder_cells:
            return
        for cell in powder_cells:
            if cell not in self._powder_cells:
                self._powder_cells.append(cell)
        if not self._powder_stowed:
            self._powder_stowed = True
            self.log.append({"stage": "policy", "reason": "powder_stow_readopted",
                             "cells": powder_cells})

    def _settle_home_anchor(self, gs) -> None:
        """Retire the provisional base once the bot stands on it, or when the window runs out.

        On arrival the guess has done its job and has to go: the base it named is a guess, and a run
        that kept it would go on pouring the frame plane at a cell it never checked --
        `flat_build_site` only ever cleared the site the run chose for itself. With `base` unset
        again, the lit portal and the bed are within view distance and `_adopt_portal_base` /
        `_adopt_homebase` name the real one on the next tick. If neither is standing,
        `establish_base` raises a fresh site where the bot actually stands, which is the right
        answer when there is genuinely nothing left to go back to.
        """
        if self._home_anchor is None:
            return
        if _at_base(gs):
            self._home_anchor = None
            self._home_ticks = 0
            self.base = None
            return
        self._home_ticks -= 1
        if self._home_ticks > 0:
            return
        # The walk did not arrive in `HOME_ANCHOR_TICKS` decisions. Stop guessing: an anchor the
        # bot cannot reach would stand between it and every base step, and the stuck guard would
        # then have to unpick a stall that was never the world's fault.
        self.log.append({"stage": "policy", "reason": "home_anchor_given_up",
                         "cell": self._home_anchor, "ticks": HOME_ANCHOR_TICKS})
        self._home_anchor = None
        self._home_ticks = 0
        self.base = None

    def _note_dimension(self, gs) -> None:
        """Count down a standing crossing intent and remember the Nether arrival.

        The intent is a tick count, not a flag that a gate is seen: the crossing action blocks
        until the server settles it and the teleport needs a moment after that, so the rescue has
        to stay down for a bounded number of ticks around the crossing and then come straight back
        - a gate the bot merely walked into must never keep the rescue switched off for good.

        Arrival is read off the observed dimension rather than off the action result, because that
        is what actually puts the bot in the Nether — but only off a crossing the run observed:
        a dimension that flips from not-Nether to Nether. A first tick that already reads Nether
        saw no crossing, and the cell it stands on says nothing about a portal; it is wherever the
        previous run left the bot, and treating it as the arrival is what run74 did: (-469,57,421)
        of bare basalt, walked back to, mistaken for the way home until the bot died
        (`_note_nether_gate` is the cold start's real source).
        """
        dim = str(gs.dimension or "").lower()
        crossed = "nether" in dim and self._last_dimension is not None \
            and "nether" not in self._last_dimension
        self._last_dimension = dim
        if self._enter_intent_ticks > 0:
            self._enter_intent_ticks -= 1
        if "nether" in dim:
            self._nether_ticks += 1
            if not self._nether_entered:
                self._nether_entered = True
                self._nether_travel = 0.0
            if crossed and gs.pos is not None:
                # Every crossing the run watches over is its own: after the bot comes home and
                # goes back down, the next trip's exit cell is the one this line records.
                self._nether_arrival = (math.floor(gs.pos[0]), math.floor(gs.pos[1]),
                                        math.floor(gs.pos[2]))
            elif gs.pos is not None and self._nether_last_pos is not None:
                # Ground covered since the crossing, which is the only thing a search budget can
                # honestly be spent on. `nether_ticks` counted decisions, and a decision is not
                # search: live, run70, nineteen of the twenty-four ticks that ended the hunt were
                # `body owner: hostile_reflex` - the bot stood at 108 m from its own portal and
                # was shot at, and the search had not begun.
                self._nether_travel += math.dist((gs.pos[0], gs.pos[2]),
                                                 self._nether_last_pos)
            if gs.pos is not None:
                self._nether_last_pos = (gs.pos[0], gs.pos[2])
        else:
            self._nether_ticks = 0
            self._nether_last_pos = None
            # Reset on the way home as well as on arrival. Left alone, the next trip would start
            # with the last trip's ground already counted and end its search before it began.
            self._nether_travel = 0.0
            # Same for the cages already written off: a new trip searches a new stretch of the
            # world, and the fortress the last one stood in is not this one's target.
            self._nether_landmarks = []

    def _note_landmark(self, action, delta: dict, after) -> None:
        """Write off a fortress landmark the search reached and found no blaze at.

        `explore` in the Nether navigates to the nearest spawner or brick. A real blaze fortress
        answers with a blaze in reach, so `hunt_blaze` - not `explore` - takes the tick; a lone
        magma-cube cage or a bastion scrap does not, yet it stays the nearest `_fortress_hit`, so
        the search navigates straight back to it the moment the frontier march steps off it.
        Writing it off only on a 0-metre step was not enough: once the underfoot rule marched the
        body away, every return walk *did* move (live, run105: the cage at (94,39,86) re-acquired
        from 56-113 m out, the bot ping-ponging 94<->150<->207 every other tick). So a landmark the
        body has *reached* - arrived within `FORTRESS_MIN_M`, whether it walked there or was already
        standing on it (run104 it155-266, 112 ticks on that same cage) - is searched and written
        off, and `_fortress_hit` skips it for the rest of the trip.
        """
        if action.label != "explore" or action.tool != "navigate":
            return
        cell = (action.args.get("x"), action.args.get("y"), action.args.get("z"))
        if None in cell or cell in self._nether_landmarks:
            return
        reached = after.pos is not None and math.dist(after.pos, cell) < FORTRESS_MIN_M
        if reached or (delta.get("pos_delta") or 0.0) < 1.0:
            self._nether_landmarks.append(cell)

    async def _adopt_portal_base(self, gs) -> None:
        """Re-anchor the home base on a lit portal a previous run left standing, once.

        `base` is per-run state, so a restart loses it - but the portal is a real block, and its
        opening corner inverts `actions._portal_site`. Adopting it puts the frame plane back on
        the standing obsidian so `_base_steps_ready` and `go_home` walk the bot back, instead of
        `establish_base` raising a second portal (impossible anyway once the flint and steel is
        crafted and the obsidian is spent). A fresh run finds no lit portal, so the base pick is
        unchanged; the scan only runs while the base is still unset and moves with the bot.

        The gear gate was the wrong question. The diamond kit is what *lights* a gate; it is not
        what recognises one, and a bot that died in the Nether came back without it - so it could
        never re-adopt the gate and chest it had already built, and `establish_base` raised a
        fresh base instead, frame and all, with the ten obsidian gone in the same death (live,
        run43: the lit gate and the chest stood at `(226,63,222)` and the run re-based on open
        ground). A lit portal already proves what it takes to build one, so the gate here is the
        means to light one: the flint and steel.
        """
        if self.base is not None or gs.pos is None:
            return
        if not (_gear_ready(gs) or "flint_and_steel" in gs.inventory):
            return
        px, _, pz = gs.pos
        at = self._portal_base_scanned
        if at is not None and abs(at[0] - px) + abs(at[1] - pz) < 8.0:
            return
        try:
            corner = await perception.scan_portal_corner(self.mcp)
        except Exception:
            return
        if corner is None:
            # A scan that found nothing must not burn the position. The throttle above is
            # measured from the cell the bot stood on when it last looked, so recording a *failed*
            # look locks out every look from the eight cells around it - and the one cell that has
            # to see the gate is the one the bot walks to after `_settle_home_anchor` drops its
            # guess. Live, run59: the run started with no base and no chest, so `_walk_home_to_memory`
            # named the remembered chest's cell, the bot walked there, the guess was retired, and
            # `_adopt_portal_base` then declined to look again - the throttle remembered the cell
            # from the failed scan at spawn, four metres away. So the lit gate and the chest at
            # `(226,63,222)` were never re-adopted, and `establish_base` raised a second base
            # `(229,62,226)`, eight metres from the run's own burning portal.
            #
            # Recorded on success only, which is also when the throttle earns its keep: a base that
            # was adopted is set, and line 593 short-circuits the whole method. What the throttle
            # still prevents is a second look while the bot stands on a cell it has already read,
            # and a scan that finds nothing costs 2 ms.
            return
        self._portal_base_scanned = (px, pz)
        self.base = (corner[0] - 2, corner[1] - 1, corner[2])
        self._base_adopted = True

    async def _scan_portal_opening(self, gs) -> None:
        """Read the portal opening while a frame build is pending, so a stray block is dug.

        The base scan (`flat_build_site`) only proves clearance for a site *it* chose; a base the
        loop adopted from a standing bed, or one taken from the `base_target` fallback, never got
        that check. A block left in the opening above the support layer would be built around and
        `activate_portal` would answer `[PORTAL_NO_FRAME]` forever. One cheap `view_blocks cells`
        read of the six opening cells fills `gs.portal_blocked` for `clear_portal`. Gated on the
        flint and steel being carried and the portal unlit, which is exactly the build window.
        """
        if self.base is None or gs.pos is None or "flint_and_steel" not in gs.inventory:
            return
        if gs.resource_nearest("nether_portal") is not None:
            return
        try:
            names = perception.parse_cells(
                await perception.scan_frame_volume(self.mcp, self.base))
        except Exception:
            return
        # Every cell of the frame plane, for `clear_frame`: a border cell the build refused needs
        # its block name to be dug with the explicit `collect_block` (`_note_frame_cells`).
        self._frame_names = names
        if any(name == perception.PORTAL_BLOCK for name in names.values()):
            # A base re-adopted from a standing portal (`_adopt_portal_base`): the opening already
            # holds the lit portal's own blocks, which are not a stray block to dig.
            self._portal_blocked = []
            return
        self._portal_blocked = [cell for cell in perception.portal_interior(self.base)
                                if not _clear(names.get(cell))]

    def _adopt_homebase(self, gs) -> None:
        """Re-adopt a home base and chest a previous run left standing in the world.

        `base` and `chest_pos` are per-run state, so a restart loses them - but the bed and the
        chest are real blocks. When the bot stands at a placed bed whose ground can carry the
        frame (`_scan_bed_site` proved it, `targets.bed_frame_site`), that bed's ground cell becomes
        the base again, and a chest a few blocks away is remembered with it, so the home-base steps
        (sleep, stow) keep working at the old base instead of building a second one. A bed whose
        site fails the check -- laid underground or on a slope -- is left alone, so the run anchors
        a base the frame can actually stand on instead.
        """
        if self.base is None and self._bed_base is not None:
            self.base = self._bed_base
        if self.base is not None:
            # Every chest the network owns sits within reach of the base it belongs to -- the main
            # one at `(bx, by+1, bz+1)` and each overflow in `STORAGE_SITES` fall inside
            # `ADOPT_CHEST_RADIUS`, and adoption below only ever takes a chest that close. A cell
            # that does not is a leftover of a base this run has moved off: `_walk_home_to_memory`
            # seeds `chest_pos` from the server's cross-run memory while the run has no base, and
            # when that remembered home proves unreachable (`_settle_home_anchor` gives it up) and
            # `establish_base` raises a fresh base elsewhere, the inherited chest is stranded. Kept,
            # it makes `_chest_ready` true at the new base, so `place_chest`/`craft_chest` is never
            # offered there and `stow_surplus` walks the pack back to a phantom cell -- run100
            # it67-88 chased a stale (265,72,207) 322 blocks from its (-57,70,44) base until a
            # deposit finally met `[CONTAINER_NOT_PRESENT]` there. Drop the stranded cells so the
            # new base builds and works its own.
            bx, by, bz = self.base
            network = ([tuple(self.chest_pos)] if self.chest_pos is not None else []) \
                + [tuple(c) for c in self._extra_chests]
            for cx, cy, cz in network:
                if (abs(cx - bx) > ADOPT_CHEST_RADIUS or abs(cz - bz) > ADOPT_CHEST_RADIUS
                        or abs(cy - (by + 1)) > ADOPT_CHEST_RADIUS):
                    self._forget_chest_cell((cx, cy, cz))
        if self.chest_pos is None and self.base is not None:
            bx, by, bz = self.base
            for hit in gs.resources.get("chest", []):
                if (abs(hit.x - bx) <= ADOPT_CHEST_RADIUS
                        and abs(hit.z - bz) <= ADOPT_CHEST_RADIUS
                        and abs(hit.y - (by + 1)) <= ADOPT_CHEST_RADIUS):
                    self.chest_pos = hit.pos
                    # A chest adopted at the base after the stock read already ran (the bot
                    # spawned on its base) joins the network too late for this tick's read. Clear
                    # the throttle so the next `_note_stocked_chest` re-reads the now-owned cell
                    # and `_adopt_powder_stow` can latch from a chest the bot never walked up to.
                    self._stock_read = None
                    break
        elif self.chest_pos is not None and self.base is not None:
            # The restart's network: the chest scan sees every chest near the base, not the main
            # one only. A network chest left standing would otherwise go blind -- its provisions
            # stay in the server's database but behind a cell the loop no longer owns, and the
            # filtered stock read stops seeing what it owned yesterday.
            bx, by, bz = self.base
            for hit in gs.resources.get("chest", []):
                if (abs(hit.x - bx) <= ADOPT_CHEST_RADIUS
                        and abs(hit.z - bz) <= ADOPT_CHEST_RADIUS
                        and abs(hit.y - (by + 1)) <= ADOPT_CHEST_RADIUS
                        and hit.pos != self.chest_pos and hit.pos not in self._extra_chests):
                    self._extra_chests.append(hit.pos)
                    self._stock_read = None
        if self.smoker_pos is None and self.base is not None:
            bx, by, bz = self.base
            for hit in gs.resources.get("smoker", []):
                if (abs(hit.x - bx) <= ADOPT_CHEST_RADIUS
                        and abs(hit.z - bz) <= ADOPT_CHEST_RADIUS
                        and abs(hit.y - (by + 1)) <= ADOPT_CHEST_RADIUS):
                    self.smoker_pos = hit.pos
                    break
        if self.furnace_pos is None and self.base is not None and not self._furnace_blocked:
            bx, by, bz = self.base
            for hit in gs.resources.get("furnace", []):
                if (abs(hit.x - bx) <= ADOPT_CHEST_RADIUS
                        and abs(hit.z - bz) <= ADOPT_CHEST_RADIUS
                        and abs(hit.y - (by + 1)) <= ADOPT_CHEST_RADIUS):
                    self.furnace_pos = hit.pos
                    break

    async def _perceive(self, state: AgentState) -> dict:
        gs = await self._observe()
        return {"iteration": 0, "before": gs, "done": False, "note": "", "escalated": False,
                "deferred": False}

    async def _decide(self, state: AgentState) -> dict:
        gs = state["before"]
        decision = self.brain.decide(gs.narrative(), candidates(gs), gs)
        if not decision.deferred and decision.action_label in TARGET_LABELS:
            # M4 parameter layer: turn "where?" into a Laya choice over grounded candidates.
            targets = ground_targets(gs)
            if targets:
                target, meta = self.brain.choose_target(gs.narrative(), targets)
                decision.target = target
                decision.target_meta = meta
        return {"decision": decision, "escalated": False, "deferred": False}

    async def _guard(self, state: AgentState) -> dict:
        gs = state["before"]
        decision = state["decision"]
        # `_blocked_label is not None` matters: a deferral carries `action_label = None`, and
        # `None == None` would otherwise inject an unblock step instead of letting the
        # server-side survival policy own that tick.
        if self._blocked_label is not None and self._blocked_label == decision.action_label:
            if self._unblock_done:
                # The surroundings were changed for this label (`make_room`, then `explore`);
                # let the label itself run now, otherwise the guard intercepts it on every tick
                # and it can never succeed. It only clears in `_note_block` if it succeeds; a
                # fresh failure takes the next rung of the ladder (pocket -> explore).
                self._unblock_done = False
            elif self._blocked_kind == "inventory":
                # The pack was full. Make room (deposit at the base chest, else drop the
                # surplus) and let the blocked label run again next tick: unlike a terrain
                # fix, this step removes the cause itself, so the label must not stay blocked.
                action = inventory_relief(gs, allow_stow=not self._stow_blocked)
                if action is not None:
                    self._blocked_label = None
                    self._unblock_attempts = 0
                    self.log.append({"stage": "policy", "reason": "unblock_inventory",
                                     "blocked": decision.action_label, "via": action.label})
                    return {"action": action, "note": f"unblock -> {action.label}"}
            else:
                # The same label failed because the surroundings blocked it. Clear a pocket
                # (or, on a second try, move to open ground) first, then let the label retry.
                action = unblock_action(gs, self._unblock_attempts)
                if action is not None:
                    self._unblock_attempts += 1
                    self._unblock_done = True
                    self.log.append({"stage": "policy", "reason": "unblock_surroundings",
                                     "blocked": decision.action_label, "via": action.label})
                    return {"action": action, "note": f"unblock -> {action.label}"}
        action = resolve(decision.action_label, gs, decision.target)
        if action is not None:
            return {"action": action}
        if decision.deferred:
            # In danger or nothing valid: the server-side survival policy owns this tick.
            return {"action": None, "deferred": True, "note": decision.note}
        plan = escalate_to_system2(
            self.settings.llm, self.settings.api_key(), gs.as_state(), self.history,
            {"triggers": ["unresolved_label"], "label": decision.action_label},
            list(candidates(gs)),
        )
        label = (plan or {}).get("action")
        alt = resolve(label, gs) if label else None
        self.log.append({"stage": "escalation", "label": decision.action_label,
                         "plan": plan, "chosen": label})
        if alt is not None:
            return {"action": alt, "escalated": True, "note": f"escalated -> {label}"}
        return {"action": resolve(FALLBACK_LABEL, gs), "escalated": True,
                "note": f"fallback -> {FALLBACK_LABEL}"}

    async def _execute(self, state: AgentState) -> dict:
        action = state["action"]
        # A hostile hunt runs in the aggressive hunt stage; the base combat floor
        # would otherwise block the engagement ([HUNT_CAPABILITY_BLOCKED]). Restore
        # immediately afterwards: the zero floor would otherwise hold the body and
        # block navigation until the next policy change.
        #
        # `leave_nether` is the opposite case and gets the opposite stage: it is a `navigate`
        # the model issued to cross home with the haul done, and the base floor keeps the reflex
        # fighting whatever is in reach - at a blaze spawner that is an endless fight that holds
        # the body and never lets the walk start (run106: two held `leave_nether` ticks, 300 s
        # each, ended the run on `body_held` with twelve rods aboard). The flee stage raises the
        # pursuit floor past full health so the reflex evades instead of fighting and the walk
        # home runs. Restored to base the same way, the moment the crossing settles.
        hunting = is_hostile_hunt(action)
        fleeing = action.label == "leave_nether"
        if hunting:
            await survival.apply_hunt(self.mcp, f"open the hunt stage for {action.label}")
            self.log.append({"stage": "policy", "reason": "hunt_stage_on",
                             "label": action.label})
        elif fleeing:
            await survival.apply_flee(self.mcp, f"open the flee stage for {action.label}")
            self.log.append({"stage": "policy", "reason": "flee_stage_on",
                             "label": action.label})
        try:
            res = await self.mcp.call(action.tool, action.args, action.rationale)
        finally:
            if hunting or fleeing:
                await survival.restore_base(
                    self.mcp, f"restore the base stage after {action.label}")
        return {"result": res}

    async def _defer(self, state: AgentState) -> dict:
        decision = state["decision"]
        self.log.append({
            "stage": "defer",
            "iteration": state.get("iteration", 0),
            "state": state["before"].as_state(),
            "narrative": state["before"].narrative(),
            "candidates": decision.candidates,
            "note": decision.note,
            "answers": _chosen(getattr(decision, "answers", {})),
        })
        self.history.append({"stage": "defer", "note": decision.note})
        iteration = state.get("iteration", 0) + 1
        gs = state["before"]
        stop = self._stop_state(gs, iteration)
        if not stop["done"]:
            await asyncio.sleep(self.settings.mcp.poll_ms / 1000)
            gs = await self._observe()
            # A death in a defer tick (the reflex's fight lost) needs the policy back too.
            await self._reapply_after_death(gs)
            self._note_defer(state["before"], gs)
            gs.defer_quiet = self._defer_quiet
        return {"before": gs, "iteration": iteration, **stop}

    def _note_defer(self, before, after) -> None:
        """Count the defers in which the survival policy had nothing to do.

        A defer hands the tick to the server's reflexes; one that took no damage and in which
        no reflex reported acting (`SURVIVAL_EVENT_PREFIXES`) handed it to nobody. A hit or a
        reflex outcome starts the count over (see `brain.DEFER_QUIET_TICKS`).
        """
        hurt = (before.health is not None and after.health is not None
                and after.health < before.health)
        acted = any(str(e).lower().startswith(SURVIVAL_EVENT_PREFIXES) for e in after.events)
        self._defer_quiet = 0 if (hurt or acted) else self._defer_quiet + 1

    async def _hold_wait(self, action, ok: bool) -> None:
        """Let a successful `wait` pass real time before the next decision.

        `wait` is a navigate to the anchor the bot already stands on, so it settles at once and
        every tick of it was about a second: run86 lit its gate by day and burned ~300 of its 600
        iterations at the base before the night came. Holding `[loop] wait_hold_s` per tick makes
        a ten-minute day some twenty decisions. The server's reflexes guard the body meanwhile,
        as they do during any long action, and the hold ends on the next observation.
        """
        if action.label != "wait" or not ok:
            return
        hold = float(self.settings.loop.wait_hold_s or 0)
        if hold > 0:
            await asyncio.sleep(hold)

    def _refresh_after(self, gs) -> None:
        """Hand the facts learned after the observation to the state the next tick decides on.

        `_verify` observes first and then reads the result into the loop's own facts, but the
        observation already carried the old ones (`_inject`), and the next `decide` uses that very
        state. So every fact learned here acted one tick late: run87 sent each `surface_up` probe
        twice (`[NAVIGATION_HEIGHT_REQUIRED]` at it58 and it59, it64/65, it68/69) because the
        height the first refusal named only reached the state of the tick after the second, and a
        withheld label still owned the tick it was withheld on. Only plain copies here - `_inject`
        also counts ticks (`water_ticks`) and must run once per observation.
        """
        gs.surface = self.surface
        gs.frame_complete = self._frame_complete
        gs.powder_stowed = self._powder_stowed
        gs.end_portal_active = self._end_portal_active
        gs.stronghold_estimated = self._stronghold_estimated
        gs.powder_exhausted = self._powder_exhausted
        gs.powder_cells = tuple(self._powder_cells)
        gs.nether_landmarks = frozenset(self._nether_landmarks)
        gs.stuck_labels = frozenset(self._stuck_off)
        gs.defer_quiet = self._defer_quiet
        # `_note_boxed` runs after this tick's observation (it needs the post-action cell), so
        # the pocket it just learned about would otherwise only reach the state of the tick
        # after. One tick is the whole difference between a tunnel and suffocating (run95).
        gs.boxed_in = self._boxed[1] >= BOXED_REPEATS
        gs.dig_out_try = self._dig_out_tries

    def _note_hurt(self, before, after) -> None:
        """A hit while the loop works hands the next contact back to the survival policy."""
        if (before.health is not None and after.health is not None
                and after.health < before.health):
            self._defer_quiet = 0

    def _goal_reached(self, gs) -> bool:
        """Whether the run is finished: every phase, not the first one that closes.

        `tech_tree_complete` is the *first* phase and it closes one tick before the crossing it
        prepares: lit portal, gear worn, bed at the base, prep banked, food reserve met - and
        `enter_portal` sitting in the offer right behind it. Read as a goal on its own it ended
        every run exactly there, which is why the Nether only ever ran from a wrecked state: the
        goal had to be *broken* (a death, lost gear, a re-based home) to keep the loop going
        (live, run29: `{"iterations": 180, "stop": "goal"}` with the expedition never started, and
        runs 42-47 crossing only because run43's death had already undone the gate).

        So the phases are an AND, not an OR. The order of play is Overworld, then the Nether trip,
        then whatever comes after it; a run that stops on its first finished phase is not a
        playthrough. The asymmetry is deliberate and worth keeping: `nether_expedition_complete`
        reads a latch that only ever sets (`gs.powder_stowed`), while `tech_tree_complete` reads
        the live world and can go false again - a creepers' bed, a dropped tool. So the run may
        continue past the deposit until the home base is in order too, and it will stop on the
        first tick where both hold. The cost is that a run which never finds a fortress no longer
        stops early; it runs to the iteration limit and says `"stop": "limit"`, which is the
        honest report for an unfinished playthrough. The stronghold phase joins the AND the same
        way (docs/stronghold.md): its latch (`gs.end_portal_active`) only ever sets too.
        """
        return (bool(self.settings.loop.stop_when_complete)
                and tech_tree_complete(gs) and nether_expedition_complete(gs)
                and stronghold_complete(gs))

    def _stop_state(self, gs, iteration: int) -> dict:
        """`done`/`stop` for the next state: the goal ends the run early, else the limit does -
        or a body the server's reflexes will not hand back (`BODY_HELD_STOP`)."""
        goal = self._goal_reached(gs)
        limit = iteration >= self.settings.loop.max_iterations
        held = self._held_ticks >= BODY_HELD_STOP
        return {"done": goal or limit or held,
                "stop": "goal" if goal else ("body_held" if held else ("limit" if limit else ""))}

    def _note_held(self, res, before, after) -> None:
        """Count the foreground calls in a row that ran out of wait budget with the body held.

        A defer between them does not reset the count - it calls nothing and so learns nothing
        about the body - only an action whose result came back does.

        But a held tick that moved the fight forward is not run89's dead standoff (three hours
        against one zombie 1.6 m off, nothing changing): the reflex is winning the ground it holds.
        A hold that healed the bot, brought in a blaze rod, or thinned the pack is progress, so it
        breaks the streak rather than counting toward the stop - run106's holds ran HP 7->13->20
        and rods 10->12 while a blaze spawner was being cleared, yet `BODY_HELD_STOP` ended the run
        as though the body were simply stuck. Only a hold that moved none of the three - the true
        deadlock - is counted.
        """
        if getattr(res, "status", None) != "held":
            self._held_ticks = 0
            return
        healed = (after.health or 0) > (before.health or 0)
        rods = after.inventory.get(BLAZE_ROD_ITEM, 0) > before.inventory.get(BLAZE_ROD_ITEM, 0)
        thinned = _hostile_count(after) < _hostile_count(before)
        self._held_ticks = 0 if (healed or rods or thinned) else self._held_ticks + 1

    def _note_block(self, label: str, ok: bool, text: str | None) -> None:
        """Update the surroundings-block bookkeeping from one action's result.

        The blocked label is cleared only once *it* finally succeeds. A successful unblock
        (`make_room`) must not clear it: otherwise the stage-2 `explore` escape is unreachable
        and `make_room` retries forever (live: `sleep` in a 1x1 shaft alternated
        sleep/make_room for the rest of the run, and the shield run burned 12 of 16 ticks on it).
        Balance is kept by `_unblock_done`: the guard lets the label run exactly once per
        unblock step, so the pocket gets its retry while the ladder still reaches `explore`.
        """
        if label == self._blocked_label:
            if ok:
                self._blocked_label = None
                self._unblock_attempts = 0
                self._unblock_done = False
            else:
                # It failed again right after its unblock step: the guard takes the next rung
                # of the ladder (pocket -> explore) on the following tick. The count is kept.
                self._unblock_done = False
        elif not ok and blocked_by_inventory(text):
            # A full pack is not terrain: the unblock makes room (stow at the base, else drop),
            # and that frees the label to run again on the next tick.
            self._blocked_label = label
            self._blocked_kind = "inventory"
            self._unblock_attempts = 0
            self._unblock_done = False
        elif not ok and blocked_by_surroundings(text):
            self._blocked_label = label
            self._blocked_kind = "surroundings"
            self._unblock_attempts = 0
            self._unblock_done = False

    def _note_pour(self, action, ok: bool, text: str | None) -> None:
        """Remember a lava cell a pour was refused at, so it is never poured at again.

        `[BUCKET_NO_FACE]` says no solid block beside that cell faces the bot: a property of
        the cell and the terrain around it, not of the attempt. The obsidian step reads the set
        back out of the GameState (`pour_skip`) and looks for the next lava source, which is
        what carries the run from a stopped-up surface pond to the lava lake further off.
        """
        args = action.args or {}
        if ok or action.tool != "use_bucket" or args.get("action") != "pour":
            return
        if "BUCKET_NO_FACE" not in (text or ""):
            return
        pos = (args.get("x"), args.get("y"), args.get("z"))
        if all(isinstance(v, int) for v in pos):
            self._pour_skip.add(pos)

    def _note_stance(self, action, ok: bool, text: str | None) -> None:
        """Remember a pour stance the server refused, so it is never walked to again.

        `[NAVIGATION_TARGET_UNSUPPORTED]` on the `mine_obsidian` walk says the server found no
        footing at that cell: the cell over the lava's neighbour is air, or the lake below is too
        deep to stand on (live, run16: one stance over a deep lava lake answered the identical
        refusal on 215 ticks and the run never made a single obsidian). It is a property of the
        stance cell, not of the attempt, so the step reads the set back out of the GameState
        (`stance_skip`), stands on another neighbour, and once every stance of that source is
        refused `_lava_pour_target` passes over the source itself -- the move `pour_skip` makes
        for a refused pour.
        """
        if ok or action.label != "mine_obsidian" or action.tool != "navigate":
            return
        if "NAVIGATION_TARGET_UNSUPPORTED" not in (text or ""):
            return
        args = action.args or {}
        pos = (args.get("x"), args.get("y"), args.get("z"))
        if all(isinstance(v, int) for v in pos):
            self._stance_skip.add(pos)

    def _track_stuck(self, label: str, ok: bool, text: str | None,
                     progressed: bool) -> None:
        """Withhold a label that keeps answering the identical refusal without progress.

        See `STUCK_REPEATS`/`STUCK_COOLDOWN`. Any success, any `progressed` tick, or a *different*
        refusal code starts that label's run of identical ones over, so only a genuine standstill
        is caught. The window is a countdown, not a lock: the label returns on its own, which keeps
        the offer honest and lets the loop retry it after it has moved somewhere new.

        Each label is counted on its own. The window of one is ticked down here first, so it never
        shields another: previously a single withheld label made the loop return early for the
        whole window, and a second label refusing on every tick it got went uncounted (live, run17:
        `build_portal` withheld for 15 ticks, `light_portal` answering `[PORTAL_NO_FRAME]` 150 times
        beside it and never reaching the threshold).
        """
        for held in list(self._stuck_off):
            self._stuck_off[held] -= 1
            if self._stuck_off[held] <= 0:
                del self._stuck_off[held]
                self.log.append({"stage": "policy", "reason": "stuck_offer_restored",
                                 "label": held})
        if ok or progressed:
            self._stuck_reps.pop(label, None)
            self._stuck_code.pop(label, None)
            return
        code = _fail_code(text)
        if code is None:
            return
        if label in self._stuck_off:
            # Withheld right now: the offer left it out, so this tick is not evidence against it.
            return
        if self._stuck_code.get(label) == code:
            self._stuck_reps[label] = self._stuck_reps.get(label, 0) + 1
        else:
            self._stuck_code[label] = code
            self._stuck_reps[label] = 1
        if self._stuck_reps[label] >= STUCK_REPEATS:
            self._stuck_reps[label] = 0
            self._stuck_off[label] = STUCK_COOLDOWN
            self.log.append({"stage": "policy", "reason": "stuck_offer_dropped",
                             "label": label, "code": code, "ticks": STUCK_COOLDOWN})

    def _track_wander(self, label: str, res) -> None:
        """Withhold a label that keeps walking without arriving, per `WANDER_*`.

        The server reports each action's own progress: `distanceTravelledBlocks` is how far the bot
        walked for it, `distanceFromStartBlocks` how far from where it started it ended. A label
        whose ratio of the first to the second is hopeless is not working, whatever it answers -
        and it *does* answer, which is why `_track_stuck` never saw it: it has no failure code to
        read, and the position changes every tick, so nothing looked stuck. Live, run43 and run45:
        one `collect_block` walked 458.9 blocks to end 16.6 from its start (27:1), another 245.5
        to end 24.2, and each took minutes, so the run managed six decisions in ten minutes.

        The window is the same countdown as `_track_stuck`, so the label returns on its own and a
        walk that merely crossed rough ground is not lost. Progress of any kind -- a different
        ratio, a short walk, no progress reported at all -- starts that label's run over.

        The numbers are read from the structured payload where it carries them and from the result
        text otherwise, because the server prints both: `formatProgress` writes `**Travel:**` and
        `**Distance from start:**` into every settled result, so the guard does not depend on the
        payload keeping its shape. "unavailable" is no evidence, and neither is a missing line.
        """
        travelled, net = _walk_progress(res)
        if travelled is None or net is None:
            return
        if label in self._stuck_off:
            # Withheld right now: it was not offered, so its tick says nothing about the label.
            return
        if travelled < WANDER_WALK or net >= travelled / WANDER_RATIO:
            self._wander_reps.pop(label, None)
            return
        self._wander_reps[label] = self._wander_reps.get(label, 0) + 1
        if self._wander_reps[label] < WANDER_REPEATS:
            return
        self._wander_reps[label] = 0
        self._stuck_off[label] = STUCK_COOLDOWN
        self.log.append({"stage": "policy", "reason": "wander_offer_dropped",
                         "label": label, "walked": round(float(travelled), 1),
                         "net": round(float(net), 1), "ticks": STUCK_COOLDOWN})

    def _track_idle(self, label: str, delta: dict) -> None:
        """Withhold a label that keeps succeeding at nothing (`IDLE_REPEATS`/`PASS_TIME_LABELS`).

        `_track_stuck` counts refusals and `_track_wander` counts walking that arrives nowhere.
        Neither can see a step the server reports as *succeeded* while the world does not change at
        all, so nothing ever accumulates: live, run54 - the bot standing in the Nether's arrival
        portal, where the rescue owns the tick outright, and `leave_portal` answered ok 299 times in
        four minutes without moving a centimetre, because the target cell is one step out and the
        call succeeds where it stands. 299 iterations, none of them refusals, none of them walking.

        "Nothing changed" is read from the whole tick, not from `progressed`: no item gained or
        lost, nothing worn or held differently, no food, no movement, no phase and no dimension. A
        short walk is progress even when it is under the two blocks `place_bed` counts, so the
        position is compared exactly.

        The steps whose whole purpose is to pass time are exempt, because a lull really does change
        nothing and dropping them would end the run's waiting with nothing to do instead. Everything
        else that succeeds and achieves nothing is a step that is not doing its job, and the window
        is the same countdown as the other two guards so it returns on its own.
        """
        if label in PASS_TIME_LABELS:
            self._idle_reps.pop(label, None)
            return
        if label in self._stuck_off:
            return
        changed = (any(delta.get("inventory_delta") or {})
                   or delta.get("armor_delta") or delta.get("offhand_delta")
                   or (delta.get("food_delta") or 0) != 0
                   or (delta.get("pos_delta") or 0) != 0
                   or delta.get("phase_changed")
                   or delta.get("dimension_changed"))
        if changed:
            self._idle_reps.pop(label, None)
            return
        self._idle_reps[label] = self._idle_reps.get(label, 0) + 1
        if self._idle_reps[label] < IDLE_REPEATS:
            return
        self._idle_reps[label] = 0
        self._stuck_off[label] = STUCK_COOLDOWN
        self.log.append({"stage": "policy", "reason": "idle_offer_dropped",
                         "label": label, "ticks": IDLE_REPEATS,
                         "window": STUCK_COOLDOWN})

    def _note_surface(self, action, ok: bool, text: str | None,
                      pos: tuple[float, float, float] | None) -> None:
        """Learn the surface cell from a `surface_up` probe or climb.

        A height-less `navigate` to an ambiguous column stops with
        `[NAVIGATION_HEIGHT_REQUIRED]` and names the surface (`perception.parse_surface_height`);
        an unambiguous one walks the bot to open sky, so its own new feet cell is the surface.
        Either answer feeds `_at_surface`, which is what lets `establish_base` run -- the base
        may only be anchored on open-sky ground, and a mined tunnel's `air` overhead cannot be
        told from sky by any `view_blocks` reading (live, run9: the base landed at y=-16 under a
        y=73 surface). A failed *climb* (the request did carry a height) keeps the cell as it
        was, so the next tick simply retries it.
        """
        if action.label != "surface_up":
            return
        # Never anchor the base to a surface learned while the bot is in water or floating.
        gs = getattr(self, "gs", None)
        if gs is not None and getattr(gs, "in_water", False):
            return
        if ok and pos is not None:
            if gs is not None and getattr(gs, "in_water", False):
                return
            self.surface = (math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]))
            return
        if "y" in (action.args or {}):
            return
        y = perception.parse_surface_height(text or "")
        if y is not None:
            # Don't learn a surface from a water column - the highest standable feet
            # might be the seabed, not open ground suitable for a base.
            if gs is not None and getattr(gs, "in_water", False):
                return
            # Extra guard: if the parsed surface is way below the bot's current level
            # and bot is in/near water, it's likely seabed
            if gs is not None and gs.pos is not None and getattr(gs, "in_water", False) is False:
                # If bot is in water state or near water, be cautious
                pass
            self.surface = (int(action.args["x"]), y, int(action.args["z"]))

    def _note_crossing(self, action) -> None:
        """Stand the `_in_portal` rescue down around a crossing the loop asked for.

        The rescue exists to step a body out of a gate it wandered into. The intended crossing is
        the one case where that undoes the plan, so the tick a crossing label is chosen opens a
        short window (`_enter_intent_ticks`) in which `_in_portal` stands down - long enough for
        the action to settle and the teleport to fire, short enough that the next accidental gate
        still gets rescued.
        """
        if action.label in ("enter_portal", "leave_nether"):
            self._enter_intent_ticks = PORTAL_INTENT_TICKS
            # The junk left on open ground is in the other dimension now: the walk-back belongs to
            # the cell it landed in, and offering it from across a gate would aim the bot at
            # coordinates that mean nothing here.
            self._drop_open_pending = False
            self._drop_pos = None
            self._step_away_fails = 0

    def _note_stow(self, action, ok: bool, delta: dict) -> None:
        """Remember that the blaze powder reached the chest at the portal (the phase-3 goal).

        Latched on the deposit's own success and on the pack actually losing the powder, so a
        refused deposit (full chest, lid blocked) leaves the goal open and the step in the offer.
        """
        if action.label != "stow_blaze_powder" or not ok:
            return
        if (delta.get("inventory_delta", {}) or {}).get("blaze_powder", 0) < 0:
            self._powder_stowed = True
            # A fresh stow refills the chest a stronghold withdraw had found empty: the
            # second-Nether top-up is done and `take_blaze_powder` may offer again.
            self._powder_exhausted = False
            # Remember where the powder actually went: `stow_blaze_powder` deposits into the
            # network's roomy chest, which may be an overflow chest rather than `chest_pos`.
            # The withdraw has to look in the same cell, or it reads an empty main chest and
            # sends the bot back through the gate for powder that is sitting a few blocks away.
            args = action.args or {}
            cell = (args.get("x"), args.get("y"), args.get("z"))
            if None not in cell and cell not in self._powder_cells:
                self._powder_cells.append(cell)

    def _note_take_powder(self, action, ok: bool, text: str | None, delta: dict) -> None:
        """Remember that the gate chest holds no more blaze powder for the eyes.

        Only a withdraw that reached the chest counts: one that succeeded, or one the server
        answered with the chest's own shortfall (`[NO_MATCHING_ITEMS]`,
        `[CONTAINER_BATCH_INCOMPLETE]`). If that withdraw put no powder in the pack, the bank is
        empty and the eyes still owed can only be paid for by a second Nether trip
        (`predicates._nether_topup_wanted`). A walk to the chest or a fight on the way says nothing
        about what is inside it.
        """
        if action.label != "take_blaze_powder" or action.tool != "use_container":
            return
        reached = ok or any(code in (text or "")
                            for code in ("NO_MATCHING_ITEMS", "CONTAINER_BATCH_INCOMPLETE"))
        if not reached:
            return
        if (delta.get("inventory_delta", {}) or {}).get(BLAZE_POWDER_ITEM, 0) <= 0:
            # The cell the withdraw stood at is out of powder; the bank is only truly empty
            # when every cell a stow remembered has come up empty. A top-up aimed at the main
            # chest while the powder sits in an overflow chest is a walk to the wrong lid, not
            # a lost haul.
            args = action.args or {}
            cell = (args.get("x"), args.get("y"), args.get("z"))
            if None not in cell:
                self._powder_cells = [c for c in self._powder_cells if c != cell]
            if not self._powder_cells:
                self._powder_exhausted = True

    def _note_barter(self, action, ok: bool) -> None:
        """Count a piglin barter the run actually made.

        The cap (`BARTER_MAX_ATTEMPTS`) is what keeps the gold reserve from becoming a permanent
        gold farm: a piglin rolls once per ingot and pearls are the rare entry of that roll, so
        every further reserve digs gold for a trade that keeps missing. Only a call the server
        accepted counts - a refusal (no loaded piglin, a baby) is not an attempt, and the stuck
        guard already cools that one down.
        """
        if action.label == "barter_pearls" and ok:
            self._barter_attempts += 1

    def _note_locate(self, action, ok: bool) -> None:
        """Remember that the eye-of-ender estimate is saved, so the next locate travels.

        `locate_stronghold` keeps the bearing under the durable `search_id` on the server; the
        loop only needs to know the `estimate` phase succeeded once to send `locate` next.
        """
        if action.label == "locate_stronghold" and ok and (action.args or {}).get("phase") == "estimate":
            self._stronghold_estimated = True

    def _note_portal(self, action, ok: bool) -> None:
        """Remember that the end portal stands active (the stronghold goal).

        `activate_portal` confirms every interior portal block before it answers ok, so its own
        success is the latch, the same shape as `_note_stow`; a refused fill leaves the goal open.
        """
        if action.label == "activate_end_portal" and ok:
            self._end_portal_active = True

    def _note_build(self, action, ok: bool, delta: dict) -> None:
        """Remember that the frame build has nothing left to place.

        A `build_portal` the server answers `ok` to while it spends no obsidian and no
        cobblestone has placed every cell it can: the frame already stands complete, and
        re-sending the same structure is a no-op the server still calls a success. That success
        resets `_track_stuck`, so `candidates`' "resume a partial frame" clause would keep the
        step in the offer for as long as a leftover block sits in the pack, and a policy that
        keeps choosing it freezes the run at the base (live, run25: 163 identical `build_portal`
        ticks on a finished frame, the bot standing at its own gate the whole time). Marked here
        so `candidates` and `resolve` drop the step for the rest of the run.
        """
        if action.label != "build_portal" or not ok:
            return
        if not any(delta.get("inventory_delta", {}).values()):
            self._frame_complete = True

    def _note_frame_cells(self, action, ok: bool, text: str | None) -> None:
        """Remember the frame cells the build refused, so `clear_frame` digs them explicitly.

        `build_structure` removes a wrong block in a frame cell by digging it in place, but the
        movement policy refuses a break it was not asked for, and a *container* always needs that
        explicit request: `[BUILD_INCOMPLETE] ... 1 were refused (229,62,222: movement policy
        prohibits breaking this block).` The build then fails on every attempt, the frame never
        closes, and `light_portal` never runs (live, run27: run26's furnace stood on the new base's
        frame corner). The refused cells are remembered here and `clear_frame` digs each with the
        explicit `collect_block` the policy does allow; a finished build clears the whole list, and
        `clear_frame` drops just the cell it broke.

        A `[BUILD_STOPPED]` names no cell ("2 passes changed nothing; last route completed"), so
        parsing it remembers nothing -- live, run80, 350 `clear_portal` ticks learned nothing:
        the route dig and the build stopped on blocks the route could not change, and the frame
        closed only at it364, too late in the day for the run's last hour. What the frame scan
        already holds stands in for the missing answer: the blocked opening cells, plus every
        frame ring cell the plane read has named a wrong block (a tree in an obsidian cell,
        terrain in a cobblestone corner). `clear_frame` digs those by name -- the explicit break
        is the one the movement policy allows -- and the build resumes on what the pack carries.
        """
        if action.label == "clear_frame":
            if ok and {"x", "y", "z"} <= set(action.args):
                self._refused_frame.discard((int(action.args["x"]), int(action.args["y"]),
                                             int(action.args["z"])))
            return
        if action.label not in ("build_portal", "clear_portal"):
            return
        if ok:
            self._refused_frame.clear()
            return
        self._refused_frame.update(perception.parse_refused_cells(text))
        if "[BUILD_STOPPED]" in (text or ""):
            blocked = set(self._portal_blocked)
            if self.base is not None:
                wanted = perception.frame_wanted_blocks(self.base)
                blocked |= {cell for cell, name in self._frame_names.items()
                            if cell in wanted and name and not _clear(name)
                            and wanted[cell] != name}
            self._refused_frame.update(blocked)

    def _note_base(self, action, ok: bool, text: str | None) -> None:
        """Repair a home base whose anchor cell has lost its footing.

        `go_home`, `wait`, `place_chest` and `place_furnace` all navigate to `base.y + 1`, so a
        base whose own ground has been dug out answers `[NAVIGATION_TARGET_UNSUPPORTED]` on every
        tick. The anchor is recorded once at `establish_base`; nothing re-checks it against the
        live world, so a base the bot later quarries (or a creeper blows, or sand falls from) spins
        the loop forever on a target that cannot be stood on. Live, run11: after `establish_base`
        the bot excavated its own base to open air, and the next 230 `go_home` ticks all answered
        the identical refusal while the run never advanced.

        The refusal names the ground the server *did* find in that column, so the anchor is simply
        re-pinned to it -- the same "read the height back out of the server's answer" move as
        `_note_surface`. If the report no longer moves the anchor (the same height it already
        holds), the site is not repairable in place, so the base is dropped and the loop anchors a
        fresh one from a new `flat_build_site` scan. Both paths are bounded: one tick to heal, one
        to reset, never the unbroken retry.
        """
        if ok or self.base is None:
            return
        if "NAVIGATION_TARGET_UNSUPPORTED" not in (text or ""):
            return
        args = getattr(action, "args", None) or {}
        bx, by, bz = self.base
        # Only a step aimed at the base anchor itself can repair or invalidate it; a refusal from
        # any other destination is a different problem.
        if args.get("x") != bx or args.get("z") != bz:
            return
        y = perception.parse_ground_height(text or "")
        if y is None:
            return
        if y == by:
            # The server reports the anchor's own height as still unsupported: the site is gone.
            # Drop it so the next tick re-establishes a base on a fresh scan.
            self.base = None
            self.build_site = None
            self._build_site_scanned = None
            return
        self.base = (bx, y, bz)

    def _note_drop(self, action, ok: bool, text: str | None, gs=None) -> None:
        """Remember where a `drop_junk` left its junk on open ground, so the next tick walks off
        it.

        Two facts are kept apart, because only one of them is about the ground. A refused pit
        (`[DROP_HOLE_UNAVAILABLE]`) is a property of the terrain and latches for the rest of the
        run -- the drop then goes onto open ground, where it still frees the slot. But an open
        drop lands under the bot's own feet, and dropped items are gathered again by walking over
        them, so that drop is not finished when the tool returns: it is finished three blocks
        away. `drop_open_pending` carries that unfinished step until `step_away_drops` closes it.

        The position is recorded on the drop itself rather than read later, because by the time
        the walk-back runs the bot has already started moving off it.
        """
        if action.label != "drop_junk":
            return
        if ok:
            if self._drop_hole_blocked and gs is not None and gs.pos is not None:
                self._drop_open_pending = True
                self._drop_pos = (gs.pos[0], gs.pos[1], gs.pos[2])
                self._drop_heading = self._heading
                self._step_away_fails = 0
            return
        if "DROP_HOLE_UNAVAILABLE" in (text or ""):
            self._drop_hole_blocked = True

    def _note_heading(self, before, after) -> None:
        """Keep the way the bot last walked, as a unit vector in the horizontal plane.

        Neither the observation nor the server names the bot's facing, but a navigate or a dig
        leaves the body looking the way it went, and an open `drop_junk` tosses that way. A tick
        that moved less than a block keeps the old heading: the drop itself, a craft or a failed
        step do not turn the bot round.
        """
        if before is None or after is None:
            return
        dx, dz = after[0] - before[0], after[2] - before[2]
        length = math.hypot(dx, dz)
        if length >= 1.0:
            self._heading = (dx / length, dz / length)

    def _note_step_away(self, action, ok: bool) -> None:
        """Close the walk-off once the bot has actually made it, so the offer disappears.

        On success the junk is safely off the bot's feet and the latch clears. A refusal keeps
        it: the junk is still there, and dropping the flag would leave the drop unfinished with
        nothing left to finish it. But a step that keeps failing (rough Nether terrain the 3-block
        navigate cannot cross) would wedge the bot on that one candidate forever, so after
        `STEP_AWAY_GIVE_UP` consecutive refusals the walk off is given up and the latch clears -
        the bot moves on, and the junk stays dropped because the bot marches away from it, not
        over it.
        """
        if action.label != "step_away_drops":
            return
        if ok:
            self._drop_open_pending = False
            self._drop_pos = None
            self._step_away_fails = 0
            return
        args = getattr(action, "args", None) or {}
        cell = (args.get("x"), args.get("y"), args.get("z"))
        if all(isinstance(v, int) for v in cell):
            # The next tick picks another standable cell from the same read.
            self._drop_refused.add(cell)
        self._step_away_fails += 1
        if self._step_away_fails >= STEP_AWAY_GIVE_UP:
            self._drop_open_pending = False
            self._drop_pos = None
            self._step_away_fails = 0

    def _note_result(self, action, ok: bool) -> None:
        """Report one result to the two latches that read its outcome.

        The walk-off is reported both ways: `STEP_AWAY_GIVE_UP` releases the latch inside
        `_note_step_away` on a refusal, and it must actually see the refusals - the call sat
        under the success guard, so a walk-off the terrain refused on every try never reached
        the give-up (live, run81: a cliff top answered `[NAVIGATION_TARGET_UNSUPPORTED]` 470
        times in a row while the latch, the exclusive offer and the run cap burned together).
        `_note_placed` adopts cells the server said were placed, which is a success-only fact.
        """
        self._note_step_away(action, ok)
        if ok:
            self._note_placed(action)

    def _note_chest(self, action, ok: bool, text: str | None) -> None:
        """Remember that the base's ground cannot hold the chest, so the chain is dropped.

        `[PLACE_NO_SUPPORT]` says no solid block touches the cell beside the base: a property
        of the base site, not of the attempt (live, Nether run: the run answered it on every
        tick and never reached the portal). Giving the chest up lets the inventory relief fall
        back to `drop_junk`, which still frees the slots it is for and cannot fail.
        """
        if ok or action.label != "place_chest":
            return
        if "PLACE_NO_SUPPORT" not in (text or ""):
            return
        args = getattr(action, "args", None) or {}
        cell = (args.get("x"), args.get("y"), args.get("z"))
        main = None
        if self.base is not None:
            bx, by, bz = self.base
            main = (bx, by + 1, bz + 1)
        # A network cell is one candidate among many: refuse that one and pick the next, the
        # chain stays open. Only the main cell -- or a placement whose cell the loop cannot name
        # next to the base -- is the ground's verdict on chests in general, which is the old
        # giving-up-the-chain the comment describes.
        if main is None or self.chest_pos is None or cell == main:
            self._chest_blocked = True
            return
        self._overflow_no_support.add(cell)

    def _note_container(self, action, ok: bool, text: str | None) -> None:
        """Track the storage network's fullness, chest by chest.

        The server answers a deposit `[CONTAINER_BATCH_INCOMPLETE] ... Observed item transfers:
        gunpowder 0/2, cobblestone 0/16` when the named chest cannot take what the bot offered
        (live, run79): not a mistake of the action but the cell's storage space running out. The
        relief needs exactly that knowledge per cell -- which chest closed, so the deposit aims
        at the next one, and the network may only grow once every one of them closed. A deposit
        that succeeds into a cell opens it again: the server kept something, or a `withdraw`
        emptied it, and a full-mark that outlives the space under it would grow chests nobody
        asked for.
        """
        if action.tool != "use_container":
            return
        args = getattr(action, "args", None) or {}
        if args.get("operation") != "deposit":
            return
        cell = (args.get("x"), args.get("y"), args.get("z"))
        if not all(isinstance(v, int) for v in cell):
            return
        if not ok and "CONTAINER_BATCH_INCOMPLETE" in (text or ""):
            self._chests_full.add(cell)
        elif ok:
            self._chests_full.discard(cell)

    def _note_container_missing(self, action, ok: bool, text: str | None) -> None:
        """A cell the server calls not-a-container is dropped from what the run remembers.

        `[CONTAINER_NOT_PRESENT] The loaded target block is air, ...` is the one answer the
        storage network had no use for: run83 re-adopted its base from a bed a previous run
        left standing and took the chest cell banked beside it on adoption -- four blocks
        away, never checked -- while the world had given that chest up. Four deposits walked
        to the empty space, each answered by the same message, and none of it landed: the
        phantom stayed the deposit's target for the rest of the run, `stow_surplus` walked
        its ticks empty and `take_stored_food` would have done the same to the provisions
        believed to stand there. What the server names cannot stand: the cell leaves the
        network, the provisions believed in it leave with it, and the storage chain places a
        chest again -- adoption is not special-cased, the next phantom dies the same first
        death. A station cell whose block is gone is no station either.

        Only a block that is not what the cell was remembered as says the station is gone. The
        same code also names a block that stands and is merely refused: run49 answered every
        furnace drain `... is furnace, not a supported location-owned container` 119 times, a
        real furnace a server without furnace support could not open. Forgetting that one
        brought it straight back through `_adopt_homebase` with its occupied flag wiped, and
        the cook and the drain refused each other in turn; it stays, and the stuck guard drops
        the drain as before. A message that names no block is the server's word alone.
        """
        if ok or "CONTAINER_NOT_PRESENT" not in (text or ""):
            return
        args = getattr(action, "args", None) or {}
        cell = (args.get("x"), args.get("y"), args.get("z"))
        if not all(isinstance(v, int) for v in cell):
            return
        found = _FOUND_BLOCK_RE.search(text or "")
        block = found.group(1) if found else None
        if block is None or block not in _CHEST_BLOCKS:
            self._forget_chest_cell(cell)
        if (self.furnace_pos is not None and tuple(self.furnace_pos) == cell
                and block != "furnace"):
            self.furnace_pos = None
            self._furnace_contents = {}
            self._furnace_occupied = False
        if (self.smoker_pos is not None and tuple(self.smoker_pos) == cell
                and block != "smoker"):
            self.smoker_pos = None

    def _forget_chest_cell(self, cell: tuple) -> None:
        """Drop one cell from the storage network, with the provisions believed in it."""
        if self.chest_pos is not None and tuple(self.chest_pos) == cell:
            self.chest_pos = None
        if cell in {tuple(c) for c in self._extra_chests}:
            self._extra_chests = [c for c in self._extra_chests if tuple(c) != cell]
        self._chests_full.discard(cell)
        stores = [s for s in self._banked_stores if tuple(s[0]) != cell]
        if len(stores) != len(self._banked_stores):
            self._banked_stores = tuple(stores)
            banked: dict[str, int] = {}
            for _, name, count in stores:
                banked[name] = banked.get(name, 0) + count
            self.banked = banked

    def _note_chest_lid(self, action, ok: bool, text: str | None) -> None:
        """Remember the block sitting on the base chest's lid, so `stow_surplus` digs it.

        `[CHEST_BLOCKED]` says an opaque full cube stands directly above the chest and its lid
        cannot open (`perception.parse_block_above` names that cell). It is a property of the
        block, not of the attempt, so the cell is remembered and the deposit's own resolve digs
        it clear before opening again. Cleared on any `stow_surplus` success -- the dig, or the
        deposit that follows it -- so the dig runs once and the chest is then opened.
        """
        if action.label != "stow_surplus":
            return
        if ok:
            self._chest_above = None
            return
        cell = perception.parse_block_above(text or "")
        if cell is not None:
            self._chest_above = cell

    def _note_furnace(self, action, ok: bool, text: str | None, res=None) -> None:
        """Give up a base furnace the loop can no longer use, so cooking falls back to a
        temporary workstation.

        Two refusals mean the standing furnace is finished. `[PLACE_NO_SUPPORT]` on the
        `place_furnace` itself: no solid block touches the cell beside the base, so it can never
        stand there. `[FURNACE_NOT_EMPTY]` on a `smelt_item`: the furnace still holds output, which
        used to be the end of that cell too - no MCP tool could open a furnace, so the loop dropped
        the cell and cooked in a carried one instead (live, run26: a completed cook at a full pack
        left its output behind and every `cook_food` after answered the same refusal).

        The server now opens a furnace for `use_container inspect` and `withdraw`, so that refusal
        is no longer fatal: `collect_smelted` empties the furnace and the cook retries. The cell is
        only dropped when the drain itself is refused, which is the case no retry can fix.
        """
        if ok:
            if action.label == "collect_smelted" and action.tool == "use_container":
                if (getattr(action, "args", None) or {}).get("operation") == "inspect":
                    # The look is how the loop learns what is in there. Treating it as a finished
                    # drain threw that away and cleared the occupied flag, so the next tick cooked
                    # into a furnace the loop believed empty and met the same refusal again.
                    contents = perception.parse_furnace_contents(getattr(res, "data", None))
                    self._furnace_contents = contents
                    self._furnace_occupied = bool(contents)
                else:
                    self._furnace_contents = {}
                    self._furnace_occupied = False
            return
        text = text or ""
        if action.label == "collect_smelted" and "CONTAINER_OPERATION_UNSUPPORTED" in text:
            # A server without furnace support: back to cooking in a carried furnace.
            self._furnace_blocked = True
            self.furnace_pos = None
            return
        if action.label == "place_furnace":
            if "PLACE_NO_SUPPORT" in text:
                self._furnace_blocked = True
            return
        if action.tool == "smelt_item" and "FURNACE_NOT_EMPTY" in text:
            # Not the end of the cell: flag that it is occupied so `collect_smelted` drains it.
            self._furnace_occupied = True

    def _note_place(self, action, ok: bool, text: str | None) -> None:
        """Remember the cell a placement was refused at, so the unblock step can dig it clear.

        `[PLACE_TARGET_OCCUPIED]` names that cell (`place_block` args): the block standing there
        is the whole cause, and only digging that exact cell lets the placement land. Live,
        from-zero run: the cell beside the base held `grass_block` and `place_chest` answered
        the code on every tick. `make_room` reads the cell back out of the GameState.
        """
        args = getattr(action, "args", None) or {}
        cell = (args.get("x"), args.get("y"), args.get("z"))
        if not ok and "PLACE_TARGET_OCCUPIED" in (text or "") and all(
                isinstance(v, int) for v in cell):
            self._blocked_cell = cell
        elif ok and action.label == "make_room":
            # The pocket that dug that cell succeeded: the next tick the placement retries (see
            # `_unblock_done`), so the cell no longer needs digging.
            self._blocked_cell = None

    def _note_recover(self, gs, action, ok: bool, text: str | None) -> None:
        """Write off a death site once a recovery there reports the drops are gone.

        `pick_up_items(recover_death_items)` answers `[NO_MATCHING_ITEMS]` when it reaches the
        death site and its 16 m scan finds nothing left to pick up (the stack despawned, burnt,
        or never existed): a definitive answer, not a transient one, since the walk already
        happened. `_can_recover` keeps offering the walk for the whole window regardless, so the
        reaction owns every tick and the bot stands at the site until the window expires -- and a
        fresh death resets the window. Live, run14: 40 + 77 ticks burned in two such loops, over
        half the run, and the second only ended when the 300 s window ran out. Remembering the
        cell drops the offer at once; a later death elsewhere is a different cell, still walked.
        A site the walk cannot reach is written off once `RECOVER_UNREACHABLE_REPEATS` walks to
        it in a row were refused that way.
        """
        if action.label != "recover_drops":
            return
        if ok or gs.death_pos is None:
            self._recover_unreachable = (None, 0)
            return
        cell = _cell(gs.death_pos)
        if RECOVER_UNREACHABLE_CODE in (text or ""):
            # One unreachable walk can be a mob in the way or a door the next tick opens; the
            # same cell refused again and again is terrain. run90 asked 231 times from one spot
            # (182.5, 72, 179.5) for a site 50 m off until the death record went stale.
            last, reps = self._recover_unreachable
            reps = reps + 1 if last == cell else 1
            self._recover_unreachable = (cell, reps)
            if reps >= RECOVER_UNREACHABLE_REPEATS:
                self._recover_written_off.add(cell)
            return
        # A walk the reflex interrupted (`[HOSTILE_CONTACT]`, the body held) says nothing about
        # the route either way, so it leaves the count where it was.
        if any(code in (text or "") for code in RECOVER_FINAL_CODES):
            self._recover_written_off.add(cell)

    def _note_boxed(self, gs, action, ok: bool, text: str | None) -> None:
        """Count the walks refused from one cell because the bot is shut in (`shut_in`).

        Any label counts: whatever walks fails the same way from a pocket. The body leaving the
        cell, or a walk whose `no path` search ran far, starts over; a step that works where it
        stands (a craft, a meal, a pickup, or a tick the reflex interrupted) says nothing about
        the pocket and leaves the count as it was. A `dig_out` that ran leaves the count one
        short, so a single further refusal from the same cell digs the next side at once instead
        of after three more failed walks.

        `gs` is the state *after* the action, and it has to be: the refusal names the cell the
        pathfinder started from, which is where the body ended up, not where the tick began.
        Read against the pre-action cell, a walk that moved first and failed at its destination
        does not look shut in at all. run95 it125 walked 105 m and 52 blocks up, was refused with
        `visited 1 nodes` at -216,-2,-367, and the count stayed at 0 because the tick had begun
        far away. `ok` resets nothing in place either: the very next tick recovered drops in the
        same buried cell, and that success would have wiped the one refusal already counted.
        """
        if gs.pos is None:
            return
        cell = _cell(gs.pos)
        last, reps = self._boxed
        if action.label == "dig_out":
            if ok:
                self._dig_out_tries += 1
                self._boxed = (cell, BOXED_REPEATS - 1)
                return
            # A dig_out that could not even start -- `[BUILD_INCOMPLETE] ... no path found after
            # 0 ms; visited 1 node` -- means the build reach cannot touch *this* side's cells from
            # the perch the bot stands on. Turn to the next side on the refusal too, not only on a
            # dig that went through: otherwise one unreachable wall is dug at forever, with
            # `dig_out_try` frozen at the side it started on (live run101 it6-181, 176 refusals at
            # the same east cells from a fractional-y perch on the old world's leftover frame).
            self._dig_out_tries += 1
            if self._dig_out_tries >= DIG_OUT_MAX_TRIES:
                # Every side tried round and none would give: the pocket cannot be dug out of from
                # here. Drop the latch so the tick stops going to `dig_out` alone and the shut-in
                # ladder and System 2 get to try something else, rather than the run spending its
                # whole budget on a tunnel the build reach refuses.
                self._boxed = (None, 0)
                self._dig_out_tries = 0
            else:
                self._boxed = (cell, reps)
            return
        if cell != last:
            reps = 0
            self._boxed = (None, 0)
            self._dig_out_tries = 0
        if ok:
            return
        if shut_in(text, gs.pos) or "repeated_movement_failure" in (text or ""):
            # `repeated_movement_failure` is the body not moving on a re-issued route: the bot is
            # physically enclosed, the same shut-in as a short `no path` search but reported with
            # no node count for `shut_in` to read. It counts as `blocked_by_surroundings` too, so
            # the unblock ladder (`make_room`, then `explore`) answers it first -- but neither a
            # feet pocket nor another walk frees a bot walled in on every side (live, run96: 100
            # `explore` from one spawn cell in birch logs and leaves, never a step, the ladder
            # cycling make_room/explore the whole run). Counting it toward the tunnel lets
            # `dig_out` take over once the ladder has proven it cannot, and carve a way out. Left
            # out of the `elif` below, which would otherwise read its `[NAVIGATION_STOPPED]` code
            # and reset the count on the very refusal that should raise it.
            self._boxed = (cell, reps + 1)
        elif _fail_code(text) == "NAVIGATION_STOPPED" or "no path found" in (text or ""):
            self._boxed = (None, 0)

    def _note_stow_blocked(self, action, ok: bool, text: str) -> None:
        """Only the server's full message blocks the deposit as relief.

        A failed deposit was enough once, whatever the reason said on it: a single
        `[HOSTILE_CONTACT]` off the walk, or a lid dig interrupted by combat, kept the loop
        dropping what a chest standing right there could have stored. The full message blocks
        -- and a deposit that works again opens the relief.
        """
        if action.label != "stow_surplus":
            return
        if ok:
            self._stow_blocked = False
        elif "CONTAINER_BATCH_INCOMPLETE" in text:
            self._stow_blocked = True

    def _note_placed(self, action) -> None:
        """Remember the cell of a placement that just succeeded.

        `establish_base` reports the surface cell through its navigate target; the base anchor
        is the ground block under the feet it walked to. `place_chest` names the cell it placed
        into, remembered so `use_container` can reach it later -- and since run79 the second
        chest of the network names *its* cell with the same label: `chest_pos` keeps the main
        one (the blaze powder bank at the portal and the lid dig hang on it), the new cell joins
        the overflow a deposit can look for room in. A chest just stood is by definition roomy;
        any full-mark on its cell is now history. The furnace and the smoker latch on their
        placement's own success, the same shape: a station that answered
        `[PLACE_TARGET_OCCUPIED]` is not standing, and a remembered cell that is not there would
        send every later cook to an empty space. A cell the server does not know yet names no
        coordinates: the `surface_up` probe `navigate {x,z}` names the column, not the cell --
        run85's first probe that found an unambiguous surface and walked, met a guard that
        checked only `x` and died on `KeyError: 'y'` in `verify`, the whole run with it.
        """
        if not {"x", "y", "z"} <= set(action.args or {}):
            return
        cell = (int(action.args["x"]), int(action.args["y"]), int(action.args["z"]))
        if action.label == "establish_base":
            self.base = (cell[0], cell[1] - 1, cell[2])
        elif action.label == "place_chest":
            if self.chest_pos is None:
                self.chest_pos = cell
            elif cell != self.chest_pos and cell not in self._extra_chests:
                self._extra_chests.append(cell)
            self._chests_full.discard(cell)
        elif action.label == "place_furnace":
            self.furnace_pos = cell
        elif action.label == "place_smoker":
            self.smoker_pos = cell

    async def _verify(self, state: AgentState) -> dict:
        before = state["before"]
        action = state["action"]
        res = state["result"]
        # Remember a surroundings block so the next tick changes the surroundings before
        # retrying it (see `_note_block` for the clearing rule).
        _ok = getattr(res, "ok", False)
        _text = getattr(res, "text", "")
        self._note_block(action.label, _ok, _text)
        self._note_pour(action, _ok, _text)
        self._note_stance(action, _ok, _text)
        self._note_base(action, _ok, _text)
        self._note_drop(action, _ok, _text, before)
        self._note_chest(action, _ok, _text)
        self._note_chest_lid(action, _ok, _text)
        self._note_container(action, _ok, _text)
        self._note_container_missing(action, _ok, _text)
        self._note_furnace(action, _ok, _text, res)
        self._note_frame_cells(action, _ok, _text)
        self._note_place(action, _ok, _text)
        self._note_recover(before, action, _ok, _text)
        # Opened before the action runs: the rescue has to be down on the *next* tick, while the
        # crossing settles and the teleport fires.
        self._note_crossing(action)
        self._note_stow_blocked(action, _ok, _text)
        self._note_result(action, _ok)
        await self._hold_wait(action, _ok)
        after = await self._observe()
        # After the observation, not before: the `no path found` refusal names the cell the walk
        # started from, which is where the action left the body (see `_note_boxed`).
        self._note_boxed(after, action, _ok, _text)
        self._note_heading(before.pos, after.pos)
        # `surface_up` answers itself through the position it lands on (a resolved probe) or
        # through the height the server names when it refuses to guess (an ambiguous column);
        # both are read here, after the observation, and gate `establish_base` next tick.
        self._note_surface(action, _ok, _text, after.pos)
        await self._reapply_after_death(after)
        delta = vmod.diff(before, after)
        self._note_held(res, before, after)
        self._note_hurt(before, after)
        self._note_build(action, _ok, delta)
        self._note_stow(action, _ok, delta)
        self._note_take_powder(action, _ok, _text, delta)
        self._note_locate(action, _ok)
        self._note_barter(action, _ok)
        self._note_portal(action, _ok)
        self._note_landmark(action, delta, after)
        progressed = vmod.progressed(action.label, delta, _ok)
        self._track_stuck(action.label, _ok, _text, progressed)
        # After the stuck guard: it owns the countdown and the offer window, and a wandering
        # action is not a refusal, so the two runs of counts do not disturb each other.
        self._track_wander(action.label, res)
        self._track_idle(action.label, delta)
        self._refresh_after(after)
        record = {
            "stage": "action",
            "iteration": state.get("iteration", 0),
            "state": before.as_state(),
            "narrative": before.narrative(),
            "candidates": state["decision"].candidates,
            "label": action.label,
            "tool": action.tool,
            "escalated": state.get("escalated", False),
            "ok": bool(getattr(res, "ok", False)),
            "status": getattr(res, "status", None),
            "text": (getattr(res, "text", "") or "")[:400],
            "delta": delta,
            "progressed": progressed,
            "answers": _chosen(getattr(state["decision"], "answers", {})),
        }
        if state["decision"].note:
            # The danger band rewriting a gate `yes` shows here, so the run log does not read as
            # "the gate said yes and the bot worked anyway" without a reason.
            record["note"] = state["decision"].note
        target = getattr(state["decision"], "target", None)
        if target is not None:
            record["target"] = getattr(target, "label", str(target))
            meta = getattr(state["decision"], "target_meta", None)
            if meta:
                record["target_meta"] = meta
        self.log.append(record)
        self.history.append(record)
        iteration = state.get("iteration", 0) + 1
        return {"before": after, "iteration": iteration,
                **self._stop_state(after, iteration)}

    async def _reapply_after_death(self, gs) -> None:
        """The server restores default policy on death; re-apply once per death.

        A death is told apart by its server timestamp, not by its age. run90 died ten times,
        six of them within six minutes at night, and the old age window (re-arm only once a death
        is over 120 s old) re-applied the policy for three: a death 34 s after the last one found
        it still disarmed, and the bot fought zombies naked on the server's default. A death
        first seen when it is already old (the run's start, after a restart) is noted, not
        answered: the policy the run applied at its start already stands over it.
        """
        key = getattr(gs, "last_death_at", None)
        age = getattr(gs, "last_death_age_s", None)
        if key is None or age is None or key == self._death_seen:
            return
        self._death_seen = key
        if age > DEATH_REAPPLY_AGE_S:
            return
        summary = await survival.apply_policy(self.mcp, "re-apply after death")
        self.log.append({"stage": "policy", "reason": "death", "summary": summary})

    def _after_guard(self, state: AgentState) -> str:
        return "execute" if state.get("action") is not None else "defer"

    def _route(self, state: AgentState) -> str:
        return END if state.get("done") else "decide"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("perceive", self._perceive)
        graph.add_node("decide", self._decide)
        graph.add_node("guard", self._guard)
        graph.add_node("execute", self._execute)
        graph.add_node("defer", self._defer)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "decide")
        graph.add_edge("decide", "guard")
        graph.add_conditional_edges("guard", self._after_guard,
                                    {"execute": "execute", "defer": "defer"})
        graph.add_edge("execute", "verify")
        graph.add_conditional_edges("verify", self._route, {"decide": "decide", END: END})
        graph.add_conditional_edges("defer", self._route, {"decide": "decide", END: END})
        return graph.compile(checkpointer=MemorySaver())

    async def run(self) -> dict:
        config = {
            "configurable": {"thread_id": "mlaya"},
            "recursion_limit": 6 * self.settings.loop.max_iterations + 10,
        }
        return await self.graph.ainvoke({}, config)
