"""The run's milestones, resolved once, in one order.

Written against the cold start on a half-played world: the ~40 latches in `Agent.__init__` come
back `False`/`[]`, and the two reads that could correct them - `view_status` and `view_blocks` -
only answer inside loaded chunks. A run that restarts 300 m from its own base therefore sees no
gate, no bed, no frame, and the honest reading of that is "nothing has been built yet". The
server's own tables answer across the death, the dimension and the restart, so `Progress` is the
place where the ledger, the scan and the latch meet, and the order is fixed:

    durable ledger fact  >  world scan  >  run latch  >  False

The fixture rows are the shapes `query_bot_data` returns for those tables, taken from the live
per-bot database.
"""
from __future__ import annotations

import unittest

from mlaya import perception, progress
from mlaya.actions import _eyes_owed, _frame_empty


def _payload(rows):
    """The `query_bot_data` envelope the loop reads."""
    return {"response": {"data": {"result": {"query": {"rows": rows}}}}}


class _Read:
    def __init__(self, data):
        self.data, self.text = data, ""


class _Mcp:
    """A stand-in executor: one canned row set per SQL, keyed by the query itself."""

    def __init__(self, by_sql):
        self.by_sql = by_sql
        self.sql: list[str] = []

    async def read(self, tool, args, why):
        sql = args["sql"]
        self.sql.append(sql)
        if sql not in self.by_sql:
            raise RuntimeError(f"no such table: {sql}")
        return _Read(_payload(self.by_sql[sql]))


GATE_ARGS = '{"x": -9, "y": 70, "z": -52}'
NETHER_LIT = ('{"dimension":"overworld","target":{"x":-9,"y":70,"z":-52},'
              '"portal":{"kind":"nether","portalBefore":0,"portalAfter":6,"activated":true},'
              '"status":"succeeded"}')
END_FILLED = ('{"dimension":"overworld","target":{"x":12,"y":70,"z":-40},'
              '"portal":{"kind":"end","center":{"x":12,"y":70,"z":-40},'
              '"eyesBefore":5,"eyesAfter":9,"portalBefore":0,"portalAfter":9,"activated":true},'
              '"status":"succeeded"}')
BED_SLEEP = ('{"status":"succeeded","sleep":{"asleep":false,"morning":true,"respawnSet":true,'
             '"placedBed":true,"bedPosition":{"x":4,"y":69,"z":-48}}}')


class LedgerBootstrapTest(unittest.IsolatedAsyncioTestCase):
    """What the server's own record says the world already contains."""

    def setUp(self):
        self.by_sql = {
            perception.GATE_CHUNK_SQL: [["the_nether", -1, -1], ["overworld", -1, -1]],
            perception.PORTAL_LEDGER_SQL: [[GATE_ARGS, END_FILLED], [GATE_ARGS, NETHER_LIT]],
            perception.BED_LEDGER_SQL: [[GATE_ARGS, BED_SLEEP]],
            perception.DIMENSION_LEDGER_SQL: [["the_nether"], ["overworld"]],
            perception.BEARING_SQL: [["stronghold", "observed", 137.4]],
            perception.CHECKPOINT_READ_SQL: [],
        }

    async def test_the_ledger_names_the_gate_and_the_frame(self) -> None:
        facts = await progress.bootstrap_progress(_Mcp(self.by_sql))
        self.assertTrue(facts.gate_lit)
        self.assertTrue(facts.frame_complete)
        self.assertEqual(facts.gate_cell, (-9, 70, -52))
        self.assertTrue(facts.nether_entered)

    async def test_the_end_frame_is_counted_from_the_server_not_the_pack(self) -> None:
        # `eyesAfter` is the world counting the sockets, including the ones the world generated
        # filled. The inventory delta only ever counted what this bot put in.
        facts = await progress.bootstrap_progress(_Mcp(self.by_sql))
        self.assertEqual(facts.frame_eyes, 9)
        self.assertTrue(facts.end_portal_active)

    async def test_a_bed_in_the_ledger_is_kept_as_a_cell(self) -> None:
        facts = await progress.bootstrap_progress(_Mcp(self.by_sql))
        self.assertEqual(facts.bed_cells, ((4, 69, -48),))

    async def test_a_saved_bearing_is_a_bearing(self) -> None:
        facts = await progress.bootstrap_progress(_Mcp(self.by_sql))
        self.assertTrue(facts.stronghold_estimated)
        self.by_sql[perception.BEARING_SQL] = [["stronghold", "pending", None]]
        self.assertFalse((await progress.bootstrap_progress(_Mcp(self.by_sql))).stronghold_estimated)

    async def test_a_world_with_no_ledger_still_boots(self) -> None:
        # Every read is guarded on its own: an absent table is an absent fact, not a dead run.
        facts = await progress.bootstrap_progress(_Mcp({}))
        self.assertFalse(facts.gate_lit)
        self.assertFalse(facts.end_portal_active)
        self.assertEqual(facts.frame_eyes, 0)

    async def test_the_checkpoint_note_restores_the_counters(self) -> None:
        note = (perception.CHECKPOINT_PREFIX +
                ' {"pearl_wave_trips": 3, "barter_attempts": 5, "frame_eyes": 4}')
        self.by_sql[perception.CHECKPOINT_READ_SQL] = [[note]]
        facts = await progress.bootstrap_progress(_Mcp(self.by_sql))
        self.assertEqual(facts.pearl_wave_trips, 3)
        self.assertEqual(facts.barter_attempts, 5)
        self.assertEqual(facts.checkpoint["frame_eyes"], 4)


class PrecedenceTest(unittest.TestCase):
    """Ledger beats scan beats latch, everywhere and in that order."""

    def test_a_ledger_gate_beats_a_blank_scan_and_an_empty_latch(self) -> None:
        facts = progress.DurableFacts(gate_lit=True, frame_complete=True)
        p = progress.progress_of(perception.GameState(), facts)
        self.assertTrue(p.gate_lit)
        self.assertEqual(p.source["gate_lit"], "ledger")

    def test_a_scan_beats_a_latch_that_never_set(self) -> None:
        gs = perception.GameState(resource_found={"nether_portal": 6})
        p = progress.progress_of(gs, progress.DurableFacts())
        self.assertTrue(p.gate_lit)
        self.assertEqual(p.source["gate_lit"], "scan")

    def test_a_latch_holds_when_the_chunk_is_unloaded(self) -> None:
        # The whole reason the latch exists: the gate burns, but its chunk is not loaded.
        gs = perception.GameState(portal_lit=True)
        p = progress.progress_of(gs, progress.DurableFacts())
        self.assertTrue(p.gate_lit)
        self.assertEqual(p.source["gate_lit"], "latch")

    def test_nothing_says_the_gate_burns(self) -> None:
        p = progress.progress_of(perception.GameState(), progress.DurableFacts())
        self.assertFalse(p.gate_lit)
        self.assertEqual(p.source["gate_lit"], "default")

    def test_the_scan_may_undo_the_frame_latch(self) -> None:
        # The one exception the loop already had: a plane that names its own obsidian as air is
        # the frame talking about itself (run103), so it refutes even a durable fact. A plane with
        # *no* obsidian named at all says nothing and leaves the ledger standing.
        base = (0, 70, 0)
        cells = perception.frame_obsidian_cells(base)
        gs = perception.GameState(base=base,
                                  frame_names={cell: ("obsidian" if i < 4 else "air")
                                               for i, cell in enumerate(cells)})
        p = progress.progress_of(gs, progress.DurableFacts(frame_complete=True))
        self.assertFalse(p.frame_complete)
        self.assertEqual(p.source["frame_complete"], "scan")

    def test_a_blank_plane_leaves_the_ledger_standing(self) -> None:
        gs = perception.GameState(base=(0, 70, 0))
        p = progress.progress_of(gs, progress.DurableFacts(frame_complete=True))
        self.assertTrue(p.frame_complete)
        self.assertEqual(p.source["frame_complete"], "ledger")

    def test_powder_in_the_pack_is_a_scan_and_a_latch_is_a_latch(self) -> None:
        carried = perception.GameState(dimension="minecraft:overworld",
                                       inventory={"blaze_powder": 4})
        self.assertEqual(progress.progress_of(carried).source["powder_banked"], "scan")
        stowed = perception.GameState(dimension="the_nether", powder_stowed=True)
        self.assertEqual(progress.progress_of(stowed).source["powder_banked"], "latch")

    def test_a_ledger_bed_counts_only_beside_the_gate(self) -> None:
        gate = (-9, 70, -52)
        near = progress.DurableFacts(bed_cells=((-9, 70, -48),), gate_cell=gate)
        self.assertTrue(progress.progress_of(perception.GameState(), near).bed_at_base)
        far = progress.DurableFacts(bed_cells=((400, 69, 400),), gate_cell=gate)
        self.assertFalse(progress.progress_of(perception.GameState(), far).bed_at_base)
        # With no anchor at all the ledger cannot say which base a bed belongs to.
        self.assertFalse(progress.progress_of(perception.GameState(),
                                             progress.DurableFacts(bed_cells=((-9, 70, -48),))).bed_at_base)

    def test_the_eyes_owed_are_sized_on_the_frame_the_world_reports(self) -> None:
        # A frame that arrived with five eyes seated owes seven, not twelve: the rod haul and the
        # pearl waves are sized on this number.
        facts = progress.DurableFacts(frame_eyes=5, stronghold_estimated=True)
        gs = perception.GameState(dimension="minecraft:overworld", base=(-9, 70, -52))
        p = progress.progress_of(gs, facts)
        self.assertEqual(p.frame_eyes, 5)
        self.assertEqual(p.eyes_owed, 7)
        gs.frame_eyes = p.frame_eyes
        gs.stronghold_estimated = True
        self.assertEqual(_eyes_owed(gs), 7)
        self.assertEqual(_frame_empty(gs), 7)

    def test_an_open_portal_owes_nothing(self) -> None:
        facts = progress.DurableFacts(end_portal_active=True, frame_eyes=12,
                                      stronghold_estimated=True)
        p = progress.progress_of(perception.GameState(), facts)
        self.assertEqual(p.eyes_owed, 0)
        self.assertEqual(p.source["end_portal_active"], "ledger")

    def test_a_restart_mid_stronghold_reads_its_own_ledger(self) -> None:
        # Every latch is False, the base chunks are unloaded, and the only thing that knows the
        # gate was lit, the frame built and the portal opened is the server's record of the runs
        # that did it. That is the point: "this world has a burning gate" answers "where am I",
        # not "who built it".
        facts = progress.DurableFacts(gate_lit=True, frame_complete=True, end_portal_active=True,
                                      frame_eyes=12, stronghold_estimated=True, nether_entered=True,
                                      bed_cells=((-9, 70, -48),), gate_cell=(-9, 70, -52))
        gs = perception.GameState(dimension="the_nether")
        p = progress.progress_of(gs, facts)
        self.assertTrue(p.gate_lit and p.frame_complete and p.bed_at_base and p.end_portal_active)
        self.assertTrue(p.nether_entered)
        self.assertEqual(p.eyes_owed, 0)
        self.assertEqual(p.source["frame_eyes"], "ledger")
        self.assertEqual(p.source["nether_entered"], "ledger")


class MilestoneNarrativeTest(unittest.TestCase):
    """The stand has to be the first thing the model sees, and the first thing the log holds."""

    def test_the_sentence_leads_the_narrative(self) -> None:
        gs = perception.GameState(dimension="minecraft:overworld", portal_lit=True,
                                  bed_pos=(4, 69, -48), base=(4, 68, -48), powder_stowed=True)
        gs.progress = progress.progress_of(gs, progress.DurableFacts())
        line = gs.narrative()
        self.assertIn("Milestones: the gate burns", line)
        self.assertLess(line.index("Milestones"), line.index("Inventory"))

    def test_a_cold_start_says_so(self) -> None:
        gs = perception.GameState(dimension="minecraft:overworld")
        gs.progress = progress.progress_of(gs, progress.DurableFacts())
        self.assertEqual(gs.progress.sentence(),
                         "Milestones: no gate burns yet, no bed is laid, "
                         "no blaze powder is banked, the end portal frame holds 0 of 12 eyes")

    def test_the_state_logs_the_stand_and_its_sources(self) -> None:
        facts = progress.DurableFacts(gate_lit=True)
        gs = perception.GameState(dimension="minecraft:overworld")
        gs.progress = progress.progress_of(gs, facts)
        logged = gs.as_state()["progress"]
        self.assertEqual(logged["gate_lit"], True)
        self.assertEqual(logged["source"]["gate_lit"], "ledger")
        back = perception.GameState.from_state(gs.as_state())
        self.assertTrue(back.progress.gate_lit)
        self.assertEqual(back.progress.source["gate_lit"], "ledger")
