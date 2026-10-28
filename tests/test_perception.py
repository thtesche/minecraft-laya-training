from __future__ import annotations

import json
import unittest

from mlaya.perception import (EDIBLE_NAMES, FORTRESS_RADIUS, GROUND_NAMES, RESCUE_GROUND_NAMES,
                              RESCUE_RADIUS, RESOURCE_LIMIT, BlockHit,
                              frame_obsidian_cells, frame_volume, parse_block_above, parse_box,
                              parse_cells, parse_furnace_contents, parse_ground_height,
                              parse_frontier_heading, parse_remembered_gate,
                              parse_remembered_stock, parse_remembered_stocks,
                              REMEMBERED_GATE_SQL,
                              parse_portal_corner, parse_refused_cells, parse_resource_counts,
                              parse_resources, parse_status, parse_surface_height,
                              portal_interior, build, GameState)


def _status(situation: dict) -> dict:
    return {"response": {"data": {"result": {"situation": situation}}}}


class ResourceCountParseTest(unittest.TestCase):
    """`parse_resource_counts` keeps the per-name `found` total across every loaded chunk, which
    the radius-limited hit list drops. A surface bot needs it to know that loaded diamond ore
    exists below, so the change that offers the tunnel dig can fire without a nearby hit."""

    def test_found_total_is_kept_outside_the_hit_radius(self) -> None:
        data = {"response": {"data": {"result": {"blocks": {"find": [
            {"name": "diamond_ore", "found": 15479, "listed": [
                {"x": 0, "y": 8, "z": 0, "name": "diamond_ore", "distance": 109.6},
            ]},
            {"name": "iron_ore", "found": 0, "listed": []},
        ]}}}}}
        self.assertEqual(parse_resource_counts(data), {"diamond_ore": 15479})

    def test_missing_find_block_is_empty(self) -> None:
        self.assertEqual(parse_resource_counts({}), {})


class ResourceParseRadiusTest(unittest.TestCase):
    """`parse_resources` bounds its hits at `RESOURCE_RADIUS`, but the water rescue's ground
    scan is read to `RESCUE_RADIUS`: the shore it had to swim to sat at 65.1 m while the 64 m
    read left only a cave floor to aim at (live, run19b). No other group widens the bound."""

    _DATA = {"response": {"data": {"result": {"blocks": {"find": [
        {"name": "grass_block", "found": 40, "listed": [
            {"x": -117, "y": 62, "z": -88, "name": "grass_block", "distance": 65.1,
             "above": "air", "below": "dirt"},
            {"x": -116, "y": 62, "z": -86, "name": "grass_block", "distance": 66.0,
             "above": "air", "below": "dirt"},
        ]},
    ]}}}}}

    def test_the_default_read_drops_ground_past_the_resource_radius(self) -> None:
        self.assertEqual(parse_resources(self._DATA), {})

    def test_the_rescue_radius_keeps_the_shore(self) -> None:
        kept = parse_resources(self._DATA, radius=RESCUE_RADIUS)
        self.assertEqual([h.distance for h in kept["grass_block"]], [65.1, 66.0])


class FortressRadiusTest(unittest.TestCase):
    """The Nether landmark group is read to `FORTRESS_RADIUS`, not the 64 m every other group uses.

    `view_blocks.find` has no radius of its own -- it returns the nearest hits *anywhere loaded* --
    so the 64 m keep was the loop's, and it discarded the one thing this group exists for. A
    fortress 70 m off was in the answer and thrown away, which is why `explore` kept marching on
    open lava sea and four Nether runs never saw a fortress at all.
    """

    _DATA = {"response": {"data": {"result": {"blocks": {"find": [
        {"name": "nether_bricks", "found": 12, "listed": [
            {"x": 118, "y": 64, "z": -96, "name": "nether_bricks", "distance": 70.0,
             "above": "air", "below": "netherrack"},
            {"x": 140, "y": 64, "z": -96, "name": "nether_bricks", "distance": 220.0,
             "above": "air", "below": "netherrack"},
            {"x": 400, "y": 64, "z": -96, "name": "nether_bricks", "distance": 400.0,
             "above": "air", "below": "netherrack"},
        ]},
    ]}}}}}

    def test_the_resource_radius_would_have_dropped_the_whole_fortress(self) -> None:
        self.assertEqual(parse_resources(self._DATA, RESOURCE_LIMIT), {})

    def test_the_fortress_radius_keeps_a_fortress_at_seventy_and_two_hundred_metres(self) -> None:
        kept = parse_resources(self._DATA, RESOURCE_LIMIT, FORTRESS_RADIUS)
        self.assertEqual([h.distance for h in kept["nether_bricks"]], [70.0, 220.0])

    def test_the_bound_is_wider_than_any_plausible_view_distance(self) -> None:
        # The honest shape for a landmark group: read what the world has loaded and let distance
        # decide whether it is worth walking to, instead of deciding by proximity first.
        self.assertGreaterEqual(FORTRESS_RADIUS, 256.0)


class NetherSearchReadsTest(unittest.IsolatedAsyncioTestCase):
    """The Nether reads are wired the way the two fixes need, and only down there.

    The wiring is where both bugs lived: the landmark group was read with the 64 m bound of every
    other group, and nothing asked the server where its unexplored ground was. The Overworld must
    stay untouched -- the frontier read costs a call and the Overworld's own `explore` branch is
    working -- so the gate is the observed dimension.
    """

    class _Read:
        def __init__(self, data):
            self.data = data
            self.text = None

    class _Mcp:
        """Answers the reads `build` makes, recording which tools were asked for."""

        def __init__(self, dimension: str):
            self.dimension = dimension
            self.calls: list = []
            self.frontier_heading = 254.0

        async def read(self, tool: str, args: dict, why: str):
            self.calls.append(tool)
            if tool == "view_status":
                return NetherSearchReadsTest._Read({"response": {"data": {"result": {
                    "situation": {"dimension": self.dimension, "position": {"x": 10.5, "y": 65.0,
                                                                         "z": 4.5},
                                  "health": 20, "food": 18, "inventory": [], "mobs": []}}}}})
            if tool == "view_blocks":
                names = args["find"]["block_names"]
                if "nether_bricks" in names:
                    return NetherSearchReadsTest._Read({"response": {"data": {"result": {
                        "blocks": {"find": [{"name": "nether_bricks", "found": 1, "listed": [
                            {"x": 80, "y": 64, "z": 4, "name": "nether_bricks",
                             "distance": 70.0, "above": "air", "below": "netherrack"}]}]}}}}})
                return NetherSearchReadsTest._Read({})
            if tool == "query_bot_data":
                rows = [[self.frontier_heading]] if self.frontier_heading is not None else []
                return NetherSearchReadsTest._Read({"response": {"data": {"result": {"query": {
                    "columns": ["heading_degrees"], "rows": rows}}}}})
            return NetherSearchReadsTest._Read({})

    async def test_the_fortress_seventy_metres_out_reaches_the_state_in_the_nether(self) -> None:
        mcp = self._Mcp("minecraft:the_nether")
        gs = await build(mcp)
        self.assertEqual([h.distance for h in gs.resources["nether_bricks"]], [70.0])

    async def test_the_frontier_heading_reaches_the_state_in_the_nether(self) -> None:
        gs = await build(self._Mcp("minecraft:the_nether"))
        self.assertEqual(gs.frontier_heading, 254.0)

    async def test_a_world_with_no_unobserved_edge_leaves_the_fallback_in_place(self) -> None:
        mcp = self._Mcp("minecraft:the_nether")
        mcp.frontier_heading = None
        self.assertIsNone((await build(mcp)).frontier_heading)

    async def test_the_overworld_costs_no_frontier_read(self) -> None:
        mcp = self._Mcp("minecraft:overworld")
        gs = await build(mcp)
        self.assertNotIn("query_bot_data", mcp.calls)
        self.assertIsNone(gs.frontier_heading)


class LoggedStateRoundTripTest(unittest.TestCase):
    """A recorded tick can be questioned directly, which is what was missing.

    run53 carried a furnace for 341 of 377 records and `place_furnace` never ran. The decision log
    could not say why: it held no `furnace_pos`, no `furnace_blocked` and no occupied flag, so the
    state had to be reconstructed by hand to ask a predicate about it - three attempts here, three
    different missing fields. The furnace facts are logged now, and `from_state` is the inverse, so
    the question gets asked on the tick that was recorded instead of on a guess at it.
    """

    def _state(self) -> GameState:
        g = GameState()
        g.inventory = {"furnace": 1, "coal": 4, "diamond_pickaxe": 2}
        g.tools = {"pickaxe": "diamond", "sword": "diamond"}
        g.pos = (225.5, 64.0, 223.5)
        g.dimension = "minecraft:overworld"
        g.phase = "day"
        g.base = (226, 63, 222)
        g.chest_pos = (229, 62, 224)
        g.furnace_pos = (229, 62, 222)
        g.furnace_occupied = True
        g.furnace_contents = {"cooked_beef": 1}
        g.furnace_blocked = False
        g.base_adopted = False
        g.resources = {"furnace": [BlockHit("furnace", "furnace", 229, 62, 222, 4.5)],
                        "white_bed": [BlockHit("white_bed", "white_bed", 226, 64, 223, 2.5)],
                        "obsidian": [BlockHit("obsidian", "obsidian", 228, 64, 222, 2.0)]}
        return g

    def test_the_furnace_facts_are_in_the_logged_state(self) -> None:
        logged = self._state().as_state()
        self.assertEqual(logged["furnace_pos"], (229, 62, 222))
        self.assertTrue(logged["furnace_occupied"])
        self.assertEqual(logged["furnace_contents"], {"cooked_beef": 1})
        self.assertFalse(logged["furnace_blocked"])

    def test_a_logged_state_rebuilds_into_the_one_that_was_recorded(self) -> None:
        original = self._state()
        rebuilt = GameState.from_state(original.as_state())
        for field in ("inventory", "tools", "pos", "dimension", "phase", "base", "chest_pos",
                      "furnace_pos", "furnace_occupied", "furnace_contents",
                      "furnace_blocked", "base_adopted"):
            self.assertEqual(getattr(rebuilt, field), getattr(original, field), field)

    def test_the_rebuilt_state_answers_the_questions_the_run_asked(self) -> None:
        # The predicates that decided the furnace's fate, asked on the recorded tick.
        from mlaya.actions import _furnace_needs_draining, _furnace_ready, _furnace_site
        g = GameState.from_state(self._state().as_state())
        self.assertTrue(_furnace_ready(g))
        self.assertTrue(_furnace_needs_draining(g))
        self.assertEqual(_furnace_site(g), (226, 64, 221))
        # Resources come back as hits, so a spatial predicate reads the same as it does live.
        self.assertEqual([h.pos for h in g.resources["furnace"]], [(229, 62, 222)])

    def test_an_older_or_hand_written_state_still_yields_a_usable_one(self) -> None:
        g = GameState.from_state({"inventory": {"furnace": 1}, "base": (0, 63, 0)})
        self.assertEqual(g.inventory, {"furnace": 1})
        self.assertIsNone(g.furnace_pos)
        self.assertFalse(g.furnace_occupied)
        self.assertEqual(g.resources, {})
        self.assertIsNone(g.pos)


class BlindWorldTest(unittest.IsolatedAsyncioTestCase):
    """A `view_status` with no position stops the run instead of feeding it empty decisions.

    Live, run72: the Minecraft connection ended, the transport said so, and the loop kept going.
    Four ticks were logged in two seconds - all `get_wood`, all refused, all at `position: None`
    and `health: None`, with the *previous* base still on the state. Every distance predicate read
    false, the candidate list collapsed to whatever survives without a single fact, and the run
    decided four times about a world it could not see.

    The MCP layer was hardened in `b5b1981` and this is the case it does not cover: a transport
    that answers, with nothing in it.
    """

    class _Read:
        def __init__(self, data=None, text=""):
            self.data, self.text = data, text

    class _Exec:
        def __init__(self, data):
            self.data = data

        async def read(self, tool, args, why):
            return BlindWorldTest._Read(self.data)

    async def test_a_status_without_a_position_stops_the_run(self) -> None:
        from mlaya import perception
        with self.assertRaises(perception.PerceptionBlind):
            await perception.build(self._Exec(None))

    async def test_the_message_names_the_cause_rather_than_the_symptom(self) -> None:
        from mlaya import perception
        with self.assertRaises(perception.PerceptionBlind) as caught:
            await perception.build(self._Exec(None))
        self.assertIn("without a position", str(caught.exception))

    async def test_a_real_status_still_builds(self) -> None:
        from mlaya import perception
        payload = {"response": {"data": {"result": {"situation": {
            "position": {"x": 1.0, "y": 64.0, "z": 2.0, "onGround": True},
            "vitals": {"health": 20.0, "food": 18.0},
        }}}}}
        gs = await perception.build(self._Exec(payload))
        self.assertEqual(gs.pos, (1.0, 64.0, 2.0))

    def test_the_cli_turns_it_into_one_line(self) -> None:
        # Not a traceback: the user should not have to read a stack to find a sentence about
        # sockets.
        from mlaya import cli
        import inspect
        src = inspect.getsource(cli.cmd_run)
        self.assertIn("perception_blind", src)
        self.assertIn("return 2", src)


class RememberedStockParseTest(unittest.TestCase):
    """The chest's contents, read out of the server's container database.

    `parse_remembered_home` already read this table for the cell and left `contents_json` unread.
    That unread column is the whole reason a bot with a full chest still hunted: `hunt_food`
    counts the pack, and food banked at home was not in any number it could see.

    The server's shape is `contents: [{slot, item, count}]`, stored as a JSON string in the
    column, so the parser takes both the string and the already-decoded list.
    """

    @staticmethod
    def _row(cell, contents):
        """A `query_bot_data` answer carrying one row: three cells plus the contents column."""
        return {"response": {"data": {"result": {"query": {"rows": [list(cell) + [contents]]}}}}}

    def test_reads_the_cell_and_counts_every_slot(self) -> None:
        payload = self._row((229, 62, 224), json.dumps([
            {"slot": 4, "item": "cooked_mutton", "count": 21},
            {"slot": 7, "item": "obsidian", "count": 10},
            {"slot": 9, "item": "beef", "count": 3},
            {"slot": 11, "item": "cooked_mutton", "count": 5},
        ]))
        cell, counts = parse_remembered_stock(payload)
        self.assertEqual(cell, (229, 62, 224))
        # Two mutton stacks in different slots are one item of twenty-six.
        self.assertEqual(counts["cooked_mutton"], 26)
        self.assertEqual(counts["beef"], 3)
        self.assertEqual(counts["obsidian"], 10)

    def test_an_already_decoded_list_is_taken_as_well(self) -> None:
        payload = self._row((1, 2, 3), [{"slot": 0, "item": "bread", "count": 4}])
        self.assertEqual(parse_remembered_stock(payload)[1], {"bread": 4})

    def test_an_empty_chest_is_a_fact_not_a_failure(self) -> None:
        self.assertEqual(parse_remembered_stock(self._row((1, 2, 3), "[]")), ((1, 2, 3), {}))

    def test_no_row_means_no_chest_was_ever_opened(self) -> None:
        empty = {"response": {"data": {"result": {"query": {"rows": []}}}}}
        self.assertIsNone(parse_remembered_stock(empty))

    def test_unreadable_json_is_not_guessed_at(self) -> None:
        # A half-written or foreign contents column must read as unknown, never as an empty chest:
        # "no food banked" would offer the hunt, which is the thing this read exists to prevent.
        self.assertIsNone(parse_remembered_stock(self._row((1, 2, 3), "{not json")))

    def test_the_edible_list_is_the_one_the_candidates_use(self) -> None:
        # Two copies of "what counts as food" is how the reader and the gate drift apart: the
        # chest would be filtered with one list and counted with the other.
        from mlaya.actions import EDIBLE
        self.assertEqual(set(EDIBLE_NAMES), set(EDIBLE))


class StorageNetworkParseTest(unittest.TestCase):
    """Every chest the server remembers, and the network fields of the logged state.

    One chest fills, the run sets another beside it (run79: a full chest, a full pack and
    nothing in the state that said *which* chest had closed). The newest-first read could only
    see the chest touched last; this one lists them all, and the cells' fullness is a state the
    log can answer afterwards.
    """

    @staticmethod
    def _payload(rows):
        return {"response": {"data": {"result": {"query": {"rows": list(rows)}}}}}

    def test_every_row_comes_back_with_its_own_counts(self) -> None:
        payload = self._payload([
            [1, 2, 3, json.dumps([{"item": "bread", "count": 4}])],
            [9, 10, 11, json.dumps([{"item": "cooked_beef", "count": 12},
                                    {"item": "cooked_beef", "count": 3}])]])
        self.assertEqual(parse_remembered_stocks(payload),
                         [((1, 2, 3), {"bread": 4}), ((9, 10, 11), {"cooked_beef": 15})])

    def test_a_broken_row_drops_without_blinding_the_others(self) -> None:
        # The network read's rows are independent containers: one torn `contents_json` must not
        # answer "no chest holds anything" for the chest next to it.
        payload = self._payload([
            [1, 2, 3, "{not json"],
            [9, 10, 11, json.dumps([{"item": "bread", "count": 2}])]])
        self.assertEqual(parse_remembered_stocks(payload), [((9, 10, 11), {"bread": 2})])

    def test_no_rows_is_not_a_failure_just_an_unopened_world(self) -> None:
        self.assertEqual(parse_remembered_stocks(self._payload([])), [])

    def test_the_network_fields_survive_the_log_round_trip(self) -> None:
        # `from_state` is the inverse of `as_state` for diagnosis after the fact -- and JSON
        # turns every tuple into a list, which has to come back hashable (a full chest is
        # tested against a set).
        import json as jsonlib
        g = GameState()
        g.chest_pos = (0, 71, 1)
        g.extra_chests = ((0, 71, 3), (1, 71, 2))
        g.chests_full = frozenset({(0, 71, 1)})
        g.banked_stores = (((0, 71, 3), "cooked_beef", 12),)
        g.banked = {"cooked_beef": 12}
        rebuilt = GameState.from_state(jsonlib.loads(jsonlib.dumps(g.as_state())))
        self.assertEqual(rebuilt.extra_chests, ((0, 71, 3), (1, 71, 2)))
        self.assertEqual(rebuilt.chests_full, frozenset({(0, 71, 1)}))
        self.assertEqual(rebuilt.banked_stores, (((0, 71, 3), "cooked_beef", 12),))
        # hashable, because every predicate that aims at a chest tests it against a set:
        self.assertIn((0, 71, 1), rebuilt.chests_full)

    def test_the_powder_cells_survive_the_log_round_trip(self) -> None:
        # The cells a stow actually filled are what `take_blaze_powder` aims at, so a logged
        # tick has to answer "where did the powder go" after the fact - tuples through JSON.
        import json as jsonlib
        g = GameState(powder_stowed=True,
                      powder_cells=((-196, -46, -98), (-190, -46, -100)))
        rebuilt = GameState.from_state(jsonlib.loads(jsonlib.dumps(g.as_state())))
        self.assertEqual(rebuilt.powder_cells, ((-196, -46, -98), (-190, -46, -100)))

    def test_an_old_log_yields_an_empty_network_not_a_crash(self) -> None:
        # run79's own entries predate the fields: defaults, not guesses.
        g = GameState.from_state({"chest_pos": [0, 71, 1]})
        self.assertEqual(g.extra_chests, ())
        self.assertEqual(g.chests_full, frozenset())
        self.assertEqual(g.banked_stores, ())

    def test_the_narration_names_the_network_and_its_fullness(self) -> None:
        # Every narrative clause is capitalised by the state itself; the assertions follow it.
        g = GameState(chest_pos=(0, 71, 1))
        plain = g.narrative()
        self.assertIn("chest stands at the base", plain)
        self.assertNotIn("(it is full)", plain)
        g.chests_full = frozenset({(0, 71, 1)})
        self.assertIn("chest stands at the base (it is full)", g.narrative())
        g.extra_chests = ((0, 71, 3),)
        g.chests_full = frozenset({(0, 71, 1), (0, 71, 3)})
        self.assertIn("chest stands at the base, plus 1 overflow chest beside it: every one of "
                      "them is full", g.narrative())
        # Room left in the network is *not* told as full -- "build another chest" must not
        # stand next to a chest that could still take the surplus.
        g.chests_full = frozenset({(0, 71, 1)})
        text = g.narrative()
        self.assertIn("plus 1 overflow chest beside it (the main one is full)", text)
        self.assertNotIn("every one of them is full", text)


class RememberedGateParseTest(unittest.TestCase):
    """The Nether side of the last crossing, read out of the server's event database.

    A run that starts in the Nether saw no crossing, and the cell it stands on says nothing
    about a portal (run74: (-469,57,421) of basalt taken for the "arrival", the bot starved at
    it). The server records every `player_dimension_change` with the position it delivered the
    bot at, in the database that outlives the death and the restart - the same kind of read as
    the remembered chest, for the way out. `REMEMBERED_GATE_SQL` asks SQLite for the newest
    arrival in the Nether; the parser floors it to the cell the gate stands in.
    """

    @staticmethod
    def _row(cells):
        return {"response": {"data": {"result": {"query": {"rows": [list(cells)]}}}}}

    def test_the_arrival_position_becomes_its_cell(self) -> None:
        # The live crossing at the base portal's Nether side: the delivered position is the
        # cell centre, and the floor is the cell the portal block stands in.
        row = self._row((33.42975141085543, 55, 34.5))
        self.assertEqual(parse_remembered_gate(row), (33, 55, 34))

    def test_negative_positions_floor_the_right_way(self) -> None:
        # A delivery west of the origin must not round toward zero: the gate cell is what the
        # walk is asked for, and int(-2.5) would put the bot a cell east of it.
        self.assertEqual(parse_remembered_gate(self._row((-2.5, 31.0, -10.2))), (-3, 31, -11))

    def test_no_row_is_no_crossing_not_a_guess(self) -> None:
        empty = {"response": {"data": {"result": {"query": {"rows": []}}}}}
        self.assertIsNone(parse_remembered_gate(empty))
        self.assertIsNone(parse_remembered_gate({}))

    def test_the_row_is_asked_of_the_events_the_server_writes(self) -> None:
        # The read has to name the server's own crossing event, and the Nether side of it:
        # `to`, not `from` - the from side is the cell the bot left.
        self.assertIn("player_dimension_change", REMEMBERED_GATE_SQL)
        self.assertIn("'the_nether'", REMEMBERED_GATE_SQL)
        self.assertIn("$.to", REMEMBERED_GATE_SQL)


class NarrativeSaysWhatTheRulesDoTest(unittest.TestCase):
    """The narrative is the model's only view of the state, so its sentences have to mean something.

    Two defects, both about the same thing - a sentence in the state text that no rule acts on.

    Health and hunger mixed a category with a measurement in one slot: `health is full` beside
    `hunger is 12/20`. One variable, two ways of being written, and the model has to serve both.
    Each is now a number plus the band the candidates actually use, so the text cannot describe a
    state the loop is not acting on.

    The middle hunger band is the one worth naming: between `EAT_HUNGER` and `REGEN_HUNGER` a
    wounded bot neither eats nor heals, and it waits there until something hits it. A three-way
    "fine / middling / critical" wording folds that four-point gap into the top band.
    """

    def test_a_wounded_bot_says_its_health_in_one_way_not_two(self) -> None:
        text = GameState(health=12.0, food=20.0).narrative()
        self.assertIn("Health 12/20, wounded", text)
        self.assertNotIn("Health is 12/20", text)

    def test_hunger_names_the_band_the_gate_uses(self) -> None:
        self.assertIn("Hunger 9/20, below the eating floor",
                      GameState(health=20.0, food=9.0).narrative())
        self.assertIn("Hunger 17/20, not regenerating",
                      GameState(health=20.0, food=17.0).narrative())
        self.assertIn("Hunger 19/20, regenerating",
                      GameState(health=20.0, food=19.0).narrative())

    def test_the_bands_are_the_gates_themselves_not_a_parallel_ladder(self) -> None:
        # If these drift from `actions`, the narrative starts describing states the loop ignores.
        from mlaya.actions import EAT_HUNGER, REGEN_HUNGER
        from mlaya.perception import EAT_HUNGER as P_EAT, REGEN_HUNGER as P_REGEN
        self.assertEqual(EAT_HUNGER, P_EAT)
        self.assertEqual(REGEN_HUNGER, P_REGEN)

    def test_a_full_barrel_still_reads_as_full(self) -> None:
        # The bands are for the numbers, not a rewrite of the easy case.
        text = GameState(health=20.0, food=20.0).narrative()
        self.assertIn("Health is full", text)
        self.assertIn("Hunger is full", text)

    def test_another_player_is_not_in_the_state_text(self) -> None:
        # Nothing in the candidate list has ever keyed on a second player - the goal is a solo
        # playthrough - so the sentence spent the model's attention on noise. It stays on the
        # state, so the log still records who was standing there.
        gs = GameState(health=20.0, food=20.0)
        gs.players = ["Sanskrita"]
        self.assertNotIn("players nearby", gs.narrative())
        self.assertEqual(gs.players, ["Sanskrita"])

    def test_a_cave_floor_does_not_read_as_the_surface(self) -> None:
        # `onGround` is physics contact - true a hundred blocks under the sky too. run78 stood
        # at y=-38 with "(on the ground)" in every tick, and that reads as "surface" to anyone.
        # Say what the flag means, nothing more.
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (-35.0, -38.0, 93.0)
        gs.on_ground = True
        text = gs.narrative()
        self.assertIn("standing on solid floor", text)
        self.assertNotIn("(on the ground)", text)
        gs.on_ground = False
        self.assertIn("in the air", gs.narrative())

    def test_death_is_gone_once_the_gear_is_lost_anyway(self) -> None:
        # Outside `recover_drops`' own window there is nothing left to say about the death:
        # run78 read a 14-hour-old death from the *previous* world on every tick (the server
        # remembers deaths across world changes and restarts) while the label never fired.
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 64.0, 0.0)
        gs.death_pos = (-9.0, 72.0, 102.0)
        gs.last_death_cause = "MineAI was slain by Zombie"
        gs.last_death_age_s = 50462.0
        self.assertNotIn("Last death", gs.narrative())

    def test_the_open_window_names_the_site_it_asks_to_walk_back_to(self) -> None:
        # Inside the window the sentence is useful - and the distance to walk back, relative to
        # the bot, is what the recovery walk needs; the absolute cell is per-world noise.
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 64.0, 0.0)
        gs.death_pos = (-9.0, 72.0, 102.0)
        gs.last_death_cause = "MineAI was slain by Zombie"
        gs.last_death_age_s = 120.0
        text = gs.narrative()
        # The clause is capitalised by the narrative's own rule, like every other.
        self.assertIn("Last death: 120s ago MineAI was slain by Zombie", text)
        self.assertIn("103 blocks away", text)
        self.assertNotIn("z=102", text)
        self.assertIn("can still be recovered", text)

    def test_a_death_too_far_to_walk_is_not_offered_as_information(self) -> None:
        # Same reach as the label itself: 320 m out the gear is gone on foot anyway.
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 64.0, 0.0)
        gs.death_pos = (900.0, 70.0, 0.0)
        gs.last_death_cause = "MineAI was slain by Zombie"
        gs.last_death_age_s = 60.0
        self.assertNotIn("Last death", gs.narrative())


class RelativeNarrativeTest(unittest.TestCase):
    """Position, base and death read relative to the bot, never as absolute x/z.

    Raw x/z are per-world random and carry no transferable signal, so the head would only have
    to learn to ignore them - exactly why "players nearby" was cut. `y` survives as a depth band
    against the frame every world shares (bedrock ~-64, sea level ~63, diamonds below 16), and the
    base and the death site are told as a distance to walk plus a coarse up/down relation.
    `as_state()` keeps the absolute coordinates for the log and the System 2 rebuild.
    """

    def test_no_absolute_x_or_z_survives_in_the_narrative(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (123.0, 70.0, -456.0)
        gs.base = (200, 12, -400)
        gs.death_pos = (130.0, 68.0, -450.0)
        gs.last_death_cause = "MineAI was slain by Zombie"
        gs.last_death_age_s = 100.0
        text = gs.narrative()
        self.assertNotIn("x=", text)
        self.assertNotIn("z=", text)
        # The raw figures must not leak in some other shape either.
        self.assertNotIn("123", text)
        self.assertNotIn("456", text)

    def test_the_depth_band_reads_from_y_alone(self) -> None:
        surface = GameState(health=20.0, food=20.0)
        surface.pos = (0.0, 63.0, 0.0)
        surface.on_ground = True
        self.assertIn("At the surface (standing on solid floor)", surface.narrative())

        mid = GameState(health=20.0, food=20.0)
        mid.pos = (0.0, 30.0, 0.0)
        mid.on_ground = True
        self.assertIn("Underground (standing on solid floor)", mid.narrative())

        deep = GameState(health=20.0, food=20.0)
        deep.pos = (0.0, -38.0, 0.0)
        deep.in_water = True
        self.assertIn("Deep underground (in water)", deep.narrative())

    def test_the_base_is_a_distance_and_a_coarse_direction(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 70.0, 0.0)
        # Far away horizontally and deep below the bot -> a distance plus "far below".
        gs.base = (30, 20, 40)
        text = gs.narrative()
        self.assertRegex(text, r"Home base \d+ blocks away and far below")
        self.assertNotIn("home base at", text.lower())

    def test_a_base_within_reach_still_reads_as_at_the_base(self) -> None:
        # The proximity clause (within 4 blocks of the base in x/z) is unchanged.
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.3, 71.0, 1.4)
        gs.base = (0, 71, 1)
        text = gs.narrative()
        self.assertIn("At the home base", text)
        self.assertNotIn("blocks away", text)

    def test_the_base_level_with_the_bot_drops_the_direction(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.5, 70.0, 0.5)
        gs.base = (40, 70, 0)  # roughly level with the bot, well out in x/z
        text = gs.narrative()
        self.assertRegex(text, r"Home base \d+ blocks away\.")
        self.assertNotIn("and above", text)
        self.assertNotIn("and below", text)

    def test_the_death_site_is_a_distance_not_a_cell(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 64.0, 0.0)
        gs.death_pos = (3.0, 68.0, 0.0)  # 5 blocks away (3,4,0)
        gs.last_death_cause = "MineAI was slain by Zombie"
        gs.last_death_age_s = 60.0
        text = gs.narrative()
        self.assertIn("Last death: 60s ago MineAI was slain by Zombie 5 blocks away", text)
        self.assertIn("can still be recovered", text)


class RelativeRenderHelpersTest(unittest.TestCase):
    """The depth band and the vertical relation are the two coarse scales the narrative reads."""

    def test_depth_band_boundaries(self) -> None:
        from mlaya.perception import DEEP_Y, SURFACE_Y, _depth_band
        self.assertEqual(_depth_band(SURFACE_Y), "at the surface")
        self.assertEqual(_depth_band(SURFACE_Y - 1), "underground")
        self.assertEqual(_depth_band(DEEP_Y), "underground")
        self.assertEqual(_depth_band(DEEP_Y - 1), "deep underground")

    def test_vertical_relation_is_coarse(self) -> None:
        from mlaya.perception import _rough_vertical
        self.assertEqual(_rough_vertical(0.0), "level")
        self.assertEqual(_rough_vertical(5.0), "above")
        self.assertEqual(_rough_vertical(-5.0), "below")
        self.assertEqual(_rough_vertical(25.0), "far above")
        self.assertEqual(_rough_vertical(-25.0), "far below")


class RelativeNarrativeTest(unittest.TestCase):
    """Position, base and death speak in relations, not absolute cells.

    Raw x/z are per-world random and not transferable, so the head would only learn to ignore
    them - the same reason "players nearby" was cut. The depth band is read against the one frame
    every world shares (bedrock ~-64, sea level ~63, diamonds below 16), and the base and the
    death site are a distance to walk plus a coarse vertical relation. `as_state` still logs the
    absolute cells; only what Laya reads goes relative.
    """

    def test_the_depth_band_reads_against_the_shared_frame(self) -> None:
        from mlaya.perception import _depth_band, DEEP_Y, SURFACE_Y
        self.assertEqual(_depth_band(SURFACE_Y), "at the surface")
        self.assertEqual(_depth_band(72.0), "at the surface")
        self.assertEqual(_depth_band(30.0), "underground")
        self.assertEqual(_depth_band(DEEP_Y), "underground")
        self.assertEqual(_depth_band(DEEP_Y - 1), "deep underground")
        self.assertEqual(_depth_band(-38.0), "deep underground")

    def test_the_vertical_relation_is_coarse_and_signed(self) -> None:
        from mlaya.perception import _rough_vertical
        self.assertEqual(_rough_vertical(0.0), "level")
        self.assertEqual(_rough_vertical(3.0), "level")
        self.assertEqual(_rough_vertical(10.0), "above")
        self.assertEqual(_rough_vertical(40.0), "far above")
        self.assertEqual(_rough_vertical(-10.0), "below")
        self.assertEqual(_rough_vertical(-40.0), "far below")

    def test_position_says_the_band_not_the_cell(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (1234.0, 72.0, -987.0)
        gs.on_ground = True
        text = gs.narrative()
        self.assertIn("At the surface (standing on solid floor)", text)
        self.assertNotIn("x=", text)
        self.assertNotIn("z=", text)
        self.assertNotIn("y=", text)
        # The absolute cell is still logged for diagnosis and the S2 rebuild.
        self.assertEqual(gs.as_state()["position"], (1234.0, 72.0, -987.0))

    def test_the_base_is_a_distance_and_a_direction_not_a_cell(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 70.0, 0.0)
        gs.base = (30, 12, 0)  # far below and a walk off
        text = gs.narrative()
        self.assertIn("Home base", text)
        self.assertIn("blocks away", text)
        self.assertIn("far below", text)
        self.assertNotIn("x=", text)

    def test_standing_at_the_base_reads_as_the_base(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.base = (10, 64, 10)
        gs.pos = (10.4, 64.0, 10.4)  # inside the 4-block proximity box
        text = gs.narrative()
        self.assertIn("At the home base", text)
        self.assertNotIn("blocks away", text)

    def test_a_level_base_names_no_direction(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 64.0, 0.0)
        gs.base = (40, 64, 0)  # same depth, a walk away
        text = gs.narrative()
        self.assertIn("blocks away", text)
        self.assertNotIn("above", text)
        self.assertNotIn("below", text)

    def test_one_distance_word_not_two(self) -> None:
        # A block is a metre; the narrative says it one way so the head learns one scale.
        gs = GameState(health=20.0, food=20.0)
        gs.pos = (0.0, 64.0, 0.0)
        gs.base = (40, 64, 0)
        gs.resources = {"diamond_ore": [BlockHit(query="diamond_ore", name="diamond_ore",
                                                 x=5, y=60, z=0, distance=6.0)]}
        text = gs.narrative()
        self.assertIn("diamond_ore at 6 blocks", text)
        self.assertNotIn("at 6m", text)
        self.assertIn("blocks away", text)


class PhasePersonaTest(unittest.TestCase):
    """One sentence, one phase, both model interfaces from the same truth.

    The head is trained on exactly the rendering it meets at run time, so the phase has to be
    said once, from what the world gives off, and reused verbatim by the System 2 brief. A run
    still climbing may not claim to prepare for the nether, and a finished kit may not keep
    saying "climb" while every step since the sword is the crossing.
    """

    def _kit(self, gs: GameState) -> GameState:
        gs.dimension = "minecraft:overworld"
        gs.tools = {"pickaxe": "diamond", "sword": "diamond", "shield": "other"}
        gs.armor = {s: {"tier": "diamond", "item": f"diamond_{s}", "left": None, "max": None}
                    for s in ("helmet", "chestplate", "leggings", "boots")}
        return gs

    def test_the_climb_reads_as_the_climb(self) -> None:
        text = GameState(health=20.0, food=20.0).narrative()
        self.assertTrue(text.startswith(
            "You are a Minecraft player working toward a full diamond kit. "), text[:80])

    def test_a_finished_kit_reads_as_the_crossing(self) -> None:
        gs = self._kit(GameState(health=20.0, food=20.0))
        self.assertIn("You are a Minecraft player preparing for the nether. ", gs.narrative())

    def test_the_nether_and_the_end_own_their_work(self) -> None:
        gs = GameState(health=20.0, food=20.0)
        gs.dimension = "minecraft:the_nether"
        self.assertIn("killing blazes for blaze rods", gs.narrative())
        gs.dimension = "minecraft:the_end"
        self.assertIn("fighting the ender dragon", gs.narrative())

    def test_system_two_reads_the_same_sentence_from_the_state_dict(self) -> None:
        # `escalate_to_system2` receives `gs.as_state()`, not the object: the sentence must
        # survive the round trip or the two model faces get two different runs.
        from mlaya.perception import phase_persona
        state = {"dimension": "minecraft:overworld",
                 "tools": {"pickaxe": "diamond", "sword": "diamond"},
                 "armor": {s: {"tier": "diamond"}
                           for s in ("helmet", "chestplate", "leggings", "boots")}}
        self.assertEqual(phase_persona(state),
                         "You are a Minecraft player preparing for the nether")
        self.assertEqual(phase_persona({"dimension": "minecraft:overworld", "tools": {}, "armor": {}}),
                         "You are a Minecraft player working toward a full diamond kit")

    def test_no_phase_is_claimed_about_a_run_nothing_observed(self) -> None:
        # Without the banked powder nothing says the run is past the Nether haul, so no other
        # overworld field combination may produce the stronghold sentence.
        from mlaya.perception import PHASE_STRONGHOLD, phase_persona
        self.assertEqual(phase_persona({"dimension": "minecraft:overworld"}),
                         "You are a Minecraft player working toward a full diamond kit")
        self.assertNotIn(PHASE_STRONGHOLD, phase_persona({"dimension": "minecraft:overworld"}))

    def test_the_banked_powder_turns_the_overworld_to_the_stronghold(self) -> None:
        # The stronghold phase runs from the stow until the end portal stands active; both the
        # object and the logged dict must say so, and the Nether still owns its own sentence.
        from mlaya.perception import PHASE_NETHER, PHASE_STRONGHOLD, GameState, phase_persona
        state = {"dimension": "minecraft:overworld", "powder_stowed": True}
        self.assertEqual(phase_persona(state), PHASE_STRONGHOLD)
        g = GameState(dimension="minecraft:overworld", powder_stowed=True)
        self.assertEqual(phase_persona(g), PHASE_STRONGHOLD)
        self.assertEqual(phase_persona(GameState.from_state(g.as_state())), PHASE_STRONGHOLD)
        g.end_portal_active = True
        self.assertNotEqual(phase_persona(g), PHASE_STRONGHOLD)
        self.assertEqual(phase_persona({"dimension": "minecraft:the_nether",
                                        "powder_stowed": True}), PHASE_NETHER)


class FurnaceContentsParseTest(unittest.TestCase):
    """What a `use_container inspect` on a furnace reported.

    The answer shape is the live one from run50: `use_container inspect` at (229,62,222) against
    the base furnace answered `succeeded` with `container.blockName: "furnace"` and one filled
    slot. It is the only place the loop can learn what is in the furnace, and `collect_smelted`
    issues it precisely because it holds nothing -- so if this is dropped, the step inspects
    forever and never sends the `withdraw` that would empty the cell.
    """

    ANSWER = {"response": {"data": {"state": "settled", "output": {
        "action": "use_container", "durationMs": 761,
        "result": {"status": "succeeded", "container": {
            "dimension": "overworld", "x": 229, "y": 62, "z": 222,
            "operation": "inspect", "blockName": "furnace", "slotCount": 3,
            "contents": [{"slot": 2, "item": "cooked_beef", "count": 1}],
            "observedAt": "2026-10-18T18:12:24.083Z"}}}}}}

    def test_the_filled_slot_is_read_as_item_and_count(self) -> None:
        self.assertEqual(parse_furnace_contents(self.ANSWER), {"cooked_beef": 1})

    def test_the_three_slots_of_a_furnace_add_up_per_item(self) -> None:
        # Input, fuel and output share one window, so the same name can appear twice.
        answer = {"response": {"data": {"result": {"container": {
            "blockName": "furnace", "slotCount": 3,
            "contents": [{"slot": 0, "item": "raw_beef", "count": 4},
                         {"slot": 1, "item": "coal", "count": 3},
                         {"slot": 2, "item": "raw_beef", "count": 2}]}}}}}
        self.assertEqual(parse_furnace_contents(answer),
                         {"raw_beef": 6, "coal": 3})

    def test_an_empty_window_is_empty_not_a_failure(self) -> None:
        answer = {"response": {"data": {"result": {"container": {
            "blockName": "furnace", "slotCount": 3, "contents": []}}}}}
        self.assertEqual(parse_furnace_contents(answer), {})

    def test_an_answer_without_a_window_is_empty(self) -> None:
        # A refusal or a truncated answer must not read as a furnace full of nonsense: the loop
        # cooks instead, which is the safe way round.
        self.assertEqual(parse_furnace_contents({}), {})
        self.assertEqual(parse_furnace_contents(
            {"response": {"data": {"result": {"error": "CONTAINER_NOT_PRESENT"}}}}), {})

    def test_a_contents_list_without_a_block_name_is_not_taken_for_a_window(self) -> None:
        answer = {"response": {"data": {"result": {
            "inventory": {"contents": [{"item": "cooked_beef", "count": 1}]}}}}}
        self.assertEqual(parse_furnace_contents(answer), {})


class RescueGroundCastTest(unittest.TestCase):
    """The water rescue reads its own, wider ground cast, and accepts a roofed foothold.

    Live run20: the bot poured its water bucket on a cave lava pool and stood in the 1x1 water
    cell at (105,32,-18). The escape cell, cobblestone at (105,32,-17) with `cave_air` overhead,
    sat 0.3 m off -- but the ground scan asked for the four soil names only, and `standable`
    demanded open sky, so `nearest_ground` aimed at grass 37 m off / 34 blocks up and the
    pathfinder stalled 20 s a tick, ~15 ticks running.
    """

    def test_the_cast_keeps_the_four_soil_names(self) -> None:
        self.assertTrue(set(GROUND_NAMES).issubset(RESCUE_GROUND_NAMES))

    def test_the_cast_covers_the_rock_a_cave_pour_leaves(self) -> None:
        for name in ("stone", "cobblestone"):
            self.assertIn(name, RESCUE_GROUND_NAMES)

    def test_the_cast_fits_one_view_blocks_find_call(self) -> None:
        # `view_blocks find` takes at most eight names per call, and the ground scan is one call.
        self.assertLessEqual(len(RESCUE_GROUND_NAMES), 8)

    def test_a_roofed_cell_is_a_rescue_foothold_but_not_open_ground(self) -> None:
        roofed = BlockHit(query="cobblestone", name="cobblestone", x=0, y=0, z=0,
                          distance=0.3, above="cave_air")
        self.assertTrue(roofed.rescue_standable)
        self.assertFalse(roofed.standable)


class PortalCornerParseTest(unittest.TestCase):
    """`parse_portal_corner` reads a lit portal's opening corner *past* `RESOURCE_RADIUS`, which
    `parse_resources` deliberately drops. That corner inverts `actions._portal_site`, so a restart
    can re-anchor its base on the portal a previous run left standing (`graph._adopt_portal_base`)
    - the 64 m scan alone never sees a portal the bot has wandered away from."""

    # One 2x3 opening at z=257, x=-154..-153, y=63..65 - the portal's own cells, listed well
    # outside the scan radius (`distance` above 64).
    _DATA = {"response": {"data": {"result": {"blocks": {"find": [
        {"name": "nether_portal", "found": 6, "listed": [
            {"x": -154, "y": 63, "z": 257, "name": "nether_portal", "distance": 76.0},
            {"x": -154, "y": 64, "z": 257, "name": "nether_portal", "distance": 76.2},
            {"x": -154, "y": 65, "z": 257, "name": "nether_portal", "distance": 76.4},
            {"x": -153, "y": 63, "z": 257, "name": "nether_portal", "distance": 76.1},
            {"x": -153, "y": 64, "z": 257, "name": "nether_portal", "distance": 76.3},
            {"x": -153, "y": 65, "z": 257, "name": "nether_portal", "distance": 76.5},
        ]},
        {"name": "white_bed", "found": 2, "listed": [
            {"x": -165, "y": 28, "z": 262, "name": "white_bed", "distance": 78.3},
        ]},
    ]}}}}}

    def test_reads_the_low_western_corner_past_the_scan(self) -> None:
        self.assertEqual(parse_portal_corner(self._DATA), (-154, 63, 257))

    def test_no_portal_hits_is_none(self) -> None:
        self.assertIsNone(parse_portal_corner({}))
        self.assertIsNone(parse_portal_corner({"response": {"data": {"result": {"blocks": {
            "find": [{"name": "nether_portal", "found": 0, "listed": []}]}}}}}))
        self.assertIsNone(parse_portal_corner({"response": {"data": {"result": {"blocks": {
            "find": [{"name": "diamond_ore", "found": 3, "listed": [
                {"x": 0, "y": 8, "z": 0, "name": "diamond_ore", "distance": 10.0}]}]}}}}}))


class BoxParseTest(unittest.TestCase):
    """`parse_box` flattens a `view_blocks box` read into {(x, y, z): block} for the base-site
    scan (the frame needs its whole bottom row on level ground, which the sparse `find` hits
    cannot show)."""

    _DATA = {"response": {"data": {"result": {"blocks": {"box": {
        "center": {"x": 10, "y": 64, "z": -5},
        "halfWidth": 1, "halfHeight": 1,
        "layers": [
            {"y": 65, "rows": [
                {"z": -6, "blocks": ["air", "air", "air"]},
                {"z": -5, "blocks": ["stone", "air", "stone"]},
                {"z": -4, "blocks": ["air", "air", "air"]},
            ]},
            {"y": 64, "rows": [
                {"z": -6, "blocks": ["dirt", "dirt", "dirt"]},
                {"z": -5, "blocks": ["dirt", "grass_block", "dirt"]},
                {"z": -4, "blocks": ["dirt", "dirt", "dirt"]},
            ]},
        ],
    }}}}}}

    def test_every_cell_maps_to_its_block_name(self) -> None:
        grid = parse_box(self._DATA)
        self.assertEqual(len(grid), 18)  # 2 layers * 3 z-rows * 3 x-blocks
        self.assertEqual(grid[(9, 64, -5)], "dirt")
        self.assertEqual(grid[(10, 64, -5)], "grass_block")
        self.assertEqual(grid[(11, 64, -5)], "dirt")
        self.assertEqual(grid[(10, 65, -5)], "air")
        self.assertEqual(grid[(11, 65, -5)], "stone")

    def test_malformed_box_is_empty(self) -> None:
        self.assertEqual(parse_box({}), {})
        self.assertEqual(parse_box({"response": {"data": {"result": {"blocks": {
            "box": {"center": None}}}}}}), {})


class FrameOpeningParseTest(unittest.TestCase):
    """`parse_cells` + `portal_interior`: the opening read that gates `clear_portal`.

    The opening is read cell by cell (`view_blocks cells`, not a box), so it needs its own
    parser; a block found there sits above the support layer and blocks the build.
    """

    _DATA = {"response": {"data": {"result": {"blocks": {"cells": [
        {"x": 1, "y": 2, "z": 3, "name": "air", "shape": "open"},
        {"x": 4, "y": 5, "z": 6, "name": "cobblestone", "shape": "solid"},
    ]}}}}}

    def test_every_cell_maps_to_its_block_name(self) -> None:
        self.assertEqual(parse_cells(self._DATA),
                         {(1, 2, 3): "air", (4, 5, 6): "cobblestone"})

    def test_malformed_cells_read_is_empty(self) -> None:
        self.assertEqual(parse_cells({}), {})
        self.assertEqual(parse_cells({"response": {"data": {"result": {"blocks": {}}}}}), {})

    def test_the_interior_is_the_two_by_three_opening_above_the_ground(self) -> None:
        # Base anchor (-197, -47, -98): the frame stands at z = -98 with its bottom row on the
        # ground, so the opening is the 2x3 interior at (-195..-194, -46..-44).
        cells = portal_interior((-197, -47, -98))
        self.assertEqual(len(cells), 6)
        self.assertEqual(set(cells),
                         {(x, y, -98) for x in (-195, -194) for y in (-46, -45, -44)})


class FrameVolumeTest(unittest.TestCase):
    """`frame_volume`: the whole 5x5 plane the build may touch, opening included.

    The frame read has to cover the border as well as the opening: a container the build refuses
    can sit in a border cell (`clear_frame`), and only a name for that exact cell lets
    `collect_block` be asked for it.
    """

    def test_the_plane_is_five_by_five_and_contains_the_opening(self) -> None:
        base = (228, 62, 222)
        cells = frame_volume(base)
        self.assertEqual(len(cells), 25)
        self.assertEqual(set(cells), {(x, y, 222) for x in range(228, 233)
                                      for y in range(62, 67)})
        self.assertTrue(set(portal_interior(base)) <= set(cells))

    def test_the_run27_frame_corner_is_in_the_plane(self) -> None:
        # Run26's furnace stood at (229, 62, 222), the new base's frame corner: the scan must
        # return that cell so `clear_frame` can name and dig it.
        self.assertIn((229, 62, 222), frame_volume((228, 62, 222)))


class FrameObsidianCellsTest(unittest.TestCase):
    """`frame_obsidian_cells`: the ten blocks a finished nether portal frame holds.

    A portal needs obsidian in every block that touches the 2x3 opening; the four outer corners
    are free, which is why the run fills them with cobblestone. Reading exactly these cells tells
    a finished frame from a partial one (`_frame_complete_at_base`).
    """

    def test_the_ten_cells_are_the_blocks_touching_the_opening(self) -> None:
        base = (-197, -47, -98)
        cells = frame_obsidian_cells(base)
        self.assertEqual(len(cells), 10)
        # Bottom and top rows between the corners, and both columns between the rows.
        self.assertEqual(
            set(cells),
            {(-195, -47, -98), (-194, -47, -98),          # bottom middle
             (-195, -43, -98), (-194, -43, -98),          # top middle
             (-196, -46, -98), (-196, -45, -98), (-196, -44, -98),   # left column
             (-193, -46, -98), (-193, -45, -98), (-193, -44, -98)})  # right column

    def test_the_cells_are_inside_the_frame_plane(self) -> None:
        base = (228, 63, 221)
        self.assertTrue(set(frame_obsidian_cells(base)) <= set(frame_volume(base)))


class ParseRefusedCellsTest(unittest.TestCase):
    """`parse_refused_cells` reads the frame cells a build refused to touch.

    `build_structure` cannot dig a container in a frame cell in place: the movement policy refuses
    a break it was not asked for. The refusal names the cell, and `clear_frame` digs it with the
    explicit `collect_block` the policy does allow (live, run27: run26's furnace on the new base's
    frame corner refused on every `build_portal`).
    """

    def test_the_refused_cell_is_read(self) -> None:
        text = ("[BUILD_INCOMPLETE] 1 cell still wrong: 1 were refused (229,62,222: movement "
                "policy prohibits breaking this block).")
        self.assertEqual(parse_refused_cells(text), [(229, 62, 222)])

    def test_several_refused_cells_keep_their_order(self) -> None:
        text = ("[BUILD_INCOMPLETE] 2 cells still wrong: 2 were refused (1,64,-3: movement policy "
                "prohibits breaking this block), (1,65,-3: movement policy prohibits breaking this "
                "block).")
        self.assertEqual(parse_refused_cells(text), [(1, 64, -3), (1, 65, -3)])

    def test_a_cell_refused_for_another_reason_is_not_dug(self) -> None:
        # A route failure is retried by the build; digging it out from under the build is wrong.
        text = "[BUILD_INCOMPLETE] 1 cell still wrong: 1 were refused (5,6,7: no route)."
        self.assertEqual(parse_refused_cells(text), [])

    def test_any_other_answer_or_empty_text_is_empty(self) -> None:
        self.assertEqual(parse_refused_cells("[BUILD_COMPLETE] 16 placed"), [])
        self.assertEqual(parse_refused_cells("(1,2,3: movement policy prohibits breaking this block)"), [])
        self.assertEqual(parse_refused_cells(""), [])
        self.assertEqual(parse_refused_cells(None), [])


class ArmourParseTest(unittest.TestCase):
    """Worn armour must be read from the inventory stack location (head/torso/legs/feet).
    `situation.tools.armour` lists the carried set per class whether or not the piece is on
    the body, so treating it as "worn" reported armour that was still in the backpack."""

    def test_carried_armour_stays_inventory(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 28, "location": "main", "name": "diamond_chestplate", "count": 1},
            ]},
            "tools": {"armour": [
                {"class": "chestplate", "tier": "diamond", "item": "diamond_chestplate",
                 "slot": 28},
            ]},
        })
        gs = parse_status(st)
        self.assertEqual(gs.inventory.get("diamond_chestplate"), 1)
        self.assertEqual(gs.armor, {})

    def test_worn_armour_is_not_also_carried(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 6, "location": "torso", "name": "diamond_chestplate", "count": 1,
                 "durability": {"remaining": 528, "maximum": 528}},
                {"slot": 5, "location": "head", "name": "diamond_helmet", "count": 1,
                 "durability": {"remaining": 363, "maximum": 363}},
            ]},
            "tools": {"armour": [
                {"class": "helmet", "tier": "diamond", "item": "diamond_helmet", "slot": 5},
                {"class": "chestplate", "tier": "diamond", "item": "diamond_chestplate",
                 "slot": 6},
            ]},
        })
        gs = parse_status(st)
        self.assertNotIn("diamond_chestplate", gs.inventory)
        self.assertNotIn("diamond_helmet", gs.inventory)
        self.assertEqual(gs.armor["chestplate"]["item"], "diamond_chestplate")
        self.assertEqual(gs.armor["chestplate"]["tier"], "diamond")
        self.assertEqual(gs.armor["chestplate"]["max"], 528)
        self.assertEqual(gs.armor["helmet"]["item"], "diamond_helmet")
        self.assertEqual(gs.armor["helmet"]["left"], 363)


class OffhandParseTest(unittest.TestCase):
    """The off-hand item reports a location of its own and leaves the carried inventory, exactly
    like armour. This is what tells the loop a shield is actually worn (`equip_shield` done)."""

    def test_offhand_item_is_not_also_carried(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 40, "location": "off-hand", "name": "shield", "count": 1},
            ]},
        })
        gs = parse_status(st)
        self.assertEqual(gs.offhand, "shield")
        self.assertNotIn("shield", gs.inventory)
        self.assertIn("Off-hand: shield", gs.narrative())

    def test_carried_shield_leaves_the_offhand_empty(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 20, "location": "main", "name": "shield", "count": 1},
            ]},
        })
        gs = parse_status(st)
        self.assertIsNone(gs.offhand)
        self.assertEqual(gs.inventory.get("shield"), 1)


class HostileParseTest(unittest.TestCase):
    """`view_status.nearby.hostiles` rows carry `distance` directly; `mobs` rows nest it under
    `nearest`. Reading only the nested form left every distance None, so the narrative said
    "spider x1" with no range and the danger gate deferred every tick over a mob it could not
    place (live: a zombie 13.5 m away in a cave, health and hunger full, the bot stood still)."""

    def test_direct_hostile_distance_is_read(self) -> None:
        st = _status({"nearby": {"hostiles": [
            {"name": "zombie", "distance": 13.0, "position": {"x": -43, "y": 44, "z": -27}},
        ]}})
        gs = parse_status(st)
        self.assertEqual(gs.hostiles, [("zombie", 1, 13.0)])
        self.assertEqual(gs.nearest_hostile, 13.0)
        self.assertIn("zombie x1 at 13 blocks", gs.narrative())

    def test_nested_hostile_distance_is_read_from_the_mob_list(self) -> None:
        st = _status({"nearby": {"mobs": [
            {"name": "spider", "kind": "hostile", "count": 2,
             "nearest": {"distance": 3.5, "position": {"x": 0, "y": 64, "z": 0}}},
        ]}})
        gs = parse_status(st)
        self.assertEqual(gs.hostiles, [("spider", 2, 3.5)])

    def test_hostile_without_a_distance_is_still_listed(self) -> None:
        st = _status({"nearby": {"hostiles": [{"name": "creeper"}]}})
        gs = parse_status(st)
        self.assertEqual(gs.hostiles, [("creeper", 1, None)])
        self.assertIn("creeper x1", gs.narrative())


class PiglinBarterParseTest(unittest.TestCase):
    """`nearby.mobs` rows are grouped by name:age, so the barter partner is the nearest *adult*
    piglin with its loaded `entityId` - the exact id `barter` demands. A baby piglin id is refused
    by the server, so babies are not listed at all."""

    def test_adult_piglins_are_listed_with_their_ids(self) -> None:
        st = _status({"nearby": {"mobs": [
            {"name": "piglin", "kind": "hostile", "age": "adult", "count": 3,
             "nearest": {"entityId": 412, "distance": 11.5,
                        "position": {"x": 3, "y": 64, "z": -2}}},
            {"name": "piglin", "kind": "hostile", "age": "baby", "count": 1,
             "nearest": {"entityId": 413, "distance": 4.0,
                        "position": {"x": 1, "y": 64, "z": -2}}},
        ]}})
        gs = parse_status(st)
        self.assertEqual(gs.piglins, [(412, 11.5)])
        self.assertEqual(gs.piglin_nearest(), (412, 11.5))

    def test_a_baby_piglin_is_not_a_barter_partner(self) -> None:
        st = _status({"nearby": {"mobs": [
            {"name": "piglin", "kind": "hostile", "age": "baby", "count": 2,
             "nearest": {"entityId": 9, "distance": 3.0}},
        ]}})
        gs = parse_status(st)
        self.assertEqual(gs.piglins, [])
        self.assertIsNone(gs.piglin_nearest())

    def test_the_nearest_adult_is_the_partner(self) -> None:
        st = _status({"nearby": {"mobs": [
            {"name": "piglin", "kind": "hostile", "age": "adult", "count": 2,
             "nearest": {"entityId": 5, "distance": 27.0}},
            {"name": "piglin", "kind": "hostile", "age": "adult", "count": 1,
             "nearest": {"entityId": 6, "distance": 9.0}},
        ]}})
        gs = parse_status(st)
        self.assertEqual(gs.piglin_nearest(), (6, 9.0))

    def test_the_piglin_list_survives_a_log_round_trip(self) -> None:
        # The id is a loop-owned fact like `chest_pos`: without it in `as_state` a recorded tick
        # cannot be asked why the barter was offered or skipped.
        st = _status({"nearby": {"mobs": [
            {"name": "piglin", "kind": "hostile", "age": "adult", "count": 1,
             "nearest": {"entityId": 77, "distance": 6.0}},
        ]}})
        gs = parse_status(st)
        self.assertEqual(GameState.from_state(gs.as_state()).piglin_nearest(), (77, 6.0))


class ParseSurfaceHeightTest(unittest.TestCase):
    """`parse_surface_height` reads the surface feet cell from a `navigate` height-required stop.

    A height-less `navigate` resolves an unambiguous column by itself but refuses to guess between
    floors: it stops with the floors it found, highest first. That is the one surface signal the
    client has -- `view_blocks` carries no heightmap -- and it is what `surface_up` climbs back to.
    """

    def test_the_surface_is_the_highest_listed_floor(self) -> None:
        text = ("[NAVIGATION_STOPPED] Pathfinder stopped: [NAVIGATION_HEIGHT_REQUIRED] Column -47, "
                "4 has highest block podzol at y=73; observed standable feet heights: 74, -15, "
                "-54. Supply y to choose a floor.")
        self.assertEqual(parse_surface_height(text), 74)

    def test_a_single_floor_still_reads(self) -> None:
        text = "[NAVIGATION_HEIGHT_REQUIRED] Column 0, 0 ... standable feet heights: 64."
        self.assertEqual(parse_surface_height(text), 64)

    def test_a_real_navigation_or_other_text_is_none(self) -> None:
        self.assertIsNone(parse_surface_height("[BUCKET_FAILED] the use of bucket did not take"))
        self.assertIsNone(parse_surface_height(""))
        self.assertIsNone(parse_surface_height(None))
        self.assertIsNone(parse_surface_height("standable feet heights: none."))


class ParseGroundHeightTest(unittest.TestCase):
    """`parse_ground_height` reads the ground a `[NAVIGATION_TARGET_UNSUPPORTED]` stop names.

    A `navigate` whose target cell has no footing reports the height it *did* find in the column,
    which is what a home base lost to its own digging has to be re-pinned to (`graph._note_base`).
    """

    def test_the_reported_ground_height_is_read(self) -> None:
        text = ("[NAVIGATION_TARGET_UNSUPPORTED] No usable footing was observed within 1 blocks of "
                "419, 74, 594. The ground in that column is at y=70. Choose an observed supported "
                "height, or pass build: true to scaffold to this point deliberately.")
        self.assertEqual(parse_ground_height(text), 70)

    def test_a_negative_ground_height_reads(self) -> None:
        text = "[NAVIGATION_TARGET_UNSUPPORTED] ... The ground in that column is at y=-16."
        self.assertEqual(parse_ground_height(text), -16)

    def test_another_failure_or_empty_text_is_none(self) -> None:
        self.assertIsNone(parse_ground_height("[NAVIGATION_STOPPED] repeated_movement_failure"))
        self.assertIsNone(parse_ground_height("[NAVIGATION_TARGET_UNSUPPORTED] no height given"))
        self.assertIsNone(parse_ground_height(""))
        self.assertIsNone(parse_ground_height(None))


class ParseBlockAboveTest(unittest.TestCase):
    """`parse_block_above` reads the cell whose block stops a chest lid.

    `use_container` refuses to open a chest under an opaque full cube and names that block:
    `[CHEST_BLOCKED] The chest at (229, 62, 224) cannot open: cobblestone is directly above it at
    (229, 63, 224).` The cell after "above it at" is what `stow_surplus` has to dig so the lid
    can swing (live, run26: a buried base put a cobblestone on the chest and the deposit failed on
    every tick, so the full pack could never be unloaded).
    """

    def test_the_block_above_the_chest_is_read(self) -> None:
        text = ("[CHEST_BLOCKED] The chest at (229, 62, 224) cannot open: cobblestone is directly "
                "above it at (229, 63, 224).")
        self.assertEqual(parse_block_above(text), (229, 63, 224))

    def test_negative_coordinates_read(self) -> None:
        text = "[CHEST_BLOCKED] The chest at (-3, 64, 7) cannot open: stone is directly above it at (-3, 65, 7)."
        self.assertEqual(parse_block_above(text), (-3, 65, 7))

    def test_any_other_failure_or_empty_text_is_none(self) -> None:
        self.assertIsNone(parse_block_above("[CHEST_CLEARANCE_UNLOADED] The cell above is not loaded."))
        self.assertIsNone(parse_block_above("[INVENTORY_FULL] No free slot"))
        self.assertIsNone(parse_block_above(""))
        self.assertIsNone(parse_block_above(None))


if __name__ == "__main__":
    unittest.main()
