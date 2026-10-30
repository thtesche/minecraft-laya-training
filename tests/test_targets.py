"""M4 parameter layer: target candidates from `view_blocks` hits and their binding."""
from __future__ import annotations

import math
import unittest

from mlaya.actions import resolve
from mlaya.perception import BlockHit, GameState
from mlaya.targets import (STEP_AWAY_MIN, Target, _compass, base_target, bed_frame_site, boxed_in,
                           flat_build_site, frontier_heading, ground_targets, nearest_ground,
                           step_away_cell)


def hit(query: str, name: str, x: int, y: int, z: int, distance: float,
        above: str | None = "air", below: str | None = "stone") -> BlockHit:
    return BlockHit(query=query, name=name, x=x, y=y, z=z, distance=distance,
                    above=above, below=below)


class CompassTest(unittest.TestCase):
    def test_cardinal_points(self) -> None:
        self.assertEqual(_compass(0, -10), "north")   # -z is north
        self.assertEqual(_compass(10, 0), "east")
        self.assertEqual(_compass(0, 10), "south")
        self.assertEqual(_compass(-10, 0), "west")

    def test_zero_offset(self) -> None:
        self.assertEqual(_compass(0, 0), "here")


class GroundTargetsTest(unittest.TestCase):
    def _gs(self) -> GameState:
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        gs.resources = {
            "stone": [
                hit("stone", "stone", 2, 60, 2, 2.8),        # nearest
                hit("stone", "stone", 30, 60, 0, 30.0),      # farthest
            ],
            "grass_block": [
                hit("grass_block", "grass_block", 0, 60, 20, 20.0),
                hit("grass_block", "grass_block", 0, 60, 20, 19.9),  # same column, dropped
            ],
            "logs": [  # not a destination query
                hit("logs", "acacia_log", 40, 70, 0, 40.0),
            ],
            "iron_ore": [  # not a destination query
                hit("iron_ore", "iron_ore", 50, 40, 0, 50.0),
            ],
        }
        return gs

    def test_only_destination_queries(self) -> None:
        labels = " ".join(t.label for t in ground_targets(self._gs()))
        self.assertNotIn("acacia_log", labels)
        self.assertNotIn("iron_ore", labels)

    def test_filters_not_standable(self) -> None:
        gs = self._gs()
        gs.resources["stone"].append(hit("stone", "stone", 10, 55, 0, 5.0, above="stone"))
        labels = [t.label for t in ground_targets(gs)]
        self.assertNotIn("stone@10,0", labels)

    def test_dedupes_by_column(self) -> None:
        labels = [t.label for t in ground_targets(self._gs())]
        self.assertEqual(labels.count("grass_block@0,20"), 1)

    def test_farthest_first(self) -> None:
        distances = [t.hit.distance for t in ground_targets(self._gs())]
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_label_and_text(self) -> None:
        targets = ground_targets(self._gs())
        farthest = targets[0]
        self.assertEqual(farthest.label, "stone@30,0")
        self.assertIn("30m", farthest.text)
        self.assertIn("east", farthest.text)
        self.assertIn("open above", farthest.text)

    def test_near_ground_dropped_when_far_exists(self) -> None:
        # stone@2,2 is underfoot; a farther destination exists, so it is not offered.
        labels = [t.label for t in ground_targets(self._gs())]
        self.assertNotIn("stone@2,2", labels)

    def test_boxed_in_falls_back_to_nearest(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        gs.resources = {"stone": [hit("stone", "stone", 1, 60, 0, 1.0)]}
        labels = [t.label for t in ground_targets(gs)]
        self.assertEqual(labels, ["stone@1,0"])

    def test_empty_without_resources(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        self.assertEqual(ground_targets(gs), [])


class BoxedInTest(unittest.TestCase):
    """`boxed_in` is the signal that turns an `explore` into a frontier push."""

    def test_true_when_only_adjacent_ground(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        gs.resources = {"stone": [hit("stone", "stone", 1, 60, 0, 1.0)]}
        self.assertTrue(boxed_in(gs))

    def test_false_when_outward_ground_exists(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        gs.resources = {"stone": [hit("stone", "stone", 30, 60, 0, 30.0),
                                  hit("stone", "stone", 1, 60, 0, 1.0)]}
        self.assertFalse(boxed_in(gs))

    def test_true_without_resources(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        self.assertTrue(boxed_in(gs))

    def test_ignores_non_standable_outward_ground(self) -> None:
        # A far block with no air above is not a destination; the bot is still boxed in.
        gs = GameState()
        gs.pos = (0.0, 60.0, 0.0)
        gs.resources = {"stone": [hit("stone", "stone", 30, 60, 0, 30.0, above="stone"),
                                  hit("stone", "stone", 1, 60, 0, 1.0)]}
        self.assertTrue(boxed_in(gs))


class FrontierHeadingTest(unittest.TestCase):
    """The heading points away from the home base, into ground not worked yet."""

    def test_away_from_the_base_on_each_side(self) -> None:
        gs = GameState()
        gs.base = (0, 64, 0)
        for pos, expected in (((0.0, 64.0, -100.0), 0.0),    # north
                              ((100.0, 64.0, 0.0), 90.0),     # east
                              ((0.0, 64.0, 100.0), 180.0),    # south
                              ((-100.0, 64.0, 0.0), 270.0)):  # west
            gs.pos = pos
            self.assertEqual(frontier_heading(gs), expected, pos)

    def test_north_without_a_base(self) -> None:
        gs = GameState()
        gs.pos = (5.0, 64.0, 5.0)
        self.assertEqual(frontier_heading(gs), 0.0)

    def test_north_on_the_base(self) -> None:
        gs = GameState()
        gs.base = (5, 64, 5)
        gs.pos = (5.0, 64.0, 5.0)
        self.assertEqual(frontier_heading(gs), 0.0)


class BaseTargetTest(unittest.TestCase):
    """The home-base pick: nearest open ground overhead, not a deep cave floor or stone."""

    def test_picks_the_open_ground_overhead(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 64.0, 0.0)
        gs.resources = {
            "grass_block": [hit("grass_block", "grass_block", -1, 70, -2, 6.0)],
            "stone": [hit("stone", "stone", 2, 65, 0, 2.0)],
        }
        target = base_target(gs)
        self.assertEqual(target.label, "grass_block@-1,-2")
        self.assertEqual(target.pos, (-1, 70, -2))

    def test_prefers_ground_overhead_not_a_deep_cave_floor(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 64.0, 0.0)
        gs.resources = {"grass_block": [
            hit("grass_block", "grass_block", 0, 71, 0, 4.0),
            hit("grass_block", "grass_block", 0, 50, 0, 3.0),   # deeper, nearer
        ]}
        self.assertEqual(base_target(gs).pos, (0, 71, 0))

    def test_below_the_bot_takes_the_highest_ground_not_the_deepest_hole(self) -> None:
        # Live run26: the bot stood on the surface at y=64 beside ground at y=63 and a dug-out
        # dip at y=61. Nothing sat at or above its feet, and the old fallback -- `min` over
        # every hit -- anchored the base on the dip floor, two blocks under the surface; the
        # frame, chest and furnace then went into the ground around it. Take the highest ground.
        gs = GameState()
        gs.pos = (0.0, 64.0, 0.0)
        gs.resources = {"grass_block": [
            hit("grass_block", "grass_block", 1, 63, 0, 1.5),
            hit("grass_block", "grass_block", 2, 61, 0, 2.5),   # the dip floor: lower, nearer
        ]}
        self.assertEqual(base_target(gs).pos, (1, 63, 0))

    def test_none_without_ground(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 64.0, 0.0)
        gs.resources = {"stone": [hit("stone", "stone", 2, 65, 0, 2.0)]}
        self.assertIsNone(base_target(gs))


class NearestGroundTest(unittest.TestCase):
    """The water rescue's target: the closest dry standable cell at any level."""

    def test_picks_the_nearest_cell_below_the_waterline(self) -> None:
        # `base_target` would drop the shore for sitting below the floating bot; the rescue must
        # not, because that lower dry cell is exactly the ground the bot can still climb onto.
        gs = GameState()
        gs.pos = (0.0, 63.0, 0.0)
        gs.resources = {"sand": [
            hit("sand", "sand", 40, 60, 0, 40.0),
            hit("sand", "sand", 3, 62, 1, 3.2),
        ]}
        self.assertEqual(nearest_ground(gs).pos, (3, 62, 1))

    def test_a_shore_at_the_bot_level_beats_a_nearest_cave_floor(self) -> None:
        # Live run19b, to the block: treading water at (-181,62,-84), the only standable hit
        # inside 64 m was a cave floor 32 m **below** -- a dive the pathfinder cannot walk
        # ("closest node was -182,59,-85"). The shore at the bot's own level sat 65.1 m off,
        # just past `RESOURCE_RADIUS`, so only the wider rescue scan lists it. Prefer the
        # waterline cell.
        gs = GameState()
        gs.pos = (-181.5, 62.1, -84.5)
        floor = hit("grass_block", "grass_block", -182, 30, -85, 32.0)
        gs.resources = {"grass_block": [floor]}
        gs.rescue_ground = {"grass_block": [floor, hit("grass_block", "grass_block",
                                                       -117, 62, -88, 65.1)]}
        self.assertEqual(nearest_ground(gs).pos, (-117, 62, -88))

    def test_a_roofed_ledge_at_the_bot_level_beats_far_open_grass(self) -> None:
        # Live run20, to the block: the bot poured its water bucket on a cave lava pool and stood
        # in the 1x1 water cell at (105,32,-18). The escape, cobblestone at (105,32,-17) with
        # `cave_air` overhead, sat 0.3 m off -- but the ground scan asked for the four soil names
        # only and `standable` demanded open sky, so the rescue aimed at grass 37 m off / 34 blocks
        # up instead and the pathfinder stalled 20 s a tick, ~15 ticks running.
        gs = GameState()
        gs.pos = (105.5, 32.05, -17.3)
        gs.rescue_ground = {
            "cobblestone": [hit("cobblestone", "cobblestone", 105, 32, -17, 0.6,
                                above="cave_air")],
            "grass_block": [hit("grass_block", "grass_block", 94, 67, -24, 37.7)],
        }
        self.assertEqual(nearest_ground(gs).pos, (105, 32, -17))

    def test_a_far_roofed_floor_does_not_beat_open_sky(self) -> None:
        # The run19b dive must stay out: a roofed cell counts only at the waterline, so a cave
        # floor far below the swimming body is still not a target a swim completes.
        gs = GameState()
        gs.pos = (0.0, 63.0, 0.0)
        gs.rescue_ground = {
            "stone": [hit("stone", "stone", 1, 40, 0, 23.0, above="cave_air")],
            "grass_block": [hit("grass_block", "grass_block", 20, 63, 0, 20.0)],
        }
        self.assertEqual(nearest_ground(gs).pos, (20, 63, 0))

    def test_a_deep_floor_is_still_taken_without_a_shore(self) -> None:
        # The preference must not drop the target a world actually offers: with no waterline
        # cell the nearest any-level cell stands, exactly as before the band.
        gs = GameState()
        gs.pos = (0.0, 63.0, 0.0)
        gs.resources = {"gravel": [hit("gravel", "gravel", 0, 40, 5, 23.0)]}
        self.assertEqual(nearest_ground(gs).pos, (0, 40, 5))

    def test_no_waterline_cell_climbs_instead_of_diving(self) -> None:
        # Live run115, to the block: treading water at (-0.5,62.4,281.5), the rescue scan listed a
        # cave floor 29 blocks below and dry ground 8 blocks up, and the rescue aimed down for 281
        # decisions, every swim timing out ("closest node was -1,59,281"). A floating body climbs.
        gs = GameState()
        gs.pos = (-0.5, 62.4, 281.5)
        gs.rescue_ground = {
            "stone": [hit("stone", "stone", -1, 33, 281, 29.4, above="cave_air")],
            "cobblestone": [hit("cobblestone", "cobblestone", 4, 70, 281, 8.0)],
        }
        self.assertEqual(nearest_ground(gs).pos, (4, 70, 281))

    def test_a_foothold_the_pathfinder_timed_out_on_is_passed_over(self) -> None:
        # `Agent._note_rescue` records the cell a stopped swim aimed at; the next tick takes the
        # other foothold rather than timing out on the same one (live run115, 281 identical walks).
        gs = GameState()
        gs.pos = (-0.5, 62.4, 281.5)
        gs.rescue_ground = {"stone": [
            hit("stone", "stone", -1, 62, 281, 0.5),
            hit("stone", "stone", 6, 62, 281, 6.5),
        ]}
        gs.rescue_skip = frozenset({(-1, 63, 281)})
        self.assertEqual(nearest_ground(gs).pos, (6, 62, 281))

    def test_ignores_ground_under_a_cave_ceiling(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 63.0, 0.0)
        gs.resources = {"dirt": [hit("dirt", "dirt", 2, 58, 0, 2.5, above="cave_air")]}
        self.assertIsNone(nearest_ground(gs))

    def test_none_without_ground(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 63.0, 0.0)
        self.assertIsNone(nearest_ground(gs))


class ExploreBindingTest(unittest.TestCase):
    def _gs(self) -> GameState:
        gs = GameState()
        gs.pos = (10.5, 64.0, -3.2)
        return gs

    def test_uses_target_when_present(self) -> None:
        target = Target(label="stone@40,10", text="stone 30m east",
                        hit=hit("stone", "stone", 40, 60, 10, 30.0))
        action = resolve("explore", self._gs(), target)
        self.assertIsNotNone(action)
        self.assertEqual(action.tool, "navigate")
        self.assertEqual(action.args["x"], 40)
        self.assertEqual(action.args["y"], 60)
        self.assertEqual(action.args["z"], 10)
        self.assertTrue(action.args["dig"])

    def test_without_an_outward_target_extends_the_frontier(self) -> None:
        # No target at all (no ground in reach) still means "leave": push north (no base).
        action = resolve("explore", self._gs())
        self.assertIsNotNone(action)
        self.assertEqual(action.tool, "explore_frontier")
        self.assertEqual(action.args["heading"], 0.0)
        self.assertEqual(action.args["chunks"], 2)

    def test_a_near_target_extends_the_frontier_instead_of_ping_ponging(self) -> None:
        # A target at the bot's feet is the boxed-in case; 211 of 298 live ticks were exactly
        # this navigate ping-pong while the portal chain waited on lava.
        target = Target(label="grass_block@11,-3", text="grass_block 1m east, open above",
                        hit=hit("grass_block", "grass_block", 11, 64, -3, 0.8))
        action = resolve("explore", self._gs(), target)
        self.assertIsNotNone(action)
        self.assertEqual(action.tool, "explore_frontier")

    def test_no_position_no_action(self) -> None:
        gs = GameState()
        self.assertIsNone(resolve("explore", gs))


class FlatBuildSiteTest(unittest.TestCase):
    """The flat base-site pick: the portal frame only fits where its whole bottom row stands on
    level ground with clear air above. The old pick took the *lowest* standable cell, which in a
    dip left 15 of 20 frame cells holding stone the build server will not dig."""

    @staticmethod
    def _grid(bx_range, bz_range, ground: int = 63, top: int = 72) -> dict:
        grid = {}
        for x in bx_range:
            for z in bz_range:
                for y in range(ground + 1, top):
                    grid[(x, y, z)] = "air"
                grid[(x, ground, z)] = "grass_block"
                grid[(x, ground - 1, z)] = "dirt"  # the subsurface the bottom row sits on
        return grid

    def test_picks_level_ground_for_the_frame(self) -> None:
        grid = self._grid(range(0, 6), range(-2, 3))
        site = flat_build_site(grid, (2.5, 64.0, 0.5))
        self.assertEqual(site, (0, 63, 0))
        bx, by, bz = site
        self.assertTrue(all(grid[(bx + i, 63, bz)] == "grass_block" for i in range(5)))
        self.assertTrue(all(grid[(bx + i, 63 + dy, bz)] == "air"
                            for i in range(5) for dy in range(1, 5)))

    def test_rejects_a_slope_that_leaves_frame_cells_solid(self) -> None:
        # Every candidate row runs east into a step up: those frame cells hold stone.
        grid = self._grid(range(0, 6), range(-2, 3))
        for x in (4, 5):
            for z in range(-2, 3):
                grid[(x, 64, z)] = "stone"
        self.assertIsNone(flat_build_site(grid, (2.5, 64.0, 0.5)))

    def test_a_plant_in_a_frame_cell_is_replaceable(self) -> None:
        grid = self._grid(range(0, 6), range(-2, 3))
        grid[(2, 65, 0)] = "short_grass"  # not a blocker: placement replaces it
        self.assertIsNotNone(flat_build_site(grid, (2.5, 64.0, 0.5)))

    def test_rejects_a_cave_floor_under_enclosed_air(self) -> None:
        # `cave_air` is the enclosed variant: a level floor with room above it inside a cave is
        # not "open surface ground" and must not become the home base (live run8: the base
        # anchored at y=-9 while the bot was still 70 m underground, and every later base step --
        # the water rescue included -- was then aimed straight down).
        grid = self._grid(range(0, 6), range(-2, 3))
        for key, name in list(grid.items()):
            if name == "air":
                grid[key] = "cave_air"
        self.assertIsNone(flat_build_site(grid, (2.5, 64.0, 0.5)))

    def test_rejects_a_hole_under_the_bottom_row(self) -> None:
        # The server digs the bottom row out and re-places it as obsidian, so a cell with a hole
        # beneath it has nothing to place against and is left wrong; the opening then never
        # closes and the portal cannot light (live, Nether run).
        grid = self._grid(range(0, 6), range(0, 1))
        grid[(2, 62, 0)] = "air"
        self.assertIsNone(flat_build_site(grid, (2.5, 64.0, 0.5)))

    def test_rejects_a_pit_floor_below_the_bots_own_footing(self) -> None:
        # Live run26: the two scan reads reach three blocks below the bot, and a dug-out dip
        # floor met every other criterion, so the base sank to it -- the bot stood at y=64, the
        # anchor landed at y=61, and the frame, chest and furnace were laid into the hillside
        # 1-2 blocks under the surrounding ground. A row below the bot's own footing (fy - 1) is
        # a pit floor, not the surface the base may stand on; its own level and the level ground
        # overhead stay admissible.
        for ground, expected in ((61, None), (62, None), (63, (0, 63, 0))):
            grid = self._grid(range(0, 6), range(-2, 3), ground=ground)
            self.assertEqual(flat_build_site(grid, (2.5, 64.0, 0.5)), expected)

    def test_no_grid_no_site(self) -> None:
        self.assertIsNone(flat_build_site({}, (0.0, 64.0, 0.0)))


class BedFrameSiteTest(unittest.TestCase):
    """The site a standing bed anchors is checked before it becomes the home base.

    `_adopt_homebase` re-anchors the base on a bed a restart left (or an early `sleep` laid), and
    that path skips `establish_base`'s `flat_build_site` clearance. A bed on a cave floor then pins
    the base where the frame cannot be raised (live, run17: the base sat nine blocks under the
    surface and the frame's top and west cells stayed stone, so `light_portal` answered
    `[PORTAL_NO_FRAME]` on 150 ticks).
    """

    @staticmethod
    def _grid(bx_range, bz_range, ground: int = 63, top: int = 72) -> dict:
        grid = {}
        for x in bx_range:
            for z in bz_range:
                for y in range(ground + 1, top):
                    grid[(x, y, z)] = "air"
                grid[(x, ground, z)] = "grass_block"
                grid[(x, ground - 1, z)] = "dirt"
        return grid

    @staticmethod
    def _bed(x: int, y: int, z: int) -> BlockHit:
        return BlockHit(query="white_bed", name="white_bed", x=x, y=y, z=z, distance=1.0)

    def test_a_clear_surface_site_is_accepted_with_the_bed_on_the_anchor(self) -> None:
        # base anchor = the bed's ground cell; the frame stands east of the bed, so the bed on the
        # anchor column must not disqualify it.
        grid = self._grid(range(0, 9), range(-2, 3))
        self.assertTrue(bed_frame_site(grid, self._bed(0, 64, 0)))

    def test_a_bed_underground_is_refused(self) -> None:
        # The run17 shape: the ground is fine, but a stone roof sits right above the bed, so the
        # frame's own cells would hold stone -- `cave_air`/stone overhead is not open sky.
        grid = self._grid(range(0, 9), range(-2, 3))
        for x in range(0, 9):
            for z in range(-2, 3):
                for y in range(64, 72):
                    grid[(x, y, z)] = "stone"
        self.assertFalse(bed_frame_site(grid, self._bed(0, 64, 0)))

    def test_a_slope_beside_the_bed_is_refused(self) -> None:
        grid = self._grid(range(0, 9), range(-2, 3))
        grid[(4, 64, 0)] = "stone"  # a frame bottom-row cell the build server will not dig
        self.assertFalse(bed_frame_site(grid, self._bed(0, 64, 0)))

    def test_a_hole_under_the_frame_row_is_refused(self) -> None:
        grid = self._grid(range(0, 9), range(-2, 3))
        grid[(2, 62, 0)] = "air"  # nothing for the bottom-row obsidian to sit on
        self.assertFalse(bed_frame_site(grid, self._bed(0, 64, 0)))

    def test_a_bed_reaching_into_the_frame_interior_is_refused(self) -> None:
        # A bed laid north-south puts its head in the frame's own column; that cell is not open sky.
        grid = self._grid(range(0, 9), range(-2, 3))
        grid[(1, 64, 0)] = "white_bed"
        self.assertFalse(bed_frame_site(grid, self._bed(0, 64, 0)))

    def test_no_grid_is_refused(self) -> None:
        self.assertFalse(bed_frame_site({}, self._bed(0, 64, 0)))


class StepAwayCellTest(unittest.TestCase):
    """The walk off an open drop aims at a cell the ground read proved standable.

    run86 dropped junk at (-90,70,-5) and walked "three blocks away" from a drop under its own
    feet - a zero vector, so always due west - to (-93,70,-6), a cell with no footing: nine
    `[NAVIGATION_TARGET_UNSUPPORTED]` in three rounds of drop and refused walk.
    """

    DROP = (-89.5, 70.0, -4.5)

    def _ground(self, floor_y: int = 69, hole_west_of: int | None = None) -> dict:
        # One read like `scan_ground`'s box: stone at `floor_y`, air above it; columns west of
        # `hole_west_of` have no floor at all (the run86 west side).
        grid = {}
        for x in range(-94, -85):
            for z in range(-9, 0):
                for y in range(67, 74):
                    hole = hole_west_of is not None and x < hole_west_of
                    grid[(x, y, z)] = "stone" if (y <= floor_y and not hole) else "air"
        return grid

    def _away(self, cell) -> float:
        return math.hypot(cell[0] + 0.5 - self.DROP[0], cell[2] + 0.5 - self.DROP[2])

    def test_the_cell_stands_on_the_floor_far_enough_from_the_drop(self) -> None:
        grid = self._ground()
        cell = step_away_cell(grid, self.DROP, self.DROP)
        self.assertIsNotNone(cell)
        self.assertEqual(cell[1], 70)
        self.assertEqual(grid[(cell[0], cell[1] - 1, cell[2])], "stone")
        self.assertGreaterEqual(self._away(cell), STEP_AWAY_MIN)

    def test_ground_without_footing_is_never_the_target(self) -> None:
        cell = step_away_cell(self._ground(hole_west_of=-90), self.DROP, self.DROP)
        self.assertIsNotNone(cell)
        self.assertGreaterEqual(cell[0], -90)

    def test_a_refused_cell_gives_way_to_the_next(self) -> None:
        grid = self._ground()
        first = step_away_cell(grid, self.DROP, self.DROP)
        second = step_away_cell(grid, self.DROP, self.DROP, frozenset({first}))
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_lava_and_magma_are_no_footing(self) -> None:
        grid = {(x, y, z): name for (x, y, z), name in self._ground().items()}
        for (x, y, z) in list(grid):
            if y == 69:
                grid[(x, y, z)] = "magma_block" if x % 2 else "lava"
        self.assertIsNone(step_away_cell(grid, self.DROP, self.DROP))

    def test_no_read_no_cell(self) -> None:
        self.assertIsNone(step_away_cell({}, self.DROP, self.DROP))

    def _tunnel(self) -> dict:
        # A one-wide tunnel running east-west at z=-5, stone everywhere else (run87 at y=6).
        grid = {}
        for x in range(-96, -82):
            for z in range(-7, -2):
                for y in range(68, 73):
                    grid[(x, y, z)] = "air" if (z == -5 and y in (70, 71)) else "stone"
        return grid

    def test_the_walk_goes_back_the_way_the_bot_came(self) -> None:
        # run87 it171-176: dug west, tossed west, and the even tie between the cells east and
        # west went west through the pile - the junk came straight back up, four rounds running.
        # With the heading the cell behind the bot wins.
        grid = self._tunnel()
        self.assertLess(step_away_cell(grid, self.DROP, self.DROP)[0], self.DROP[0])
        cell = step_away_cell(grid, self.DROP, self.DROP, heading=(-1.0, 0.0))
        self.assertGreater(cell[0], self.DROP[0])
        self.assertGreaterEqual(self._away(cell), STEP_AWAY_MIN)
        cell = step_away_cell(grid, self.DROP, self.DROP, heading=(1.0, 0.0))
        self.assertLess(cell[0], self.DROP[0])

    def test_a_dead_end_behind_still_steps_ahead(self) -> None:
        # Nothing standable behind the bot: a cell ahead beats no walk at all.
        grid = self._tunnel()
        for (x, y, z) in list(grid):
            if x > -90:
                grid[(x, y, z)] = "stone"
        cell = step_away_cell(grid, self.DROP, self.DROP, heading=(-1.0, 0.0))
        self.assertIsNotNone(cell)
        self.assertLess(cell[0], self.DROP[0])


if __name__ == "__main__":
    unittest.main()
