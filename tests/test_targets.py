"""M4 parameter layer: target candidates from `view_blocks` hits and their binding."""
from __future__ import annotations

import unittest

from mlaya.actions import resolve
from mlaya.perception import BlockHit, GameState
from mlaya.targets import (Target, _compass, base_target, boxed_in, flat_build_site,
                           frontier_heading, ground_targets)


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

    def test_none_without_ground(self) -> None:
        gs = GameState()
        gs.pos = (0.0, 64.0, 0.0)
        gs.resources = {"stone": [hit("stone", "stone", 2, 65, 0, 2.0)]}
        self.assertIsNone(base_target(gs))


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

    def test_rejects_a_hole_under_the_bottom_row(self) -> None:
        # The server digs the bottom row out and re-places it as obsidian, so a cell with a hole
        # beneath it has nothing to place against and is left wrong; the opening then never
        # closes and the portal cannot light (live, Nether run).
        grid = self._grid(range(0, 6), range(0, 1))
        grid[(2, 62, 0)] = "air"
        self.assertIsNone(flat_build_site(grid, (2.5, 64.0, 0.5)))

    def test_no_grid_no_site(self) -> None:
        self.assertIsNone(flat_build_site({}, (0.0, 64.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
