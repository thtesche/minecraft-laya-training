from __future__ import annotations

import unittest

from mlaya.perception import (GROUND_NAMES, RESCUE_GROUND_NAMES, RESCUE_RADIUS, BlockHit,
                              parse_block_above, parse_box, parse_cells, parse_ground_height,
                              parse_portal_corner, parse_resource_counts, parse_resources,
                              parse_status, parse_surface_height, portal_interior)


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
        self.assertIn("off-hand: shield", gs.narrative())

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
        self.assertIn("zombie x1 at 13m", gs.narrative())

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
