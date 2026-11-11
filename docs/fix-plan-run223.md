# Fix-Plan — Run 223 (Head `c8fa95e`)

Vier Änderungen, alle durch Live-Daten aus `runs/decisions.jsonl` und `runs/system2.jsonl`
belegt. Jede hat einen Test, der **vor** der Änderung fällt.

Reihenfolge: 1 und 2 sind reine Filter in `predicates.py` / `graph.py` (autonom nach Laufende),
3 berührt die Placement-Policy (ebenfalls Filter, kein Goal), 4 ist ein Resolve-Argument im
Planner (derselbe dig/scaffold-Muster wie überall sonst), 5 ist eine Goal-Definition und
braucht dein GO.

---

## 1. Eskalation: System 2 darf nicht das Stuck-Label selbst zurückgeben

**Befund (it 310, `runs/system2.jsonl`):**

| it | Trigger | Stuck | Plan (System 2) | Gewählt | Ergebnis |
|----|---------|-------|-----------------|---------|----------|
| 88 | unresolved_label | sleep | establish_base → get_wood → … | get_wood | ✓ |
| 89 | unresolved_label | sleep | establish_base → craft_chest → … | smelt_iron | ✓ |
| 309 | stall_loop | collect_drops | get_wood → hunt_string → craft_bow → craft_bed → place_bed | get_wood | ✓ |
| 310 | stall_loop | collect_drops | **collect_drops** → take_stored_food → hunt_string → … | collect_drops | ✗ verworfen |

Der Plan von it 310 war korrekt. Er beginnt aber mit `collect_drops` — genau dem Label, das
der Stall-Guard gerade ausgeschlossen hat.

**Ursache.** `Agent._escalate` (`src/mlaya/graph.py:1844-1848`) nimmt die **erste** Planstufe,
die `resolve()` bauen kann:

```python
for label in steps:
    built = resolve(label, gs)
    if built is not None:
        chosen, action = label, built
        break
```

`collect_drops` ist in diesem State noch resolvabel (siehe Punkt 2), also wird es als `chosen`
registriert. Der Guard in `graph.py:1791` (`chosen != decision.action_label`) verwirft die
Alternative — der Tick läuft trotzdem `collect_drops` und antwortet
`[NO_CONFIRMED_PICKUP]`. Der Budget-Zähler `_stall_escalations` war in
`graph.py:1788` **vor** dem Guard erhöht worden: ein LLM-Call, ein Tick, kein Fortschritt,
und 2 von 3 Eskalationen verbraucht.

**Änderung.**

```python
# graph.py:1821 — Signatur
def _escalate(self, gs, trigger: str, unresolved_label: str | None,
              iteration: int, exclude: frozenset[str] = frozenset()) -> tuple[Any, str | None, dict | None]:
```

```python
# graph.py:1844 — die Planstufe, die den Stall selbst meint, ist kein Ausweg
for label in steps:
    if label in exclude:
        continue
    built = resolve(label, gs)
    ...
```

und am Aufruf der Stall-Guard-Zweig:

```python
# graph.py:1789
alt, chosen, _ = self._escalate(
    gs, "stall_loop", decision.action_label, state.get("iteration", 0),
    exclude=frozenset({decision.action_label}) | set(gs.stuck_labels))
```

`unresolved_label`-Eskalationen (graph.py:1798) brauchen kein `exclude`: dort *fehlt* die
Action, sie ist nicht resolvabel, und ein Echo wäre schon durch `resolve() == None` gefiltert.

**Test** (`tests/test_graph.py`, Klasse `StallEscalationTest`, Stil wie
`test_the_same_label_back_from_the_model_runs_as_before`, Zeile 517):

```python
async def test_a_plan_starting_with_the_stuck_label_skips_to_the_next_step(self) -> None:
    agent = self._agent()
    agent._stall_dropped = {"collect_drops": STALL_ESCALATE_REPEATS}
    stuck = Action("collect_drops", "pick_up_items", {"radius": 16}, "sweep")
    food = Action("take_stored_food", "take_from_chest", {}, "grab the banked food")
    with mock.patch.object(graph, "candidates", return_value=["collect_drops", "take_stored_food"]), \
         mock.patch.object(graph, "resolve",
                           side_effect=lambda lbl, g, t=None: stuck if lbl == "collect_drops"
                           else food if lbl == "take_stored_food" else None), \
         mock.patch.object(graph, "escalate_to_system2",
                           return_value={"plan": ["collect_drops", "take_stored_food"]}):
        out = await agent._guard(self._state("collect_drops"))
    self.assertIs(out["action"], food)          # heute: out["action"] is stuck
    self.assertEqual(out["note"], "stall -> take_stored_food")
```

**Risiko:** keiner. Im schlechtesten Fall (Plan besteht nur aus dem Stuck-Label) passiert
exakt das, was heute passiert.

---

## 2. `collect_drops`: Offer ohne Reach-Filter

**Befund.** 22 Angebote in Run 223, 10 davon refusiert. Der Zusammenhang ist eindeutig:

| it | `gs.drops` | nächste geladene Drop | Distanz | Antwort |
|----|-----------|----------------------|---------|---------|
| 39 | 2 | 2.1 | ✓ | succeeded |
| 48 | 1 | 3.4 | ✓ | succeeded |
| 154 | 1 | 6.5 | ✓ | succeeded |
| 349 | 1 | 3.2 | ✓ | succeeded |
| 46 | 1 | 59.7 | ✗ | `[NO_MATCHING_ITEMS]` |
| 84 | 1 | 43.0 | ✗ | `[NO_MATCHING_ITEMS]` |
| 158 | 1 | 34.5 | ✗ | `[NO_MATCHING_ITEMS]` |
| 166 | 1 | 59.4 | ✗ | `[NO_MATCHING_ITEMS]` |
| 198 | 4 | 107.0 | ✗ | `[NO_MATCHING_ITEMS]` |
| 201 | 1 | 39.7 | ✗ | `[NO_MATCHING_ITEMS]` |
| 224 | 1 | — (Liste leer) | ✗ | `[NO_MATCHING_ITEMS]` |
| 228 | 1 | 77.5 | ✗ | `[NO_MATCHING_ITEMS]` |
| 346 | 6 | 60.9 | ✗ | `[NO_MATCHING_ITEMS]` |

Alle vier Erfolge: Drop ≤ 6,5 Blöcke. Alle Refusals: ≥ 34 Blöcke oder keine geladene Drop
überhaupt.

**Ursache.** `predicates.py:2320`:

```python
def _drops_to_collect(gs: GameState) -> bool:
    return _supply_drop(gs) is not None or (gs.drops > 0 and gs.slots_used < INVENTORY_SLOTS)
```

`gs.drops` zählt **Kills der letzten Ticks**, nicht erreichbare Items (`perception.py:980-983`).
`_supply_drop` filtert korrekt nach `DROP_SWEEP_RADIUS` (32), der Kill-Zweig filtert gar nicht.
`planner.py:1926` sweeped dann `radius: 16` um die Bot-Position — der Server findet nichts und
antwortet `[NO_MATCHING_ITEMS]`.

Der Stall-Guard (`_track_stuck`, `graph.py:2371`) greift erst nach `STUCK_REPEATS = 4`
identischen Refusals und withholdet dann 15 Ticks — das pacing, nicht die Ursache. In Run 223
hat genau ein Zyklus stattgefunden, dann kam die Eskalation (Punkt 1), die denselben Tick
wieder an `collect_drops` gab.

**Änderung.**

```python
# actions/constants.py, neben DROP_SWEEP_RADIUS
DROP_REACH = DROP_SWEEP_RADIUS   # 32.0: der Sweep findet nichts, was weiter weg ist
```

```python
# predicates.py:2320
def _nearest_loaded_drop(gs: GameState) -> float | None:
    """Distance to the closest loaded dropped item, or None when nothing is loaded."""
    dists = [d for _, _, d, _ in gs.dropped_items if d is not None]
    return min(dists) if dists else None


def _drops_to_collect(gs: GameState) -> bool:
    """Whether `collect_drops` has anything the carried pack can actually take.

    A fresh kill's drops are not named, so that sweep needs a free slot *and* a loaded item the
    server can see: `gs.drops` counts kills, not reach. Live, run223: 10 sweeps on drops 34-107
    blocks away answered `[NO_MATCHING_ITEMS]`, every sweep that ran on a drop within 7 blocks
    succeeded.
    """
    if _supply_drop(gs) is not None:
        return True
    if gs.slots_used >= INVENTORY_SLOTS:
        return False
    near = _nearest_loaded_drop(gs)
    return gs.drops > 0 and (near is None or near <= DROP_REACH)
```

`near is None` bleibt erlaubt: der Server hat Kills gemeldet, die Item-Entity ist nur noch nicht
im Scan — ein Tick Sweep ist dort vertretbar, ein Tick auf eine 107-Block-Drop nicht.

**Test** (`tests/test_actions.py`, `gs()`-Fixture, Zeile 74):

```python
def test_collect_drops_is_not_offered_for_a_drop_out_of_sweep_reach(self) -> None:
    g = gs(drops=1, slots_used=25, pos=(0.0, 70.0, 0.0),
           dropped_items=[["armadillo_scute", 1, 59.7, (-59.0, 70.0, 0.0)]])
    self.assertFalse(_drops_to_collect(g))
    self.assertIsNone(resolve("collect_drops", g))
    self.assertNotIn("collect_drops", candidates(g))

def test_collect_drops_still_offered_when_the_drop_is_underfoot(self) -> None:
    g = gs(drops=1, slots_used=25, pos=(0.0, 70.0, 0.0),
           dropped_items=[["string", 2, 6.2, (6.0, 70.0, 0.0)]])
    self.assertTrue(_drops_to_collect(g))
```

**Nebeneffekt:** `hunt_blaze`-Kills im Nether bleiben erhalten — die Drops liegen dort direkt
neben dem Bot (it 349: 3.2 Blöcke, succeeded).

---

## 3. Truhe: die Picker kennen blattbedeckte Zellen nicht

**Befund (it 304-331):**

| it | Label | Status | Text |
|----|-------|--------|------|
| 304 | craft_chest | ✓ | 8 Planken → Truhe im Pack |
| 305 | place_chest | ✗ | `[PLACE_TARGET_OCCUPIED] The target cell contains acacia_leaves, which this action will not remove.` |
| 306 | make_room | ✓ | gräbt genau diese Zelle |
| 331 | place_chest | ✓ | gesetzt |

Die Recovery funktioniert — aber 25 Ticks später, weil in der Zeit `craft_furnace`,
`craft_bowl`, `mine_gold`, `smelt_gold`, `craft_gold_helmet` … gewonnen haben.

**Ursache.** `_chest_candidates` (`predicates.py:3141-3167`) filtert Zellen, die schon eine
Truhe, ein Bett, Obsidian, einen Ofen oder ein `placement_refused`-Flag tragen — aber nicht
eine Zelle, die der Server gerade als **okupiert, aber diggbar** gemeldet hat
(`gs.blocked_cell`, gesetzt in `graph.py:3378`). `place_chest` (`planner.py:2068-2085`)
nimmt `_chest_site(gs)` blind. `make_room` (`planner.py:1933-1943`) liest `blocked_cell`
richtig — die Reihenfolge "erst diggen, dann setzen" ist aber nicht erzwungen.

`UNDIGGABLE_BLOCKS = {"white_bed", "crafting_table"}` (`constants.py:521`) ist korrekt:
Laub *kann* `make_room` entfernen (it 306), nur `place_block` darf es nicht.

**Änderung — drei Teile.**

a) **Zelle im Picker markieren, nicht vergessen.** In `_chest_candidates`:

```python
    taken |= set(gs.placement_refused)
    if gs.blocked_cell is not None:
        # The server refused this exact cell and named the block in it: it can be dug
        # (`make_room`), but not until that dig has run. Naming it again before the dig is
        # one refusal per tick.
        taken.add(tuple(gs.blocked_cell))
```

b) **Der Dig steht vor dem Setzen.** In `candidates()` direkt vor dem `place_chest`-Zweig
(planner.py:2068): wenn `gs.blocked_cell` gesetzt und nicht in `placement_refused` ist,
`make_room` anbieten, bevor `place_chest` wieder auf derselben Zelle landet.

```python
if gs.blocked_cell is not None and gs.blocked_cell not in gs.placement_refused:
    out.append("make_room")
    return out
```

c) **`place_chest` resolvet nicht auf eine bekannte blockierte Zelle** — Sicherheitsnetz,
falls (a) und (b) in einem anderen Zweig umgangen werden:

```python
cell = _chest_site(gs)
if cell is None or (gs.blocked_cell is not None and cell == tuple(gs.blocked_cell)):
    return None
```

**Test** (`tests/test_actions.py`, Stil wie `with_chest`, Zeile 112):

```python
def test_a_cell_the_placement_refused_is_not_named_again_before_the_dig(self) -> None:
    g = at_home(gs(inventory={"chest": 1, "crafting_table": 1, "oak_planks": 12}), base=(30, 90, -325))
    g.chest_pos = (30, 91, -324)
    g.chests_full = [(30, 91, -324)]
    g.blocked_cell = (30, 92, -324)          # acacia_leaves, diggbar
    self.assertNotIn((30, 92, -324), _chest_candidates(g))
    self.assertIsNone(resolve("place_chest", g))
    act = resolve("make_room", g)
    self.assertIsNotNone(act)
    self.assertEqual(act.args[0], {"x": 30, "y": 92, "z": -324, "block_name": "air"})
```

Nach dem erfolgreichen Dig löscht `_note_place` `blocked_cell` (graph.py:3395), die Zelle
kehrt in den Picker zurück und `place_chest` landet — genau der Pfad, der in Run 223
funktioniert hat, nur ohne die 25 Tick Wartezeit.

---

## 4. `travel_to_stronghold`: die Navigation kennt keinen Dig

**Befund (it 461–599):** `travel_to_stronghold` 21× failed, 14× angeboten ohne Ergebnis.
`NAVIGATION_STOPPED` ×33 ist die Top-Refusal-Kategorie des Laufs, davon 21 Timeouts
(`search timed out after 2000 ms`) auf diesem Label. Die Frame-Zelle
`(-269, -36, -1819)` liegt **98 Blöcke unter der Oberfläche** (locate bestätigt sie, während
der Bot 108 m entfernt und 98 Blöcke darüber steht — die Docstring in `_frame_distance`
beschreibt genau diesen Fall, run173 it75). Der Bot pendelt zwischen `(-268.5, 62, -1818.5)`
(direkt über dem Rahmen) und `(-299.3, 63, -1727.5)`, driftet dabei bis 350 Blöcke weg,
`escape_water` wird zum Hauptlabel (26 Angebote), und `end_portal_active` wird nie erreicht —
die Kette `craft_bed_stronghold → place_bed_stronghold → sleep_stronghold → activate_end_portal`
bleibt geschlossen, weil `_at_stronghold` (range 3) nie true wird.

**Ursache.** `stronghold.py:583`:

```python
return Action(label, "navigate", {"x": x, "y": y, "z": z,
                                 "range": STRONGHOLD_ARRIVAL_RANGE}, why)
```

Ohne `dig`/`scaffold` kann der Pathfinder die 98 Blöcke Solid nicht durchqueren und timed out.
`place_bed_stronghold` (`stronghold.py:613-615`) navigiert zur selben Region bereits mit
`"dig": True, "scaffold": True` — das Muster ist im Planner Standard (20 Stellen,
planner.py:980 ff., end.py:184).

**Änderung.**

```python
return Action(label, "navigate", {"x": x, "y": y, "z": z,
                                 "range": STRONGHOLD_ARRIVAL_RANGE,
                                 "dig": True, "scaffold": True}, why)
```

**Test** (Stronghold-Tests in `tests/test_actions.py`):

```python
def test_travel_to_the_frame_digs_when_the_cell_is_underground(self) -> None:
    g = gs(eyes=12, pos=(-268.5, 62.0, -1818.5), stronghold_frame=(-269, -36, -1819),
           stronghold_estimated=True, end_portal_active=False)
    act = resolve("travel_to_stronghold", g)
    self.assertIsNotNone(act)
    self.assertEqual(act.tool, "navigate")
    self.assertTrue(act.args["dig"] and act.args["scaffold"])   # heute: fehlen
```

**Risiko:** niedrig. Der Dig ist genau die Bewegung, die der Bett-Zweig an derselben Stelle
bereits erfolgreich nutzt; ohne sie findet der Lauf das Stronghold nie.

---

## 5. (braucht GO) Bett vor dem Crossing

`place_bed` hängt an `_portal_lit` (planner.py:569), `enter_portal` gewinnt die Tick-Entscheidung
it 219 — das Bett wurde nie gelegt, der Respawn nach dem Lava-Tod (it 250,
`(-187.48, 23.0, 114.45)`) landete 85 Blöcke vom Base. `_nether_departure_ready`
(`predicates.py:2027`) verlangt `_bed_at_base` nicht, analog zu `8298a43`, das
`_base_storage_ready` für die Truhe ergänzt hat.

Das ist eine Goal-Definition, kein Filter — ich lasse sie, bis du sagst, ob das Bett
**vor** dem Crossing Pflicht sein soll oder bewusst danach.

---

## Validierung nach dem Lauf

| Metrik | Run 223 | Ziel |
|--------|---------|------|
| `collect_drops`-Refusals | 10 | ≤ 2 |
| Eskalationen mit `chosen == unresolved` | 1 von 4 | 0 |
| Eskalations-Budget verbraucht ohne Wirkung | 1 von 3 | 0 |
| `place_chest` nach `[PLACE_TARGET_OCCUPIED]` bis zum Erfolg | 26 Ticks | ≤ 2 |
| `travel_to_stronghold`-Timeouts | 21 | ≤ 3 |
| `end_portal_active` | nie erreicht | erreicht |
| `still_ratio` | 0.062 | ≤ 0.06 |
