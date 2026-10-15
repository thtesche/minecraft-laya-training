# Architektur — minecraft-laya-training

Ein Minecraft-Agent, dessen **System-1-Entscheidungen von [Laya](https://github.com/NandhaKishorM/laya)**
kommen (nicht-autoregressive Typed-Decision-Engine, `choice`/`score`/`noul`, ein Forward-Pass).
Laya erzeugt **keinen** Text und **keine** Tool-Argumente — daher wird „spielen" in
**Entscheidungen mit bereits gültigen Optionen** zerlegt, und ein separater **Executor** spricht das
mine-ai-MCP.

## Entscheidungen (fixiert)

| Achse | Wahl |
|---|---|
| Sprache | Python ≥3.10, virtuelle Umgebung via `uv` |
| System 1 | Laya **in-process** (`laya.Router`), **zero-shot**, Checkpoint `english` |
| System 2 | OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` (`max_tokens=16384`) |
| Orchestrierung | **LangGraph** (async, `MemorySaver`) |
| MCP | Streamable HTTP `http://localhost:25575/mcp` |
| Wissen | zero-shot, in den `criteria`-Texten; keine Wissensdatei |

### Warum der `english`-Checkpoint

Der State und alle Fragen sind Englisch. Im direkten Vergleich liest `english` den State klar
besser als `multilingual` (z. B. binäres Gate: `SAFE→safe 0.96`, `DANGER→in_danger 0.92` gegen
`0.89/0.76`). Dict-State (JSON) fiel deutlich ab; **kompakter String-State** gewinnt.

## Zero-Shot-Befund (wichtig)

Systematische Tests vor dem Bau ergaben:

- **Mehrklassen-`choice` ist zero-shot unzuverlässig** (First-Option-Bias): ein 3-Wege-`macro`
  (`emergency`/`survival`/`progression`) wählte im sicheren Zustand `emergency`; auch 5-Wege-
  Progression war verrauscht. Auch `noul` ist schlecht kalibriert (langer State driftet nach oben).
- **Binäre `choice` funktioniert**: „Ist der Bot in Gefahr?" traf `SAFE→no`, `DANGER→yes`.
- **Langer State wird von Laya aufs erste Fenster gekürzt** → State muss kurz bleiben.

Konsequenz: Der Code **filtert die Optionen anhand der Vorbedingungen** (Tech-Tree), und Laya
entscheidet nur unter diesen gültigen Optionen. Dadurch gibt es keine Sackgassen, und jede
Entscheidung ist eine echte Wahl unter realisierbaren Aktionen. Das ist genau der Pfad, den
später das Kopf-Fine-Tuning (M3) verbessert.

## Schichten

```
[1] MCPExecutor   (src/mlaya/mcp.py)
    Streamable-HTTP-Client + Gating-State-Machine:
    submission_id-Fabrik, rationale-Guard (<=200), Action-ID-Parsing,
    wait_for_action (initial_wait_ms, dann poll_ms x max_polls),
    ACTION_BUSY / RESULT_NOT_RETRIEVED-Recovery: die blockierende Action-ID wird aus den
    strukturierten Daten gelesen (`unretrievedActionId` / `activeActionId`; der Text nennt nur
    den Parameternamen), ihr Ergebnis abgeholt und die eigene Einreichung wiederholt - die
    Ablehnung heißt, dass die eigene Aktion nie startete. Reads sind ungated.
[2] Perception    (src/mlaya/perception.py)
    view_status + read_recent_events + view_blocks (alle frei) -> GameState
    -> kompakter NL-State (`narrative()`); Ressourcen (logs/stone/iron_ore/coal_ore/
    diamond_ore) mit Distanz im Umkreis von 64 Blöcken (Ressourcen- und Boden-Scan sind
    zwei `view_blocks`-Gruppen, weil `find` maximal 8 Blocknamen erlaubt); Tageszeit/
    Schlafzustand, Mob-Arten (inkl. passiver) und ungelesene Notifications (Drain)
    fließen in den State.
[3] Brain (S1)    (src/mlaya/brain.py)
    Laya-Router, EIN predict()-Pass:
      in_danger  (binäres Gate)      -> defer an die Survival-Policy; `danger_matters` begrenzt ihn
                                       (Hostile ≤ 4 m Kontakt oder Health < 15, sonst arbeitet der Loop weiter)
      danger     (score, nur Logging)
      intent     (choice)            -> unter den gefilterten Kandidaten
[4] Actions       (src/mlaya/actions.py)
    candidates(gs): gültige Labels nach Vorbedingungen (Tech-Tree-Reihenfolge)
    resolve(label, gs): Label -> echtes MCP-Tool + Argumente
    Kaskade: get_wood -> craft_pickaxe -> craft_table -> mine_stone ->
    craft_stone_pickaxe -> craft_furnace -> mine_iron -> mine_coal ->
    smelt_iron -> craft_iron_pickaxe -> craft_shield -> equip_shield ->
    craft_iron_sword -> craft_iron_<slot> -> equip_armor (Eisen-Kampfkit, VOR dem
    Diamantgang) -> mine_diamond -> craft_diamond_pickaxe
    (-> craft_diamond_sword) -> craft_diamond_<slot> (Brustplatte/Hose/Schuhe/Helm)
    -> equip_armor; Bett-Kette (hunt_wool/hunt_string ->
    craft_bed -> sleep in der Nacht) und eat_food bei Hunger; Homebase
    (surface_up -> establish_base -> go_home, craft_chest/place_chest/stow_surplus) und
    Nether-Kette (craft_bucket -> fill_bucket -> mine_obsidian -> get_flint ->
    craft_flint_and_steel -> clear_portal -> build_portal -> light_portal -> place_bed); explore
    nur als letzter Fallback (wenn nichts Produktives möglich ist).
[5] Verify        (src/mlaya/verify.py)
    view_status erneut lesen; Soll/Ist-Delta (Inventar, Position, Health, Hunger,
    Phase/Schlaf); Fortschritt je Aktion
[6] Memory        (src/mlaya/memory.py)
    runs/decisions.jsonl (jede Entscheidung + Outcome)
[7] System 2      (brain.escalate_to_system2)
    struktureller Trigger (Label nicht auflösbar) -> OpenRouter wählt eine gültige Alternative
[8] Survival      (src/mlaya/survival.py)
    setzt die Server-Survival-Policy in zwei Stufen (`base`/`hunt`) beim Start und erneut
    nach jedem Tod; `read_policy` liest Revision/`effective`/`overrides`
```

## LangGraph

```
START -> perceive -> decide -> guard --+-> execute -> verify -+-> decide (loop)
                        |              |                       +-> END (Ziel erreicht
                        |              +-> defer ---------------+    oder max_iterations)
                        |                                          +-> END
                   (System-2-Eskalation bei nicht auflösbarem Label)
```

`perceive`/`decide`/`guard`/`execute`/`defer`/`verify` sind async-Knoten in `src/mlaya/graph.py`.
`execute` ruft `MCPExecutor.call` und blockiert bis zum Settle (~40–110 s); der `MemorySaver`-
Checkpointer hält den Lauf resumierbar. `defer` überspringt den Foreground-Call, wenn der Server-
Survival-Reflex zuständig ist (Gefahr / keine gültige Aktion), und re-perceivt.

## Start dieser Codebasis (M1) — erreicht

**Kriterium:** Laya wählt aus echtem `view_status` eine Aktion, der Executor führt **einen**
echten MCP-Call inkl. Gating/Retrieval aus, der Verifier liest `view_status` erneut; N Iterationen
ohne Absturz. Jede Entscheidung landet in `runs/decisions.jsonl`.

Verifizierter Lauf (`run -n 4`): `craft_pickaxe` → `craft_item` (succeeded, `+wooden_pickaxe`),
danach `mine_stone` → `collect_block` (succeeded, `+8 cobblestone`) — alle gated, kein Absturz.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m mlaya.cli probe     # health + tools/list
.venv/bin/python -m mlaya.cli decide    # perceive + entscheiden (dry run)
.venv/bin/python -m mlaya.cli run -n 10 # echter Loop (mutiert die Welt)
```

## M2 — erreicht

**Kriterium:** Ressourcen in der Nähe in den State aufnehmen (`view_blocks`) und die Progression
über Steinwerkzeuge und Ofen bis zur **Eisen-Spitzhacke** führen (seit dem Diamant-Tier bis zur
**Diamant-Spitzhacke**).

Verifizierter Lauf gegen die echte Welt:

```
[0] craft_table          craft_item    +crafting_table, -1 acacia_log
[1] craft_stone_pickaxe  craft_item    +stone_pickaxe, -3 cobblestone, -2 stick
[2] mine_iron            collect_block +3 raw_iron (+cobblestone, +dirt)
[3] mine_coal            collect_block +3 coal
[4] craft_furnace        craft_item    +furnace, -8 cobblestone
[5] smelt_iron           smelt_item    -1 coal, -3 raw_iron, +3 iron_ingot
[6] craft_iron_pickaxe   craft_item    +iron_pickaxe, -3 iron_ingot, -2 acacia_planks
```

Der State zeigt jetzt z. B.:
`nearby resources: stone at 1m, coal_ore at 11m, iron_ore at 14m, logs at 16m.`

Wichtige Umsetzungsdetails:

- 3×3-Rezepte (Stein-Spitzhacke, Ofen, Eisen-Spitzhacke, Diamant-Spitzhacke/-Schwert, die vier
  Diamant-Rüstungsteile) laufen über `temporary_workstation: true` und brauchen einen **getragenen**
  `crafting_table` (`craft_table` vorher); Schmelzen braucht einen getragenen `furnace`.
  Rüstung braucht **keine Stöcke**, nur den Werktisch.
- `mine_iron`/`mine_coal`/`mine_diamond` werden nur angeboten, wenn im Umkreis tatsächlich Erz
  geladen ist (sonst fällt die Kaskade auf `explore` zurück). `mine_diamond` braucht zusätzlich
  Tier ≥ Eisen **und das fertige Eisen-Kampfkit** (`_combat_ready`: Eisenschwert + alle vier
  Eisenteile getragen + Schild in der Off-Hand — s. Abschnitt „Kampfkit"). Die Diamant-Ziele kommen
  aus **einer** Quelle, `_next_diamond_item(gs)`
  (Spitzhacke 3 → Schwert 2, erst wenn die Diamant-Spitzhacke steht → Rüstung in der Reihenfolge
  `DIAMOND_ARMOR` = Brustplatte 8, Hose 7, Schuhe 4, Helm 5, also **Schutz pro Diamant** zuerst).
  Es ist immer nur **das nächste fehlende Teil** aktiv: `mine_diamond` mit
  `count = Kosten − vorhandene Diamanten`, sonst `craft_diamond_<slot>`. **Keine Obergrenze beim
  Tier** (live gefundener Deadlock, s. u.): eine Bedingung `tier < Diamant` hätte verboten, dass die
  Diamant-Spitzhacke selbst die zwei Diamanten für das Schwert (und später die 24 für die Rüstung)
  abbaut — beides wäre unerreichbar gewesen.
- Ein gecraftetes Rüstungsteil liegt zunächst im Inventar; `equip_armor` schickt es mit **einem**
  `equip`-Batch (ohne `destination` — Rüstung findet ihren Slot selbst) an den Körper. Ein Teil
  zählt als „vorhanden", ob **getragen** oder im Inventar, damit nichts doppelt gecraftet wird.
  Achtung: `situation.tools.armour` ist nur eine **Bequemlichkeitsliste der mitgeführten** Rüstung
  pro Klasse — sie sagt nichts darüber, ob ein Teil angelegt ist. Der **getragene** Satz wird aus
  der Inventar-`location` gelesen (`head`/`torso`/`legs`/`feet`; ein angelegtes Teil verlässt
  `main`/`hotbar`). Vorher deutete der Parser `tools.armour` fälschlich als „getragen", wodurch
  `equip_armor` unterdrückt wurde.
- **Schild** (`craft_shield` → `equip_shield`): 6 Planken + 1 Eisenbarren (3×3, braucht den
  Werktisch). Er blockt nur in der **Off-Hand**, ein Craft landet aber nur im Rucksack — erst ein
  `equip` mit `destination: "off-hand"` legt ihn dorthin. Der Server meldet den Off-Hand-Stack mit
  `location: "off-hand"`; `GameState.offhand` liest das (die Rüstungs-Slots tun dasselbe), und
  `verify.diff`/`progressed` verbuchen `offhand_delta`.
- **Eisen-Kampfkit (vor dem Diamantgang):** `craft_iron_sword` (2 Barren + Stock, 3×3) und
  `craft_iron_<slot>` (Brustplatte 8, Hose 7, Schuhe 4, Helm 5 — `IRON_ARMOR`, wie beim
  Diamantsatz **Schutz pro Barren** zuerst; immer nur das nächste fehlende Teil über
  `_next_iron_armor_item`), dann `equip_armor`. `equip_armor` legt Eisen **und** Diamant an und
  tauscht nur gegen ein **besseres** Teil (`_armor_to_wear`), sonst würde das vom Diamantstück
  verdrängte Eisenteil im Rucksack sofort wieder darübergelegt. `mine_diamond` ist hinter
  `_combat_ready` gated (Eisenschwert **oder besser**, Schild, alle vier Slots mit Eisen **oder
  besser** belegt). **Grund (live, run12):** Der Bot tauchte nur mit einer Eisenspitzhacke zu den
  Diamanten (y<16), traf Zombie und Creeper und starb; der Lauf zum Todesort verlor das Rennen gegen
  den Item-Despawn. Die Eisenstufe ist die einzige Schwert-/Rüstungsstufe vor Diamant, die der Baum
  baut. **Eisen-Verbrauch:** `_iron_goal` ist die einzige Quelle für „wie viel Eisen noch". Sie ist
  `3` für die (Ersatz-)Hacke, sonst `_combat_iron_missing` (Schild 1 + Schwert 2 + fehlende
  Rüstungsteile), und mit fertigem Diamantkit zusätzlich `_portal_iron_goal` (Eimer, Feuerzeug).
  `mine_iron` **und** `smelt_iron` (Barren < Ziel) teilen sie; `mine_iron` ist pro Aktion auf
  `IRON_MINE_BATCH = 8` gedeckelt, damit ein Rüstungsziel nicht ein einziger unbegrenzter Abbau
  wird. Alle Eisen-Crafts sind hinter die Eisen-Spitzhacke gated, damit sie die ersten drei Barren
  nicht stehlen.
- `explore` ist bewusst der **letzte Fallback** und nur dann in `candidates`, wenn kein
  produktiver Schritt möglich ist — sonst verwässert es die (zero-shot verrauschte) Wahl. Ausnahme:
  in der Lull nach dem Anzünden des Portals (s. Essen/Auto-Stop) steht an seiner Stelle `wait` —
  der Bot hält an der Basis, statt in der Dämmerung wegzulaufen (live: 5 Pathfinder-Timeouts).
- **Umgebungs-Blocker (live gefunden & behoben):** In einem 1×1-Schacht nach dem Erz-Abbau
  antwortet `craft_item`/`smelt_item` mit `[WORKSTATION_PLACEMENT_FAILED] … no_cell` (keine
  Zelle für die getragene Werkbank). In der Schleife wäre das ein **Endlos-Retry**. Der
  `graph.Agent` merkt sich das blockierte Label (`blocked_by_surroundings`) und schiebt einen
  **Unblock** ein: zuerst `make_room` (`build_structure` mit `block_name: "air"` gräbt vier
  Zellen auf Bot-Höhe frei — billig, direkt am Erz), und erst in zweiter Instanz eine
  `explore`-Navigation auf offenes Gelände (derselbe M4-Zielpool). Danach greift der Craft.
  Live: tief genug gegraben, reproduzierte `craft_furnace` `no_cell`; `make_room` (4 Zellen)
  → dasselbe `craft_furnace` ging durch (`furnace` 3 → 4). Gepinnt in `tests/test_graph.py`
  und `tests/test_actions.py::MakeRoomTest`.
  **Erweitert (Rüstungslauf):** `sleep` scheiterte im selben 1×1-Schacht dreimal mit
  `[BED_PLACEMENT_FAILED] … no … flat ground` (drei verschwendete Ticks, das Bett braucht zwei
  gestützte Zellen). `BLOCKED_CODES` enthält jetzt auch `BED_PLACEMENT_FAILED`, der Unblock
  gräbt zuerst ein Pocket (`make_room`), in dem das Bett passt.
  **Fix (Schild-Lauf):** Die geplante Eskalation „erst `make_room`, dann `explore`" war
  **unerreichbar**: `_verify` löschte das Blocked-Label und nullte `_unblock_attempts` schon,
  wenn der *Unblock* gelang — also blieb es bei `sleep` → `make_room` → `sleep` → … (der Lauf
  verbrannte 12 von 16 Iterationen). Jetzt wird das Label nur gelöscht, wenn **das blockierte
  Label selbst** gelingt (`_note_block`); ein erfolgreicher Unblock lässt Zähler und Label
  stehen, sodass der zweite Versuch wirklich `explore` nimmt. Gepinnt in
  `tests/test_graph.py::GuardUnblockTest`.
  **Ergänzung:** Der Unblock feuert nur, wenn wirklich ein Label blockiert ist
  (`_blocked_label is not None`). Eine reine Gefahr-Deferral trägt `action_label=None` und darf die
  Leiter nicht re-arman — sonst schöbe ein unblockbarer Tick einen sinnlosen `make_room` nach.
  **Ergänzung (run12):** `BLOCKED_CODES` enthält auch `NO_REACHABLE_MATCHING_TARGETS` — der Server
  meldet das, wenn er die Zielblöcke *gefunden*, aber keinen *erreichbar* hat (`contract.ts`:
  „Every loaded X block was found but none could be reached", z. B. ein Baum, den er wegen
  Liquid-Isolation verweigert). In Place wiederholt sich das endlos (live: der Bot stand unter 56
  Bäumen und antwortete jeden Tick mit `get_wood` darauf); die Unblock-Leiter (Pocket, dann
  offenes Gelände) ist genau die Änderung, die so eine Verweigerung braucht. Gepinnt in
  `tests/test_graph.py::BlockedBySurroundingsTest`.
- **Beobachtung:** Ein 2×2-Craft (`craft_pickaxe`) hat die getragene Werkbank einmal in die Welt
  gesetzt statt sie zu behalten. Die Kette heilt sich selbst, weil `craft_table` erneut angeboten
  wird, solange keine Werkbank getragen wird.
- **Gefahr: Werkzeug bricht *innerhalb* eines langen Abbaus (live gefunden).** Die
  `replace_tool`-Reaktion feuert erst bei `LOW_DURABILITY = 0.15`, und nur *zwischen* zwei Ticks.
  Ein einzelner `collect_block`, der einen langen Tunnel zum Diamanterz gräbt, kann mehr
  Haltbarkeit verbrauchen, als bis zum nächsten Check übrig ist — beim Diamant-Lauf zerbrach so die
  **Eisen-Spitzhacke** mitten im `mine_diamond` (`status: partial`, 2 Diamanten, danach Tier 2).
  Danach wird `mine_diamond` nicht mehr angeboten (Tier < Eisen), und die Kette muss den
  Eisen-Nachbau vollständig durchlaufen (`mine_iron → mine_coal → smelt_iron → craft_iron_pickaxe`).
  Das funktioniert (live durchlaufen), ist aber teuer — dieselbe „selbstheilende, aber teure"
  Klasse wie die Werkbank-Beobachtung oben.
  **Härtung (drei Teile):**
  1. `replace_tool` greift jetzt zusätzlich bei absoluter Rest-Haltbarkeit
     ≤ `MINING_HEADROOM = 80` Nutzungen, sodass der frische Ersatz derselben Stufe schon getragen
     wird, *bevor* ein langer Grabungslauf ihn aufbrauchen kann; bricht die aktive Hacke dann, schiebt
     der Server den Ersatz automatisch nach und die Stufe bleibt erhalten. Ein
     `_has_spare_pickaxe`-Guard verhindert, dass die Reserve denselben Ersatz mehrfach craftet.
  2. `ORE_MIN_TIER` macht die Spielregel explizit (Bruchstein: jede Hacke, Kohle: jede Hacke, Eisen:
     `STONE_PICK_TIER`, Diamant: `IRON_PICK_TIER`; Gold zählt als Tier 1 und zerstört daher Eisen-
     wie Diamanterz). `_can_mine(gs, block)` prüft zusätzlich, dass ein Bruch *innerhalb* des Grabens
     die Mindeststufe nicht unterschreitet — sonst zieht der Server die nachgeschobene niedrigere
     Stufe nach und zerstört das restliche Erz ohne Belohnung. Stein und Kohle sind davon
     ausgenommen: sie sind billig, und ein Verbot würde den Bootstrap blockieren (eine abgenutzte
     Holzhacke muss noch den Bruchstein für den Stein-Ersatz sammeln dürfen). `candidates` **und**
     `resolve` nutzen dieselbe Prüfung, `mine_stone`/`mine_iron`/`mine_coal`/`mine_diamond` sind
     damit deckungsgleich.
  3. Die alte Obergrenze `tier < IRON_PICK_TIER` für `mine_iron` fiel weg: da eine Steinspitzhacke
     Eisen bereits abbaut (Holz zerstört es), kann eine abgenutzte Eisenhacke jetzt das Eisen für
     ihren eigenen Ersatz nachschürfen (`iron_wanted` greift zusätzlich bei
     `_pickaxe_due_for_spare`). Eine *frische* Eisenhacke schürft weiterhin kein Eisen.
  Die Diamant-Spitzhacke (1561 Haltbarkeit) macht das Risiko danach klein.

## M2b — erreicht: Bett/Respawn, Nachtschlaf, Notifications

**Kriterium:** Ein Bett bauen, nachts darin schlafen (überspringt die Nacht **und** setzt den
Respawn-Punkt), und die ungelesenen Notifications tatsächlich lesen und in den State heben.

- **Bett-Kette** in `actions.py`: `hunt_wool` (Schaf) oder `hunt_string` (Spinne) →
  `craft_bed` (`white_bed`, 3×3 → `temporary_workstation: true`) → `sleep`.
  `sleep` wird nur nachts und mit getragenem Bett angeboten; `sleep` findet/platziert das Bett
  selbst (`view_crafting_requirements` bestätigt: `white_bed` = 3 `white_wool` + 3 Planks,
  `white_wool` = 4 `string`).
- **`white_wool` ist aus `string` craftbar** — der Server rechnet die Kette rekursiv
  (`craft_item white_bed` macht aus `string` erst Wolle, dann das Bett).
- **Validierung (live):** die volle Kette ist durchgelaufen: `craft_table` (tagsüber) → tagsüber per
  `navigate` zu einer Tier-Gruppe geroamt, bis ein Schaf in Simulationsreichweite geladen war →
  `hunt_wool` (`collect_mob_drop sheep/white_wool ×3`, ergab **4 Wolle**) → `craft_bed` (`white_bed`)
  → auf die Nacht gewartet → `sleep` **`succeeded`**, Phase **Nacht → Tag** (Nacht übersprungen).
  `sleep` **platziert das Bett in die Welt** (das Inventar verliert es; das Bett stand danach bei
  (−324, 124, −6) zu Füßen). Zwei Randbedingungen aus dem Live-Lauf:
  - **`[HUNT_TARGET_NOT_LOADED]`** — die Mob-Liste reicht bis ~150 m (`nearest.position` vorhanden),
    aber nur **geladene** (simulierte) Mobs sind jagbar (`rangeBlocks: 16`); daher muss man erst
    hinlaufen, bis das Schaf geladen ist.
  - **Körper-Besitz:** nachts hält der Server-`hostile_reflex` (bzw. der **Hunger**-Reflex) zeitweise
    den Körper („No new action started; body owner: …"), sodass ein Foreground-Craft warten muss.
    `craft_bed` kam als `cancelled` (Hunger-Reflex bei Hunger 13) zurück — das Bett war danach
    trotzdem im Inventar, d. h. der Craft war vor der Unterbrechung schon durchgelaufen.
- **Sicherheits-Gate für Hostile-Jagd:** Ein Spinnen-Lauf hat den Bot in ein
  Zombie/Skelett/Creeper-Feld geführt und eine Eisen-Spitzhacke gekostet. `hunt_string` ist
  daher nur erlaubt, wenn **genau eine** Spinne ≤10 m, **kein weiterer Hostile ≤16 m** und
  Health ≥16 ist (`_safe_spider_hunt`); sonst bleibt es außen vor.
- **Notifications:** `read_recent_events` wird jetzt mit `limit: 50` gelesen (Drain des
  Event-/Notification-Backlogs) statt 5; die neuesten `events`-Summaries **und** die
  `notifications.recentPreview` stehen im State (`narrative()`: „unread notifications (N): …").
  (In M2d ersetzt durch einen vollen Backlog-Drain, siehe unten.)

## M2c — erreicht: Survival-Policy & Essen

**Kriterium:** Die Server-Survival-Policy explizit setzen (statt auf Defaults zu fahren) und eine
**Essensbeschaffung** ergänzen, die zu „tatsächlich überleben" passt.

- **Befund:** `eat_food` existierte, aber es gab **keine** Beschaffung — in einem frischen Lauf
  konnte es praktisch nie feuern. Schlimmer: die Survival-Policy wurde **nie** gesetzt. Der Server
  lief auf Default: `food.raw.allow: "emergency_only"`, `hunger_at_most: 6`, `health_below: 10` —
  der eigene Hunger-Reflex aß roh erst bei Hunger ≤ 6 **und** Health < 10.
- **`survival.py`** hält die Policy als **zwei Stufen** (Presets), entlang der dokumentierten
  Enums:
  - **`base`** (Reisen/Progression): `navigation.scaffold_blocks` inkl. Deepslate/Tuff,
    `hostile_avoidance_multiplier: 3`, das Combat-Fenster
    (`engagement: respond_to_threats`, `engage_min_health 14`, `critical_health 10`,
    `recover_to_health 16`, `retreat: true`, `enderman_wait_ticks 40`, Timeouts) und
    `food.raw.allow: "always"` mit `hunger_at_most: 20`.
    **Getunt (2026-10-13):** vorher `17/14/18`. Die Server-Beschreibung nennt `engage_min_health`
    ausdrücklich die Schwelle für eine **verfolgende** Aktion („immediate contact defence remains
    separate") und `critical_health` die Notfall-Schwelle **vor** Kontakt. Live fiel der Bot mit
    16,5 HP knapp unter die 17er-Verfolgungsschwelle, ging einer einzelnen Spinne aus dem Weg statt
    sie zu töten und blutete unter die alte Notschwelle 14 aus (hp 16,5 → 0,3, „hide, walled 9").
    Das Fenster `14/10` lässt ihn bei 14–16 HP kämpfen und hebt sich die Notfall-Schutzreaktion
    für < 10 auf. `recover_to_health 16` hält die Heilung zwischen Kämpfen konservativ.
  - **`hunt`** (Kampf-Stufe): dieselbe Basis mit **`engage_min_health 0`, `critical_health 0`,
    `retreat: false`** — sonst blockiert der Combat-Floor eine ausdrücklich angeforderte Jagd mit
    `[HUNT_CAPABILITY_BLOCKED]`. Die Null-Stufe ist bewusst **temporär**: mit ihr hält der
    Hostile-Reflex den Körper und blockiert Navigation, deshalb direkt danach zurück auf `base`.
  - Die dritte Stufe ist das **Auto-Essen**: `food.raw.allow` (`emergency_only` = Server-Default,
    der erst bei Hunger ≤ 6 **und** Health < 10 isst, `always` = hier) mit den Schwellen
    `hunger_at_most`/`health_below`.
- Verdrahtung: `apply_policy` beim **Start** von `run` (`base`); der Graph liest `last_death_age_s`
  und setzt `base` nach jedem Tod erneut (der Server stellt bei Tod/Dimensionswechsel auf Defaults
  zurück). Der `execute`-Knoten schaltet für eine **feindliche** Jagd (`is_hostile_hunt`) auf `hunt`
  und im `finally` zurück auf `base`. `mlaya.cli policy [--apply] [--preset base|hunt]`.
- **Essen** in `actions.py`: `hunt_food` beschafft den **Mindestvorrat** (`FOOD_RESERVE = 16`;
  Nether-Ziel 64, s. Lull) — nächstes passives Tier (`FOOD_MOBS = cow/pig/chicken/sheep/rabbit`) **innerhalb
  einer tageslichtabhängigen Reichweite** (`FOOD_HUNT_RANGE_DAY = 64` / `FOOD_HUNT_RANGE_NIGHT = 16`:
  nachts nur, was schon nah ist) oder, als Rückfall, ein **einzelner Zombie** (`rotten_flesh`,
  gegated: ≤2 Hostiles ≤16 m und alle Zombies, Health ≥18). `cook_food` brät rohes Fleisch
  (`smelt_item` im getragenen Ofen). `hunt_food` wird angeboten, solange der Vorrat < `_food_goal` ist
  **und** nicht gerade gegessen wird („feed to 20 first, then bank"); die Aktion ist mit
  `allow_without_shield: true` markiert. Gegessen wird über den Server-Reflex (roh erlaubt), `eat_food`
  bleibt der manuelle Nachschlag.
  **Zwei Schwellen (run14, live):** `eat_food` greift bei Hunger < `EAT_HUNGER = 14` (die alte
  Überlebensschwelle) **oder** bei einer Wunde, die nicht heilen kann: natürliche Regeneration
  verlangt Hunger ≥ `REGEN_HUNGER = 18`, also aß ein Bot zwischen 14 und 17 weder noch heilte er und
  blutete einfach aus (live: 17 Ticks bei 2,7 HP und Hunger 17, dann vom Mob getötet). Gefressen wird
  nur bei `health < HEALTH_MAX = 20`, damit ein unverletzter Bot mit Hunger 17 seine Reserve schont.
  **Lull-Nachschlag (`_nether_idle`, live gefunden):** Sobald das Diamant-Kit steht **und** das Portal
  brennt, ist nur noch das Nacht-Bett offen — der Bot fiel tagsüber in reines `explore` (live im
  From-Zero-Lauf: 25 Idle-Ticks, davon 5 × `[NAVIGATION_STOPPED]`-Timeouts). Diese Lull wird jetzt zur
  Nether-Vorbereitung genutzt: `_food_goal` hebt das Reservenziel auf `NETHER_FOOD_RESERVE = 64` (eine
  volle Kelle; live hatte der Bot 17 Esswaren). Die übliche Faustregel für den Nether sind **64–128
  Einheiten** hochersättigenden Essens, weil Hunger beim Sprinten und Kämpfen schneller fällt und der
  Tod das ganze Inventar kostet; eine Kelle gebratenes Fleisch ist nur ein Inventarplatz, also billig.
  **Die Reserve ist Teil des Auto-Stops:** `tech_tree_complete` verlangt jetzt `_food_reserve_met`
  (die 64 Einheiten aus `_food_goal`), und die Forderung fällt — wie ein Prep-Schritt — weg, sobald
  kein Tier und kein Zombie **in Reichweite** ist (`_food_source_in_reach`: passives Tier in
  Tagesjagd-Reichweite 64 m bzw. Zombie ≤ 24 m), damit eine Welt ohne Nahrung den Lauf nie
  blockiert. Live endete run13 sonst mit nur 4× `cooked_mutton`.
  `cook_food` darf in der Lull mit **jedem** Brennstoff arbeiten (`_furnace_fuel`: Kohle/Holzkohle
  zuerst, sonst Planken/Logs — ein Holz brät genau einen Gegenstand). Gegessen wird nach **Sättigung**
  (`FOOD_SATURATION`): gebratenes Fleisch vor Brot vor rohem Fleisch vor `rotten_flesh`. Goldene
  Karotten (14,4 Sättigung — die beste Nahrung im Spiel) brauchen Karotten und Gold, die dieser Bot
  weder anbaut noch schürft, und sind damit außerhalb der Reichweite. Außerhalb der Lull bleibt es beim
  alten Verhalten (nur Kohle), damit das Braten nie Holz stiehlt, das die Kette noch braucht. Findet
  sich gar keine Arbeit mehr, hält `wait` (s. u.) die Position. Getestet in
  `tests/test_actions.py::FoodChainTest`.
- **Gefahr (live, jetzt entschärft):** `hunt_food` wählte früher das **nächstgelegene** Tier
  unabhängig von der Distanz, und `collect_mob_drop` roamt dorthin. Ein nächtlicher `run` wanderte so
  **159–190 m** und wurde **zweimal getötet** (u. a. „slain by Enderman"), was **das komplette
  Inventar** kostete — inklusive der gerade gecrafteten Eisenhacke. Auch `hunt_string` scheiterte
  dreimal, einmal tödlich. Der `in_danger`-Gate sagte in diesen Ticks **`no`** (Health 10.5, Skelett
  9.6 m) — das Zero-Shot-Gate ist auf Gefahr schlecht kalibriert (bekannter Befund, s. o.). Die
  Server-Policy kämpft erst ab Health ≥ 17 (`respond_to_threats`), greift unterhalb also nicht ein.
  **Fix:** `hunt_food` ist jetzt an eine **Tageslicht-/Distanz-Schranke** gebunden; nachts wird nur
  gejagt, was schon in Reichweite ist. Live gegengeprüft mit denselben Tieren (35/38/75/103 m):
  **nachts** (`range=16`) → `_food_target = None`, `hunt_food` fehlt in `candidates`; **tagsüber**
  (`range=64`) → `hunt_food` angeboten und ausgeführt (`collect_mob_drop sheep/mutton ×6` → 6 Mutton +
  4 Wolle). Rest-Risiko: tagsüber bleiben längere Wege bis 64 m möglich — bewusst, weil der Tagesweg
  billig ist.

## M2d — erreicht: auf Event-/Notification-Typen reagieren

**Kriterium:** Events nicht nur in den State heben, sondern auf ihre **Typen reagieren**
(„durability low" → Ersatzwerkzeug craften, „target died" → Drops ernten, Tod → Recovery).

- **Befund:** `view_status` liefert bereits die **Haltbarkeit des aktiven Werkzeugs**
  (`tools.tools[].durabilityLeft/maximumDurability/item`) und die **Todesposition**
  (`lastDeath.position`/`dimension`). Für Durability und Tod ist also **kein** Event nötig —
  nur für „target died", dessen Drops in der Welt liegen.
- **Event-Drain:** `read_recent_events` liefert die **ältesten** ungelesenen Events (max. 100),
  nicht die neuesten — bei einem 841 tiefen Backlog sah der State daher nur alte Events. `build()`
  walkt jetzt bis `remainingEventCount == 0` (Cap `EVENT_DRAIN_PAGES = 12` pro Tick), sodass jeder
  Tick die **frischen** Events sieht; `notifications.unreadCount` steht danach auf 0.
- **Reaktionen** (`actions.py`): drei neue Labels, jeweils **exklusiv** für den Tick (zeitkritisch),
  `eat_food` läuft weiter mit. Priorität: recover > replace > drops.
  - **`replace_tool`** — die aktive Spitzhacke braucht einen Ersatz bei `LOW_DURABILITY = 0.15`
    **oder** bei absoluter Rest-Haltbarkeit ≤ `MINING_HEADROOM = 80` Nutzungen (die Reaktion läuft
    nur *zwischen* Ticks, ein einzelner langer `collect_block` kann mehr verbrauchen, s. o.), und ein
    Ersatz ist craftbar (`_replacement_pickaxe`: Diamant bei ≥3 Diamanten, sonst Eisen, sonst Stein;
    nie ein unnötiger Downgrade). `_has_spare_pickaxe` verhindert einen zweiten Ersatz derselben Art
    (sonst würde die Reserve endlos nachcraften, wenn der Server eine *niedrigere* Stufe liegen
    lässt). → `craft_item` mit `temporary_workstation`.
  - **`collect_drops`** — frisches „target died"-Event (`gs.drops > 0`, Alter ≤
    `EVENT_REACT_WINDOW_S = 120` s) → `pick_up_items {radius: 16}`.
  - **`recover_drops`** — Tod ≤ `RECOVER_WINDOW_S = 300` s, gleiche Dimension, ≤ `RECOVER_RANGE =
    320` m → `pick_up_items {recover_death_items: true}` (der Server läuft selbst zur
    Todesposition zurück). Zusätzlich braucht ein **tiefer** Tod (`death_y < bot_y −
    RECOVER_DIG_DEPTH = 4`) eine **Spitzhacke** (`_can_recover`): der Weg zurück muss sich
    hinuntergraben, und mit der Hand verliert der Bot das Rennen gegen den Despawn (live, run12:
    von der Oberfläche handgegraben zu einem Tod ~60 Blöcke tiefer, verloren, danach Endlos-loop
    auf `get_wood`). Bis die Hacke gecraftet ist, läuft die normale Kette (Holz → Hacke) zuerst.
    Nach jedem Tod setzt der Graph die Policy neu (`_reapply_after_death`).
    **Schreibabbruch (run14, live):** Der Server antwortet `[NO_MATCHING_ITEMS]`, wenn er die
    Todesstelle erreicht und sein 16-m-Scan nichts mehr findet (die Drops sind despawned/verbrannt)
    — eine **definitive** Antwort, keine transiente. `_can_recover` bot den Weg trotzdem das ganze
    Fenster über an, sodass die Reaktion jeden Tick den Körper hielt und der Bot an der leeren
    Stelle stehen blieb, bis das Fenster ablief (und ein neuer Tod es zurücksetzte): live 40 + 77
    Ticks in zwei solchen Schleifen, über die Hälfte des Laufs, nachdem das Diamant-Kit ohnehin
    verloren war. `Agent._note_recover` merkt sich daher die Todeszelle (`actions._cell`), sobald
    ein finaler Code kommt (`RECOVER_FINAL_CODES` = `NO_MATCHING_ITEMS`, `NO_RECORDED_DEATH`,
    `DEATH_DIMENSION_MISMATCH`, `DEATH_RECORD_STALE`; nicht aber das transiente
    `RECOVERY_SITE_UNREACHABLE`), `GameState.recover_written_off` trägt sie und `_can_recover`
    lehnt diese Zelle ab. Ein späterer Tod woanders ist eine neue Zelle und wird weiter gelaufen.
- **State:** `GameState.tool_meta` (Klasse → `{tier,item,left,max}`), `death_pos`, `death_dim`,
  `drops`; `narrative()` hängt „low durability: …", „… mob(s) the reflex killed dropped items
  nearby" und „gear … can still be recovered" an. `verify.progressed` prüft `replace_tool` (neue
  `*_pickaxe`) und `collect_drops`/`recover_drops` (Netto-Inventar).
- **Validierung (live, alle drei Reaktionen):**
  - `replace_tool` **end-to-end** — die Steinspitzhacke kontrolliert auf 15/131 (`≤ LOW_DURABILITY`)
    abgenutzt → `candidates == ["replace_tool"]` → Laya wählt es → `_guard` bindet
    `craft_item {items:[{item_name:"stone_pickaxe"}], temporary_workstation:true}` → `succeeded`,
    danach 131/131.
  - `collect_drops` — mit der aggressiven `hunt`-Policy killte der Server-Reflex einen Mob (Event
    `hostile: target died`) → `drops = 1` → `candidates == ["collect_drops"]` → Laya wählt es →
    `_guard` bindet `pick_up_items {radius: 16}`. Die Ausführung antwortete `[NO_MATCHING_ITEMS]`
    (die Drops waren zu diesem Tick bereits entladen/weg), also ist die **Entscheidung + Bindung**
    belegt, nicht das tatsächliche Einsammeln.
  - `recover_drops` **end-to-end** — provozierter Tod in Place bei (−90.6, 71, −14.9), Respawn
    (−52.5, 68, 8.5) ≈ 45 m entfernt, 8 s her → `candidates == ["recover_drops"]` → Laya wählt es →
    `_guard` bindet `pick_up_items {recover_death_items: true}` → `succeeded`; das Inventar
    (Dirt/Cobble/Logs/Planks/Sticks + String) kam zurück. Die `hunt`-Policy wird dabei nach dem Tod
    automatisch wieder auf `base` gestellt (`_reapply_after_death`).
  - **Observed Event-Wortlaut:** `hostile: target died`, `hostile: contact ended`,
    `MineAI died at <pos> …`, `<tool> broke.` — `_count_drops` greift auf `"target died"`.
  - 22 Unit-Tests (`tests/test_actions.py`) pinnen zusätzlich alle Bindungen/Guards sowie die
    Bett- und Essenskette deterministisch.

## M3 — erreicht: Logs → Datensatz → RLCD-Kopf-Fine-Tuning

**Kriterium:** Aus dem Run-Log einen Trainingsdatensatz bauen, Laya auf den eigenen Entscheidungen
nachtrainieren (Kopf-Fine-Tuning) und den Checkpoint über `LAYA_EXTRA_MODELS` registrieren.

- **Datensatz** (`src/mlaya/dataset.py`, CLI `mlaya dataset`): `runs/decisions.jsonl` →
  Zeilen `{state, questions, expected}`. `state` ist die `narrative()`-Zeichenkette, die Laya im
  Betrieb sah; `questions` nutzt **exakt dieselben** Builder wie die Inferenz
  (`brain.intent_question`, `brain.DANGER_Q`), damit der Kopf auf demselben Rendering trainiert
  wird, das er später sieht.
- **Reward = Verifier.** Ein Tick, dessen Aktion `progressed` ist, ist ein Positivbeispiel für
  genau diese Aktion in diesem Zustand; ein Tick ohne Fortschritt hat keinen bekannten guten
  Alternativkandidaten und wird verworfen (`--include-failed` behält ihn als reine Imitation).
  Nur Ticks mit **mehr als einem** Kandidaten tragen eine `intent`-Wahl — ein Ein-Kandidaten-Tick
  ist deterministisch (der Filter hat ihn schon gewählt) und lehrt nichts. Dedup über
  `(state, candidates)`.
- **Warum Danger-Zeilen:** Laya wählt den Kopf über den **Fragetyp** (das Checkpoint trägt ein
  `type_emb.`), also teilen sich `in_danger` und `intent` **einen** `choice`-Kopf. Ein reiner
  Intent-Datensatz würde das Gate stillschweigend verlernen. Daher eine `in_danger`-Zeile pro
  Zustand: eine dokumentierte Heuristik („ein Mob greift an oder Gesundheit sehr niedrig",
  `DANGER_CLOSE_M = 4`, `DANGER_LOW_HEALTH = 8`) ist der **Boden**, eine geloggte Laya-Antwort
  greift nur, wo die Heuristik schweigt. **Getunt (2026-10-13):** vorher hatte die geloggte
  Antwort Vorrang. Live antwortete das Gate bei **0,3 HP mit Spinne in Reichweite** trotzdem
  `no` — genau die Fehlkalibrierung, die das Training beheben soll; jetzt gewinnt die Heuristik.
  **Obergrenze (2026-10-14):** Ein Zustand, der sich selbst als sicher erweist — Health ≥
  `DANGER_SAFE_HEALTH = 15` **und** der nächste Hostile **bekannt** jenseits `DANGER_SAFE_M = 8`
  m (dieselben Schwellen wie das Laufzeit-Band, s. u.) — wird als `no` gelabelt, auch wenn das
  Gate `yes` sagte. Sonst würde das Über-Auslösen für einen bloß geladenen Mob eintrainiert
  (live: run7b `yes` bei 9–24 m, Health 20). Nur ein Zustand, der seine eigene Sicherheit
  beweist, kippt die Obergrenze; ist die Hostile-Distanz unbekannt (leere `hostiles`-Liste),
  gilt weiter die geloggte Antwort.
  `--no-danger` lässt die Zeilen weg. `graph._verify` loggt dazu jetzt kompakt `answers`
  (`{qid: choice|score}`), damit künftige Läufe das echte Gate-Label liefern.
- **Split:** stabiler Hash über den State (`--eval-frac`, `--seed`), reproduzierbar; < 4 Zeilen
  ⇒ kein Eval-Slice.
- **Trainieren** (`mlaya train`, Wrap um `laya.train.finetune`): `--loss rlcd` (GRPO-artig + Soft-CE)
  oder `soft-ce`, `--train-encoder` (Default: nur Köpfe, `--freeze-encoder`), Option-Order-Augmentation
  für `choice` (`shuffle_options=("choice",)` gegen Positions-Bias), `--dry-run` prüft Datensatz
  ohne Gewichte.
- **Registrieren:** `LAYA_EXTRA_MODELS` ist ein JSON-Objekt `{name: quelle}`; ein Name, der ein
  eingebautes Checkpoint trifft, **zeigt es um**. `config.load_settings` parst die Variable nach
  `laya.extra_models` und `Brain` reicht sie als `models=` an den `Router` weiter. Damit lädt der
  Loop das Fine-Tune unter `english` **ohne Codeänderung**.
- **Validierung (frischer Welt-Lauf, 48 Ticks → Ziel):** kompletter Tech-Tree aus dem Nichts,
  `stop: "goal"`. Datensatz: **35 Zeilen** (26 Train / 9 Held-out), 17 Intent-Labels über die
  ganze Kette (`get_wood` … `equip_armor`). Kopf-Fine-Tuning (4 Epochen, `rlcd`, Encoder
  eingefroren) → 842 MB Checkpoint. Held-out (9 Items): `choice`-Accuracy **1.000 → 1.000**,
  Loss **0.734 → 0.532 (−0.20)**, ECE **0.501 → 0.392**, Brier **0.412 → 0.284**, `mean_confidence`
  0.499 → 0.608. Der `Router` lädt den Checkpoint unter `english` ohne Codeänderung. 24 Unit-Tests
  (`tests/test_dataset.py`, stdlib `unittest`).
- **Befund (Accuracy-Deckel):** Die Held-out-Accuracy ist **schon zero-shot 1.0** — der Split ist
  in-distribution (jede Zeile ist ein Zustand, den Laya im Lauf selbst gelöst hat). Ein
  Accuracy-Gewinn ist hier nicht messbar; der Fine-Tune zeigt sich in Loss, Brier und ECE. Für
  einen echten Accuracy-Vergleich braucht es einen **out-of-distribution**-Split (Zustände aus
  einem anderen Lauf), nicht mehr In-Distribution-Zeilen.
- **OOD-Ergebnis (2026-10-14):** Datensatz über **mehrere Läufe** statt eines einzelnen: 252
  Train-Zeilen (nur Ticks mit Fortschritt, über die Läufe dedupliziert; Stall-Läufe und der
  Über-Auslöser-Lauf `run7b` raus, `in_danger=yes` 28). Held-out ist ein **kompletter Lauf**
  (`run7d`, 61 Items, kein Zustand im Training). Die Zero-Shot-Accuracy auf diesem OOD-Set ist
  **0.934** (4 Fehler) — also *kein* 1.0-Deckel. Der Fine-Tune (4 Epochen, `rlcd`, Encoder
  eingefroren, MPS) lässt die Accuracy unverändert — die Labels sind die geloggte Gate-Wahl, und
  Imitation kann deren Fehler nicht beheben — verbessert aber die Kalibrierung deutlich:
  **Loss 0.474 → 0.372, ECE 0.250 → 0.132, Brier 0.261 → 0.187**, `mean_confidence` 0.691 → 0.891.
  (Der Kopf wird also treffsicher-konfident, nicht treffsicherer.)
- **Gefahr-Ticks landen jetzt im Datensatz:** Ein `in_danger=yes`-Tick führt zur **Defer** (kein
  `action`-Record), der Builder sah aber nur `action` → `danger_yes: 0`, das Gate hätte nur `no`
  gelernt. `graph._defer` loggt jetzt `narrative` + `answers`, und der Builder macht aus einem
  Defer mit Gate-`yes` (oder der Danger-Notiz) eine **reine Danger-Zeile**. Ein Defer aus anderem
  Grund („no valid action") bleibt ignoriert; die State-Heuristik wird dort bewusst **nicht**
  verwendet. Getestet in `tests/test_dataset.py` (Defer-Fälle). Für den frischen Lauf kam der Fix
  zu spät — seine 10 Gefahr-Ticks sind alte `defer`-Records ohne `narrative`; der Datensatz wurde
  daher mit `--no-danger` gebaut und das Gate bleibt zero-shot.
- **Abgelehnte Gefahr-Ticks (2026-10-13):** Ein `action`-Tick mit `progressed=false` (z. B. der
  Körper gehörte dem Hostile-Reflex) wurde bisher **ganz** verworfen — genau das waren live die
  härtesten Zustände (hp 0,3–6,5 mit Spinne). Der Builder emittiert für so einen Tick jetzt eine
  **reine Danger-Zeile** (`in_danger: yes`), wenn die Heuristik ihn als gefährlich einstuft: kein
  brauchbares Intent, aber ein ehrliches Gate-Positiv. Getestet in
  `tests/test_dataset.py::BuildRowsTest::test_a_refused_danger_tick_becomes_a_yes_row`.

- **Danger-Band zur Laufzeit (2026-10-13, verschärft 2026-10-15):** Das Gate antwortet aus dem
  Narrativ, das nur „hostile mobs within 16 blocks" sagen kann — es sieht die Distanz nicht und liest
  einen bloß geladenen Mob als Gefahr. `yes` defert **bedingungslos**, die Server-Policy hatte aber
  nichts zu tun (hp/hunger voll, kein Kontakt): run7b stand auf Diamanttiefe Tick um Tick still. Zwei
  Fixes greifen: (1) `perception.parse_status` liest die Hostile-Distanz jetzt **direkt** aus
  `nearby.hostiles` (vorher aus `nearest.distance`, wo sie fehlt → jede Distanz war `None`, das
  Narrativ ohne „at Xm"); (2) `brain.danger_matters` begrenzt das Gate: ein `yes` übergibt den Tick
  **nur**, wenn ein Hostile ≤ `DANGER_DEFER_M` m ist oder Health < `DANGER_DEFER_HEALTH = 15`.
  `DANGER_DEFER_M` war 8 m („weitere Schwelle als der Trainings-Boden"), doch run13 zeigte den
  Preis: ein Skelett/Creeper, der 5–8 m hielt (in Schussweite, nie im Kontakt), defertierte ~60
  Ticks lang, während die Policy idle war — der Loop fror in einer Höhle. Die Schwelle ist daher
  jetzt **Kontaktreichweite** (`DANGER_DEFER_M = 4`), gleich dem Trainings-Boden `DANGER_CLOSE_M`;
  die Server-Reflexe schützen den Kontakt ohnehin unabhängig, der Loop muss nur bei echter
  Körpergefahr (Kontakt oder niedrige HP) beiseitetreten. Der Builder hält seine **Obergrenze**
  (`DANGER_SAFE_M = 8` / `DANGER_SAFE_HEALTH = 15`) bewusst **weiter** als die Laufzeit (eigene
  Konstante statt des früheren Imports), damit im unentschiedenen Band zwischen Untergrenze (≤ 4 m)
  und Obergrenze (> 8 m) die geloggte Gate-Antwort stehen bleibt, statt `no` zu erzwingen.
  Die `DANGER_Q`-Kriterien wurden zudem geschärft („within a few blocks and closing in"). Getestet
  in `tests/test_brain.py` und `tests/test_perception.py`.

## M4 — erreicht: Parameter-Layer (Zielwahl über `view_blocks`-Kandidaten, `predict_shortlist`)

**Kriterium:** Ein Label sagt *was* (z. B. „explore“), aber nicht *wo*. Der Parameter-Layer füllt
die Argumente aus dem gegroundeten Zustand: statt eines fest verdrahteten Offsets wählt Laya ein
konkretes Ziel aus den `view_blocks`-Kandidaten, und `predict_shortlist` hält große Kandidatenmengen
im Kopf-Budget. Es ist dieselbe Teilung wie in der Label-Schicht — **Code filtert, Laya wählt** —
nur eine Ebene tiefer.

- **Perception** (`src/mlaya/perception.py`): `view_blocks`-Treffer werden jetzt als `BlockHit`
  behalten (`name`, `x/y/z`, `distance`, `above`, `below`). Der Server annotiert jeden Treffer mit
  `above`/`below`; `above == "air"` ist **echter Standgrund** — laut Run-Skill der verlässliche Weg,
  ein `navigate`-Ziel zu wählen. Zusätzlich zur Ressourcen-Suche fragt `build()` dieselbe
  `view_blocks`-Anfrage um `GROUND_NAMES` (`grass_block`, `sand`, `gravel`, `dirt`) ab.
- **`src/mlaya/targets.py`**: `Target` (Label + Beschreibungstext + `BlockHit`) und
  `ground_targets(gs)`. Es behält nur Standgrund aus `DESTINATION_QUERIES` (Boden + `stone`),
  dedupliziert pro `(x, z)`-Spalte, verwirft Gelände **unter den Füßen** (`EXPLORE_MIN_M = 4`) und
  sortiert **am weitesten zuerst** — so ist ein Explore-Pick ein echter Schritt nach draußen; ist der
  Bot eingeschlossen, wird stattdessen der nächste Grund angeboten. Die Texte bleiben kurz (Distanz +
  Kompassrose), weil sich alle Optionen ein `head_max_len`-Budget teilen.
- **Brain** (`src/mlaya/brain.py`): `TARGET_QID = "target"`, `target_question(targets)` und
  `choose_target(state, targets)`. Bis `TARGET_K = 6` Kandidaten läuft ein normaler `predict`; ein
  größerer Pool wird per `laya.shortlist.predict_shortlist` eingeengt, dessen `embed_fn` aus dem
  bereits geladenen Checkpoint kommt (`embed_fn_from_agent(router.load(model))`, einmalig gecacht).
  Kein Treffer ⇒ Rückfall auf den entferntesten Kandidaten.
- **Graph/Actions** (`src/mlaya/graph.py`, `actions.py`): `TARGET_LABELS = {"explore"}`. `_decide`
  baut für ein solches Label `ground_targets` und legt die Wahl als `Decision.target` (+
  `target_meta`) ab; `_guard` ruft `resolve(label, gs, target)`. `explore` bindet dann
  `navigate {x, y, z, dig, scaffold}` an die **echte** Zielposition (mit `y`, sonst rät `navigate`
  über Dächer/Laubwerk); ohne Kandidaten bleibt der alte `x + 16`-Fallback. Das Action-Log trägt
  jetzt `target` und `target_meta`.
- **Validierung (live):** `view_blocks` lieferte 8 Gruppen × 4 Treffer; daraus 7 Standable-Kandidaten,
  nach dem `min_distance`-Filter 4 (`grass_block` ~8 m nördlich). `ground_targets` beschriftet sie
  korrekt (Distanz + Richtung), `predict_shortlist` (k=2 erzwungen) engt 7 → 2 ein, Laya wählt
  `grass_block@-42,10`, und `_guard` bindet `navigate {x:-42, y:65, z:10, dig:true, scaffold:true}`.
  Auch der Pfad ohne Shortlist (Pool ≤ `TARGET_K`) und der Rückfall ohne Kandidaten sind geprüft.
  13 Unit-Tests (`tests/test_targets.py`, stdlib `unittest`).
- **Vorbehalt 1 (Reichweite):** `view_blocks find` listet je Query die **nächsten** Treffer (Limit 4).
  Die Kandidaten sind also nahes Gelände, kein Fernziel — der Parameter-Layer wählt *Richtung*, nicht
  eine Reise über die Karte. Für echte Fernerkundung bräuchte es `view_blocks box` in einem Ring oder
  `view_frontier` und damit eine eigene Abfrage.
- **Vorbehalt 2 (Häufigkeit):** `explore` ist der **letzte Fallback** und feuert selten; der Mechanismus
  ist bewusst generisch (`TARGET_LABELS`, `resolve(..., target)`), weitere Labels können ihn später
  adoptieren.

## Nächste Meilensteine

- **M3+ (offen):** Der **out-of-distribution**-Held-out-Split ist jetzt gebaut (kompletter Lauf
  `run7d`): die Zero-Shot-Accuracy liegt dort bei **0.934**, der Fine-Tune verbessert Loss/ECE/Brier,
  aber **nicht** die Accuracy — die Zeilen imitieren die geloggte Gate-Wahl, und Imitation kann deren
  Fehler nicht beheben. Für einen echten Accuracy-Gewinn bräuchte es ein stärkeres Signal als
  Imitation (z. B. Verifier-Reward direkt) oder mehr/vielfältigere Läufe. Beide `in_danger`-Klassen
  sind mit dem Defer-Fix vertreten; die neue Obergrenze kappt nur die fernen `yes`.
- **Eisen-Kette live (re-)validiert (Tier 0 → 3):** `craft_pickaxe` → `mine_stone` →
  `craft_stone_pickaxe` → `mine_iron` → `mine_coal` → `craft_furnace` → `smelt_iron` →
  `craft_iron_pickaxe` ist **live durchgelaufen**, inklusive der neuen Unblock-Recovery
  (`[WORKSTATION_PLACEMENT_FAILED]` → `make_room` gräbt ein Pocket → Craft greift; `explore`
  als zweite Instanz).
- **M2/M2b/M2c/M2d live validiert:** `replace_tool`, `hunt_food`, `collect_drops`
  (Entscheidung/Binding), `recover_drops` (s. M2c/M2d) sowie `craft_table`, `hunt_wool`, `craft_bed`,
  `sleep` (s. M2b) sind **live durchgelaufen**. Damit bleibt in M2/M2b/M2c/M2d kein Pfad ohne
  Live-Lauf; die Bett-/Essensketten sind zusätzlich durch die Determinismus-Tests gepinnt.
- **Diamant-Tier (live validiert):** `mine_diamond` → `craft_diamond_pickaxe` → (2 weitere
  Diamanten) → `craft_diamond_sword` ist **live durchgelaufen**; Endstand
  `tools = {"pickaxe": "diamond", "sword": "diamond"}` (Tier 4), `tests/test_actions.py::DiamondChainTest`.
  Zwei live gefundene Punkte: (a) die Eisen-Spitzhacke zerbrach *innerhalb* eines langen
  `collect_block` (s. o.), die Kette heilte per Eisen-Nachbau selbst; (b) ein ursprünglicher
  Tier-Deckel auf `mine_diamond` hatte das Schwert **unerreichbar** gemacht (Deadlock) — der Deckel
  ist entfernt, `mine_diamond` kennt jetzt die Zielmengen aus `_next_diamond_item`
  (3 Spitzhacke, 2 Schwert, dann das Rüstungsset).
- **Diamant-Rüstung (live validiert):** `DIAMOND_ARMOR` (Brustplatte 8, Hose 7, Schuhe 4,
  Helm 5 = 24 Diamanten) hängt sich über `_next_diamond_item` an dieselbe Kette; ein fehlendes
  Teil pro Tick, `craft_diamond_<slot>`, danach `equip_armor` (ein `equip`-Batch, ohne
  `destination`). Perception liest den **getragenen** Satz aus der Inventar-`location`
  (`head`/`torso`/`legs`/`feet`); `situation.tools.armour` ist nur die Liste der mitgeführten
  Rüstung pro Klasse. `verify` zählt ein getragenes Teil über `armor_delta` als Fortschritt.
  Unit-getestet in `tests/test_actions.py::DiamondArmorTest` (9 Tests) und
  `tests/test_perception.py::ArmourParseTest` (2 Tests). **Live durchgelaufen** (25 Ticks):
  `mine_diamond` (count 8 → 7 → 4 → 5, zusammen exakt die 24 Diamanten) →
  `craft_diamond_chestplate` → `craft_diamond_leggings` → `craft_diamond_boots` →
  `craft_diamond_helmet`; `tools = {pickaxe, sword: diamond}`, hp 20. **Korrigierter Befund:** der
  Server rüstet **nicht** selbst an — die gecrafteten Teile bleiben im Inventar (`location: main`),
  bis ein `equip` sie in einen Rüstungsslot legt. Der erste Endstand „alle vier in `armor`" war ein
  Parser-Artefakt (s. o.); nach explizitem `equip` zeigen die Stacks `head`/`torso`/`legs`/`feet`
  und `tools.armour` erst dann den getragenen Satz.
  **Zweiter Befund:** `sleep` scheiterte im 1×1-Schacht 3× mit `[BED_PLACEMENT_FAILED]`; der
  Unblock-Mechanismus deckt diesen Code jetzt mit ab (s. M2).
- **Schild (live validiert):** `craft_shield` (6 Planken + 1 Eisenbarren, `temporary_workstation`)
  → `equip_shield` (`equip` mit `destination: "off-hand"`). Perception liest den Off-Hand-Stack
  über `location: "off-hand"` in `GameState.offhand`; `verify` verbucht `offhand_delta`. Der Fund,
  der die Tier erst erreichbar machte: mit stehender Diamant-Spitzhacke war `iron_wanted` falsch,
  also bot der Loop live nur `['explore']` an, obwohl `iron_ore at 4m` in Reichweite lag —
  `_iron_goal` (3 für die Hacke/Ersatz, sonst 1 für den Schild, sonst 0) zieht das Eisen. Getestet
  in `tests/test_actions.py::ShieldTest` und `tests/test_perception.py::OffhandParseTest`.
- **Auto-Stop bei erreichtem Tech-Tree (implementiert):** `actions.tech_tree_complete(gs)` ist wahr,
  sobald das ganze Diamant-Kit steht — Diamant-Spitzhacke + Diamantschwert, **alle vier**
  Diamant-Rüstungsteile **getragen** (`_worn`, nicht nur im Inventar — sonst würde der Lauf enden,
  bevor `equip_armor` greift) und der Schild in der Off-Hand — **und** das Overworld-Ziel steht: das
  Portal ist angezündet, ein Bett daneben platziert, **kein Nether-Prep-Schritt mehr offen**
  (`_all_nether_prep_steps`: Bogen+Pfeile, Goldhelm, tiefer Bruchstein) **und die Essens-Reserve
  gebankt** (`_food_reserve_met`, die 64 Einheiten aus `_food_goal`). Prep-Schritte wie die
  Essens-Forderung fallen weg, sobald ihr Rohstoff nicht erreichbar ist (kein Spawn, kein Erz, kein
  Tier/Zombie in Reichweite), ein unerreichbares Item blockiert den Lauf also nie. `[loop]
  stop_when_complete` (Default `true`) lässt `verify`/`defer` den Lauf dann mit `stop: "goal"`
  beenden statt das Iterationsbudget abzuarbeiten; sonst endet er mit `stop: "limit"`.
  `run --ignore-goal` schaltet die Abbruchbedingung für einen Lauf ab (Datensatz-Sammlung über das
  Ziel hinaus). Getestet in `tests/test_actions.py::TechTreeCompleteTest` und
  `tests/test_graph.py::GoalStopTest`.
- **Nether-Kette (implementiert):** Neun Labels in `actions.py` führen vom fertigen Diamant-Kit zum
  offenen Portal: `craft_bucket` (3 Eisenbarren) → `fill_bucket` → `mine_obsidian` → `get_flint`
  (aus Gravel) → `craft_flint_and_steel` (2×2, **kein** Werktisch) → `clear_portal` (störende Blöcke
  aus der Öffnung, s. Homebase) → `build_portal`
  (`build_structure portal_frame`, 10 Obsidian + 4 Cobblestone-Ecken) → `light_portal`
  (`activate_portal`) → `place_bed` (`sleep`). Die ganze Kette hängt an `_gear_ready` (ein frischer
  Lauf macht erst die Diamant-Tier fertig).
  **Obsidian wird gemacht, nicht gegraben:** `mine_obsidian` hat in `resolve` zwei Phasen. Liegt
  Obsidian in Reichweite (`OBSIDIAN_REACH` = 6 m — das, was der letzte Guss geformt hat), wird es
  abgebaut (`collect_block`, genau die fehlenden Blöcke bis 10). Sonst wird der Wassereimer auf die
  nächste Lava-Zelle gegossen (`use_bucket pour` mit `x,y,z`, ab `POUR_REACH` = 4,5 m; sonst erst
  dorthin navigieren — **ohne** `dig`/`scaffold`, damit der Pathfinder nicht in die Lava tunnelt).
  Als „offen" über der Guss-Zelle zählt neben `air` auch `cave_air` (`perception.AIR_NAMES`):
  unterirdische Lavaseen sind von Höhlenluft statt `air` überdeckt, und die Guss-Prüfung verwarf
  solche Zellen zuerst (live im From-Zero-Lauf behoben).
  Natürliches Obsidian liegt nur tief (`y ≈ -55`) unter denselben Lavaseen und fällt nie unter
  `OBSIDIAN_REACH`; dorthin zu graben ist die **Wasser-Falle** (siehe Reflex unten). Angeboten wird
  der Schritt nur bei „Eimer voll **und** Lava in Reichweite“ oder „Obsidian in Reichweite“.
  **Obsidian-Latch (Schlüsseldesign):** Obsidian muss **vor** dem Flint&Steel gebankt sein — genau
  dieser Craft ist die Sperre, die `mine_obsidian` nach dem Bau (der die 10 Blöcke ausgibt) am
  erneuten Graben hindert, weil eine reine „Ressource vorhanden“-Prüfung den frischen Rahmen sonst
  maskieren würde. Damit ein **teilweise** gebauter Rahmen nicht in eine Sackgasse läuft, prüft
  `_portal_frame_started` die feste Rahmenebene (`_portal_site` ± Nachbarn) auf Obsidian: liegt dort
  welches, darf `build_portal` mit dem Rest im Inventar **fortsetzen** (nach einem Teilbau ist die
  Restmenge genau die noch fehlenden Zellen), und `light_portal` feuert nur, wenn wirklich Obsidian
  **im Rahmen** steht (`has_fs and obsidian < 10 and _portal_frame_started`) — natürliches Obsidian
  neben der Basis zählt nicht mehr.
  Getestet in `tests/test_actions.py::NetherGateTest` und `tests/test_verify.py`.
- **Wasser-Escape-Reflex (implementiert):** `candidates` prüft `gs.in_water and not gs.on_ground` als
  Allererstes (noch vor den M2d-Reaktionen) und gibt dann **nur** `["escape_water"]` zurück — hier
  wählt Laya nicht, der Zustand ist ein Rescue. `resolve("escape_water")` navigiert **zuerst zum
  nächsten trockenen, standbaren Boden** (`targets.nearest_ground`, jeder Ebene — die trockene Zelle,
  die ein schwimmender Körper wirklich erreicht, darf *unter* der Wasserlinie liegen) mit
  `dig`/`scaffold`; die Homebase ist **nur der Notnagel**, wenn kein Boden in Scan-Reichweite liegt.
  Grund: ohne Grund unter den Füßen bricht der gemeinsame Pathfinder — ein `collect_block` lief live
  370 Blöcke weit und netto 0,18 m — während `navigate` aus der Schwimmhaltung heraus weiter
  funktioniert. (**Live run8:** der Bot ertrank zehnmal und mehr, weil der Reflex stur zur
  unterirdischen Basis navigierte — 70 m senkrecht hinab, 300 m entfernt — statt zur nahen Küste; die
  Basis-zuerst-Regel war der Fehler, nicht das Modell.) Steht der Bot am Grund (Wassergraben, Teich,
  ein Guss, der ihn nicht flutet), greift der Reflex **nicht**. Getestet in
  `tests/test_actions.py::EscapeWaterTest`.
  **Eingeklemmt (`repeated_movement_failure`):** Der Server meldet diesen Stop, wenn der Körper auf
  einer erneut ausgegebenen Route **nicht** vom Fleck kommt — der Bot ist körperlich eingeschlossen.
  Live run10: in einem 1 Block weiten, gefluteten Schacht antworteten `navigate`, `collect_block` und
  `build_structure` alle so, und die erzwungene `escape_water`-Rettung lief **endlos** darauf. Der
  Code gilt jetzt als `blocked_by_surroundings`, sodass die vorhandene Unblock-Leiter greift — Tasche
  graben (`make_room`), sonst auf freie Fläche (`explore`) — statt einer Endlosschleife ohne
  Laya-Wahl. (Der Server lehnte im selben Fall Graben/Bauen zusätzlich mit `Underwater stance
  unavailable` ab, solange der Kopf im Wasser stand — ein serverseitiger Guard, `src/navigation/
  diving.ts`; nur wenn der Kopf in Luft reicht, ist Arbeit erlaubt.) Getestet in
  `tests/test_graph.py::BlockedBySurroundingsTest`.
- **Homebase (implementiert):** Ein Anker auf der Oberfläche trägt Bett, Portalrahmen und Truhe.
  **Erst hochsteigen, dann ankern (`surface_up`):** Die einzige verlässliche Aussage „hier ist
  Oberfläche" ist die **Tiefe unter der Oberfläche** — nicht der Block über dem Kopf. Ein von der
  Bot gegrabener Tunnel hinterlässt `air`, nicht `cave_air`, und ist damit von offenem Himmel durch
  *keinen* `view_blocks`-Wert zu unterscheiden (`_OPEN_SKY` wie `above == "air"` messen nur „es gibt
  Halt", nicht „es ist Himmel"). **Live run9:** die Basis landete auf `y = -16` unter einer
  `y = 73`-Oberfläche, 89 Blöcke tief, weil der Bot zum Diamanten-Graben hinabgestiegen war und der
  Scan dort einen Höhlenboden fand. Deshalb ist `establish_base` an **`_at_surface`** gebunden: nur
  wenn der Bot **in derselben Spalte** wie die gelernte Oberfläche steht **und** `|gs.pos[1] −
  gs.surface[1]| <= BASE_SURFACE_TOL` (2) gilt, wird angelegt. Die Spaltenprüfung trägt: die Zelle
  wurde für *eine* Spalte gelernt, ein Bot, der sich unter eine *andere* Spalte gräbt, darf nicht
  allein wegen ähnlicher Höhe als „an der Oberfläche" gelten. **Live run10:** Füße auf einem
  `y = 54`-Höhlenboden lagen innerhalb der alten Toleranz 8 einer in anderer Spalte gelernten
  `y = 59`-Oberfläche, `establish_base` feuerte unterirdisch, und der Weg dorthin keilte den Bot in
  einem gefluteten Schacht ein. Die Toleranz 2 fängt nur noch die einzelne Platte oder eine
  einblockige Mulde an der *eigenen* Oberfläche auf. Solange die
  Oberfläche unbekannt ist oder der Bot zu tief steht, bieten die Kandidaten **`surface_up`** an:
  ein `navigate` **ohne `y`** fragt den Server nach der Oberfläche der Spalte — er läuft bei
  eindeutiger Spalte selbst nach oben und stoppt sonst mit `[NAVIGATION_HEIGHT_REQUIRED] …
  highest block … at y=N` (`perception.parse_surface_height` nimmt die höchste der gelisteten
  Standhöhen). `Agent._note_surface` liest die Antwort in `self.surface = (x, feet_y, z)`; der
  nächste Tick steigt mit `dig`/`scaffold` zu genau dieser Zelle auf (live gemessen: von `y = -15`
  nach `y = 74` in ~92 s, 12 Cobblestone Gerüst). Wechselt die Spalte, wird neu geprüft (eine andere
  Spalte kann zig Blöcke höher/tiefer liegen); ein gescheiterter *Aufstieg* lässt die Zelle stehen
  und wird einfach wiederholt. `Agent._scan_build_site` scannt ebenfalls nur noch bei `_at_surface`,
  sonst schlüge er den Höhlenboden selbst vor. Live: `fill_bucket`↔`go_home` pendelte 45
  Iterationen, weil der Eimer auch nach dem Obsidian-Banken gefüllt werden sollte (Fix: die
  Eimer-Kette ist an `obsidian < PORTAL_OBSIDIAN and not has_fs` gebunden, wie `mine_obsidian`).
  `establish_base` läuft dann auf einen **ebenen, freien** Platz: zwei
  `view_blocks box`-Reads um den Bot — seine Ebene und vier darüber — (`Agent._scan_build_site` →
  `perception.scan_ground` → `targets.flat_build_site`) verlangen eine durchgehende Bodenreihe auf
  gleicher Höhe mit **offener Himmelsluft** (`_OPEN_SKY`) darüber **und festem Grund eine Ebene
  darunter**: der Server gräbt die
  Bodenreihe aus und setzt sie als Obsidian neu, eine Zelle über einem Loch bliebe also „ohne
  Anlegefläche" falsch stehen und die Öffnung schlösse nie (live hing `light_portal` so in
  `[PORTAL_NO_FRAME]`). Damit passt der Rahmen aufs Gelände (live lief der alte Pick
  — die *niedrigste* standbare Zelle — in eine Mulde, und 15 von 20 Rahmenzellen steckten in Stein). Ist kein Platz in
  Scan-Reichweite, greift wie zuvor `targets.base_target` (nur `GROUND_NAMES`, auf/oberhalb der
  Bot-Ebene, nächste zuerst). `_REPLACEABLE` (inkl. `cave_air`) bleibt für die
  Öffnungs-Prüfung `_clear`, wo eingeschlossene Luft korrekt ist. `establish_base` legt den Anker
  als `GameState.base` fest; `go_home`
  läuft nur zurück, wenn wirklich Basisarbeit wartet (`_base_steps_ready`). `build_portal`/`place_bed`
  sind auf `_at_base` beschränkt, `_portal_site` ankert am Base-Block statt an den Füßen.
  **Anker-Reparatur (`Agent._note_base`):** Der Anker wird einmal bei `establish_base` aus dem
  Navigationsziel gesetzt (als Blockkoordinate: `args.y − 1`); nichts prüft ihn danach noch gegen die
  lebende Welt. Gräbt der Bot seinen eigenen Basisboden weg (oder sprengt ein Creeper ihn, fällt Sand
  herab), antwortet jedes `go_home` auf das nun haltlose Ziel `base.y + 1` mit
  `[NAVIGATION_TARGET_UNSUPPORTED] No usable footing … The ground in that column is at y=N.` — **Live
  run11:** nach `establish_base` hat der Bot seine Basis zu offener Luft abgebaut, und die nächsten
  **230** `go_home`-Ticks antworteten dieselbe Verweigerung, ohne dass der Lauf vorankam (Laya wählte
  `go_home` immer wieder, weil es einen getragenen Chest loswerden wollte und `_base_steps_ready`
  dafür True blieb). Die Verweigerung nennt aber die Höhe, die der Server *tatsächlich* fand, also
  wird der Anker einfach darauf umgepinnt — dieselbe „die Höhe aus der Server-Antwort zurücklesen"-
  Bewegung wie `_note_surface` (`perception.parse_ground_height`). Bewegt die Antwort den Anker nicht
  mehr (dieselbe Höhe, die er schon trägt), ist der Platz nicht in-place reparierbar, und die Basis
  wird **verworfen** (`base`, `build_site`, `_build_site_scanned` = `None`), damit der Loop an einem
  frischen `flat_build_site` neu ankert. Beide Pfade sind beschränkt: eine Tick zum Heilen, einer zum
  Zurücksetzen, nie der unendliche Retry. `tests/test_graph.py::BaseAnchorRepairTest`,
  `tests/test_perception.py::ParseGroundHeightTest`.
  **Öffnung vor dem Bau freiräumen:** Der Scan sichert die Lichtung nur für die Basis, die *er*
  gewählt hat. Eine übernommene Basis (`_adopt_homebase`) oder eine per `base_target` gewählte hatte
  diese Prüfung nie. Darum liest der Loop während des Bau-Fensters (Flint&Steel getragen, Portal
  unlit) die sechs Innenzellen der Öffnung (`perception.scan_frame_volume` → `parse_cells` →
  `perception.portal_interior`); liegt dort ein nicht-replacebarer Block, steht in den Kandidaten
  `clear_portal` statt `build_portal` und gräbt ihn weg (`build_structure` mit `block_name: "air"`,
  höchstens vier Zellen je Aufruf — der Server startet je Zelle eine Routensuche). Ein Block über
  der Tragschicht würde sonst umgebaut und `light_portal` antwortete für immer `[PORTAL_NO_FRAME]`
  (live warf der generische Nachtschlaf ein Bett genau dorthin).
  **Schlafen setzt den Respawn:** `place_bed` (via `sleep`)
  wird **nur nachts** angeboten — bei Tag würde das Bett nur platziert, der Pausepunkt bliebe unset.
  `_bed_available` zählt auch ein **platziertes** Bett (nicht nur ein getragenes): nach einem
  manuellen Basisbau oder einem Neustart bleibt das Bett sonst als Block zurück und der Loop würde
  ohne Not Wolle für ein zweites suchen.
  **Bett erst nach dem Portal:** Vor dem angezündeten Portal wird der generische `sleep` an der
  Basis **unterdrückt** (`_at_base` und *nicht* `_portal_lit`). Der Server legt das Bett „neben den
  Bot", und an der Basis ist das genau das Portal-Innere — `build_portal` kann ein Bett dort nicht
  wegräumen (`remove_wrong_blocks` antwortet „movement policy prohibits breaking this block"). Live
  landete das Bett in der Öffnung und `light_portal` antwortete in jedem Tick mit
  `[PORTAL_NO_FRAME]`. Das Basisbett ist Sache von `place_bed` und wartet auf das *angezündete*
  Portal; abseits der Basis bleibt der Nachtschlaf unverändert.
  Eine **Truhe** (8 Planken, 3×3) hält Überschuss: `craft_chest` → `place_chest` (Koordinaten werden
  gemerkt) → `stow_surplus` (`use_container deposit`) bei ≥33/36 Slots; `drop_junk` ist nur der
  Notnagel ohne Truhe. Ein `[INVENTORY_FULL]`-Refusal löst eine eigene Relief-Stufe aus
  (`blocked_by_inventory` → `inventory_relief`: Truhe, sonst werfen; eine volle Truhe schaltet
  `allow_stow` ab). Basis/Truhe sind **pro Lauf** und werden vom Loop in jeden `GameState` injiziert.
  Kann die Truhe am Platz nicht stehen — der Server antwortet `[PLACE_NO_SUPPORT]`, wenn kein
  fester Block an die Zelle grenzt (live, Nether-Lauf: die Basis lag auf einer schmalen Kante und
  `place_chest` lief in jedem Tick auf, statt das Portal zu bauen) —, so merkt sich
  `Agent._note_chest` das und lässt die ganze Truhen-Kette fallen (`gs.chest_blocked`); das Relief
  greift dann auf `drop_junk` zurück, das dieselben Slots freimacht und nicht scheitern kann.
  **Belegte Zielzelle (`[PLACE_TARGET_OCCUPIED]`, live im From-Zero-Lauf):** enthielt die Truhen-
  Zelle einen Fremdblock (Gras), verweigerte der Server den Bau. `_note_place` schreibt die
  verweigerte Zelle nach `gs.blocked_cell`, und der Unblock wählt genau diese Zelle: `make_room`
  gräbt sie frei, danach geht `place_chest` durch (it10 → it11 → it12). Vorher lief der Lauf in
  einen Endlos-Retry auf dieselbe belegte Zelle.
  `Agent._adopt_homebase` übernimmt beim Tick-Start ein bereits **stehendes** Bett in Reichweite
  (`ADOPT_BED_RADIUS` = 12 m) als Anker und eine **Truhe** in Ankernähe (`ADOPT_CHEST_RADIUS` = 4)
  als `chest_pos` — so läuft ein Neustart an derselben Homebase weiter, statt eine zweite zu bauen
  (die Truhe steht dafür in `perception.CHAIN_NAMES` im `view_blocks`-Scan).
  **Anker aus dem stehenden Portal (live, Fortsetzen von run7c):** Das Bett-Overnehmen hat einen
  blinden Fleck — das Bett kann weit weg vom Portal stehen (hier: unterirdisch bei y=28), und `base`
  ist pro Lauf nur im Speicher. Nach einem Neustart stand der Bot 76 m vom eigenen, bereits
  leuchtenden Portal, außerhalb der 64-m-Ressourcenreichweite, sodass `_portal_lit` *falsch* war:
  `establish_base` hätte eine zweite Basis (und damit ein zweites Portal) gebaut — was nicht mehr
  geht, denn mit `flint_and_steel` im Gepäck ist `mine_obsidian` gesperrt und der Obsidian ist
  verbraucht. `Agent._adopt_portal_base` liest darum — solange `base` noch leer ist und das Kit
  steht — einmalig ein `nether_portal` **jenseits** von `RESOURCE_RADIUS`
  (`perception.scan_portal_corner`/`parse_portal_corner`, bewusst *ohne* Radiusfilter) und setzt
  `base = Ecke − (2, 1, 0)`: Die Öffnungsecke invertiert `_portal_site`, sodass die Rahmenebene
  wieder auf dem stehenden Obsidian liegt. `gs.base_adopted` fährt in den Zustand und
  `_base_steps_ready` antwortet damit `True`, sodass `go_home` den Bot zurückführt; bei Ankunft
  (`_at_base`) wird das Flag gelöscht. `_scan_portal_opening` erkennt ein bereits leuchtendes Portal
  und legt dessen Blöcke nicht als „Fremdblock in der Öffnung" aus.
  **Ofen an der Basis (live, Rückmeldung aus dem Lauf):** `smelt_item` mit `temporary_workstation`
  stellt den getragenen Ofen *für den einen Aufruf* auf und sammelt ihn danach wieder ein — beim
  Braten in der Lull (3 Stück pro Aufruf) riss der Bot den Ofen so zwischen den Chargen immer
  wieder ab. Darum stellt `place_furnace` den getragenen Ofen einmal an der Basis ab (Zelle
  `(bx, by+1, bz-1)`, neben der Truhe und frei von der Portal-Rahmenebene `z == bz`), merkt die
  Koordinaten als `gs.furnace_pos` (`Agent._verify` notiert sie, `_adopt_homebase` übernimmt beim
  Neustart einen Ofen in Ankernähe aus dem `furnace`-Scan in `CHAIN_NAMES`). `cook_food`,
  `smelt_gold` und `smelt_iron` geben `smelt_item` dann diese Koordinaten und *ohne*
  `temporary_workstation`, sodass derselbe Ofen stehen bleibt. Der Schritt läuft nur in der Lull
  (`_nether_idle`) — vor dem Portal braucht die Kette den getragenen Ofen noch zum Schmieden
  abseits der Basis —, und `_furnace_site` verweigert die Zelle, wenn dort bereits ein Bett steht
  (der `[PLACE_TARGET_OCCUPIED]`-Unblock würde sonst das Respawn-Bett weggraben); kann der Boden
  den Ofen nicht tragen (`[PLACE_NO_SUPPORT]`), fällt die Kette weg (`Agent._note_furnace` →
  `gs.furnace_blocked`) und das Braten nutzt wieder den temporären Aufbau.
  Getestet in `tests/test_actions.py::HomeBaseTest`, `tests/test_actions.py::BaseFurnaceTest`,
  `tests/test_targets.py::BaseTargetTest`,
  `tests/test_graph.py::GuardUnblockTest`, `tests/test_graph.py::PortalBaseAdoptTest`,
  `tests/test_graph.py::FurnaceSupportTest`,
  `tests/test_perception.py::PortalCornerParseTest` und `tests/test_verify.py::HomeBaseProgressTest`.
