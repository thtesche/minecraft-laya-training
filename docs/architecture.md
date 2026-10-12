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
    ACTION_BUSY / RESULT_NOT_RETRIEVED-Recovery. Reads sind ungated.
[2] Perception    (src/mlaya/perception.py)
    view_status + read_recent_events + view_blocks (alle frei) -> GameState
    -> kompakter NL-State (`narrative()`); Ressourcen (logs/stone/iron_ore/coal_ore)
    mit Distanz im Umkreis von 64 Blöcken; Tageszeit/Schlafzustand, Mob-Arten
    (inkl. passiver) und ungelesene Notifications (Drain) fließen in den State.
[3] Brain (S1)    (src/mlaya/brain.py)
    Laya-Router, EIN predict()-Pass:
      in_danger  (binäres Gate)      -> defer an die Survival-Policy
      danger     (score, nur Logging)
      intent     (choice)            -> unter den gefilterten Kandidaten
[4] Actions       (src/mlaya/actions.py)
    candidates(gs): gültige Labels nach Vorbedingungen (Tech-Tree-Reihenfolge)
    resolve(label, gs): Label -> echtes MCP-Tool + Argumente
    Kaskade: get_wood -> craft_pickaxe -> craft_table -> mine_stone ->
    craft_stone_pickaxe -> craft_furnace -> mine_iron -> mine_coal ->
    smelt_iron -> craft_iron_pickaxe; Bett-Kette (hunt_wool/hunt_string ->
    craft_bed -> sleep in der Nacht) und eat_food bei Hunger; explore nur als
    letzter Fallback (wenn nichts Produktives möglich ist).
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
                        |              |                       +-> END (max_iterations)
                        |              +-> defer ---------------+-> decide (loop)
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
über Steinwerkzeuge und Ofen bis zur **Eisen-Spitzhacke** führen.

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

- 3×3-Rezepte (Stein-Spitzhacke, Ofen, Eisen-Spitzhacke) laufen über
  `temporary_workstation: true` und brauchen einen **getragenen** `crafting_table`
  (`craft_table` vorher); Schmelzen braucht einen getragenen `furnace`.
- `mine_iron`/`mine_coal` werden nur angeboten, wenn im Umkreis tatsächlich Erz geladen ist
  (sonst fällt die Kaskade auf `explore` zurück).
- `explore` ist bewusst der **letzte Fallback** und nur dann in `candidates`, wenn kein
  produktiver Schritt möglich ist — sonst verwässert es die (zero-shot verrauschte) Wahl.

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
    `hostile_avoidance_multiplier: 3`, das vorsichtige Combat-Fenster
    (`engagement: respond_to_threats`, `engage_min_health 17`, `critical_health 14`,
    `recover_to_health 18`, `retreat: true`, `enderman_wait_ticks 40`, Timeouts) und
    `food.raw.allow: "always"` mit `hunger_at_most: 20`.
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
  Nether-Ziel 24) — nächstes passives Tier (`FOOD_MOBS = cow/pig/chicken/sheep/rabbit`) **innerhalb
  einer tageslichtabhängigen Reichweite** (`FOOD_HUNT_RANGE_DAY = 64` / `FOOD_HUNT_RANGE_NIGHT = 16`:
  nachts nur, was schon nah ist) oder, als Rückfall, ein **einzelner Zombie** (`rotten_flesh`,
  gegated: ≤2 Hostiles ≤16 m und alle Zombies, Health ≥18). `cook_food` brät rohes Fleisch
  (`smelt_item` im getragenen Ofen mit Kohle). `hunt_food` wird angeboten, solange der Vorrat < 16 ist
  **und** nicht gerade gegessen wird („feed to 20 first, then bank"); die Aktion ist mit
  `allow_without_shield: true` markiert. Gegessen wird über den Server-Reflex (roh erlaubt), `eat_food`
  bleibt der manuelle Nachschlag.
- **Gefahr (live, jetzt entschärft):** `hunt_food` wählte früher das **nächstgelegene** Tier
  unabhängig von der Distanz, und `collect_mob_drop` roamt dorthin. Ein nächtlicher `run` wanderte so
  **159–190 m** und wurde **zweimal getötet** (u. a. „slain by Enderman"), was **das komplette
  Inventar** kostete — inklusive der gerade gecrafteten Eisenhacke. Auch `hunt_string` scheiterte
  dreimal, einmal tödlich. Der `in_danger`-Gate sagte in diesen Ticks **`no`** (Health 10.5, Skelett
  9.6 m) — das Zero-Shot-Gate ist auf Gefahr schlecht kalibriert (bekannter Befund, s. o.). Die
  Server-Policy kämpft erst ab Health ≥ 17 (`respond_to_threats`), greift unterhalb also nicht ein.
  **Fix:** `hunt_food` ist jetzt an eine **Tageslicht-/Distanz-Schranke** gebunden; nachts wird nur
  gejagt, was schon in Reichweite ist. Live gegengeprüft (Nacht-Tick, nächste Tiere 35/38/75/103 m):
  `_food_target = None`, `hunt_food` wird gar nicht erst angeboten. Rest-Risiko: tagsüber bleiben
  längere Wege bis 64 m möglich — bewusst, weil der Tagesweg billig ist.

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
  - **`replace_tool`** — aktive Spitzhacke ≤ `LOW_DURABILITY = 0.15` und ein Ersatz ist craftbar
    (`_replacement_pickaxe`: Eisen bei ≥3 Ingots, sonst Stein; nie ein Downgrade) →
    `craft_item` mit `temporary_workstation`.
  - **`collect_drops`** — frisches „target died"-Event (`gs.drops > 0`, Alter ≤
    `EVENT_REACT_WINDOW_S = 120` s) → `pick_up_items {radius: 16}`.
  - **`recover_drops`** — Tod ≤ `RECOVER_WINDOW_S = 90` s, gleiche Dimension, ≤ `RECOVER_RANGE =
    120` m → `pick_up_items {recover_death_items: true}` (der Server läuft selbst zur
    Todesposition zurück). Nach jedem Tod setzt der Graph die Policy neu (`_reapply_after_death`).
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
  Zustand: eine geloggte Laya-Antwort hat Vorrang, sonst greift eine dokumentierte Heuristik
  („ein Mob greift an oder Gesundheit sehr niedrig", `DANGER_CLOSE_M = 4`, `DANGER_LOW_HEALTH = 8`).
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
- **Validierung:** 22 Ticks → 12 Datensatzzeilen (13 Multi-Kandidat, Dedup); `--dry-run` 9 Zeilen
  → 18 Items; echtes Kopf-Fine-Tuning (4 Epochen, `rlcd`, Encoder eingefroren) liefert ein
  842 MB Checkpoint. Held-out (6 Items): `choice`-Accuracy 0.833 → 0.833, Loss −0.149, Brier
  −0.065, ECE +0.047 (Temperatur nicht gefittet, nur 1 Kalibrierungsitem). Der `Router` lädt den
  Checkpoint unter `english` und entscheidet korrekt (Gate `no` → nicht deferiert, Intent
  `mine_stone`). 16 Unit-Tests (`tests/test_dataset.py`, stdlib `unittest`).
- **Vorbehalt 1 (Datenmenge):** 22 Ticks sind **statistisch bedeutungslos** — der Gewinn ist der
  **Pipeline-Beweis**, nicht ein messbarer Modellgewinn. Neu bauen (`mlaya dataset`) und
  nachtrainieren, sobald der Log wächst.
- **Vorbehalt 2 (Gate-Balance):** In diesen Läufen war der Bot **nie** in Gefahr, also tragen
  **alle** `in_danger`-Zeilen das Label `no`. Ein Retrain darauf zieht den geteilten `choice`-Kopf
  in Richtung „immer no", ohne dass ein `yes`-Fall gegengewichtet. Abhilfe: genug Läufe mit echten
  Gefahren-Ticks sammeln (dann liefert `graph._verify` die echten Gate-Labels) oder vorerst
  `--no-danger` und das Gate zero-shot belassen, bis beide Klassen vertreten sind.

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

- **M3+ (offen):** Datensatz aus deutlich mehr Läufen bauen und gegen den Zero-Shot-Stand
  evaluieren (Accuracy/Loss auf einem größeren Held-out-Split), um einen echten Gewinn zu zeigen.
- **M2/M2b/M2c/M2d live validiert:** `replace_tool`, `hunt_food`, `collect_drops`
  (Entscheidung/Binding), `recover_drops` (s. M2c/M2d) sowie `craft_table`, `hunt_wool`, `craft_bed`,
  `sleep` (s. M2b) sind **live durchgelaufen**. Damit bleibt in M2/M2b/M2c/M2d kein Pfad ohne
  Live-Lauf; die Bett-/Essensketten sind zusätzlich durch die 22 Determinismus-Tests gepinnt.
  Offen ist die **Nether-/Diamant-Kette** (Obsidian/Feuerstein, Diamant-Werkzeuge) — die ist noch
  nicht in `actions.py` abgebildet und wäre M2.
