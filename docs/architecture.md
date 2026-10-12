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
    -> kompakter NL-State (`narrative()`); Ressourcen (logs/stone/iron_ore/coal_ore/
    diamond_ore) mit Distanz im Umkreis von 64 Blöcken (Ressourcen- und Boden-Scan sind
    zwei `view_blocks`-Gruppen, weil `find` maximal 8 Blocknamen erlaubt); Tageszeit/
    Schlafzustand, Mob-Arten (inkl. passiver) und ungelesene Notifications (Drain)
    fließen in den State.
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
    smelt_iron -> craft_iron_pickaxe -> mine_diamond -> craft_diamond_pickaxe
    (-> craft_diamond_sword) -> craft_diamond_<slot> (Brustplatte/Hose/Schuhe/Helm)
    -> equip_armor; Bett-Kette (hunt_wool/hunt_string ->
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
  Tier ≥ Eisen. Die Diamant-Ziele kommen aus **einer** Quelle, `_next_diamond_item(gs)`
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
  `verify.diff`/`progressed` verbuchen `offhand_delta`. **Eisen-Verbrauch:** Weil sonst nichts mehr
  Eisen brauchte (Diamant-Spitze steht), hätte die Schleife nie wieder Eisen geschürft und das
  Schild nie erreicht (live: `candidates` war nur `['explore']` neben `iron_ore at 4m`).
  `_iron_goal` ist jetzt die einzige Quelle: 3 Barren für die (Ersatz-)Eisenhacke, danach **1** für
  den Schild, **0** sobald beides steht. `candidates` **und** `resolve` teilen sie; der Craft ist
  zudem hinter die Eisen-Spitzhacke gated, damit er die ersten drei Barren nicht stiehlt.
- `explore` ist bewusst der **letzte Fallback** und nur dann in `candidates`, wenn kein
  produktiver Schritt möglich ist — sonst verwässert es die (zero-shot verrauschte) Wahl.
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
- **Nether-Kette (offen):** Eimer/Obsidian/Flint&Steel und Rahmen bauen/anzünden/betreten
  (`activate_portal` zündet nur einen bestehenden Rahmen, baut nichts). Per Entscheidung zunächst
  zurückgestellt.
