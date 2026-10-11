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
- **Sicherheits-Gate für Hostile-Jagd:** Ein Spinnen-Lauf hat den Bot in ein
  Zombie/Skelett/Creeper-Feld geführt und eine Eisen-Spitzhacke gekostet. `hunt_string` ist
  daher nur erlaubt, wenn **genau eine** Spinne ≤10 m, **kein weiterer Hostile ≤16 m** und
  Health ≥16 ist (`_safe_spider_hunt`); sonst bleibt es außen vor.
- **Notifications:** `read_recent_events` wird jetzt mit `limit: 50` gelesen (Drain des
  Event-/Notification-Backlogs) statt 5; die neuesten `events`-Summaries **und** die
  `notifications.recentPreview` stehen im State (`narrative()`: „unread notifications (N): …").

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
  Nether-Ziel 24) — nächstes passives Tier (`FOOD_MOBS = cow/pig/chicken/sheep/rabbit`) oder, als
  Rückfall, ein **einzelner Zombie** (`rotten_flesh`, gegated: ≤2 Hostiles ≤16 m und alle Zombies,
  Health ≥18). `cook_food` brät rohes Fleisch (`smelt_item` im getragenen Ofen mit Kohle).
  `hunt_food` wird angeboten, solange der Vorrat < 16 ist **und** nicht gerade gegessen wird
  („feed to 20 first, then bank"); die Aktion ist mit `allow_without_shield: true` markiert.
  Gegessen wird über den Server-Reflex (roh erlaubt), `eat_food` bleibt der manuelle Nachschlag.

## Nächste Meilensteine

- **M2d**: Auf Notification-/Event-**Typen reagieren** (z. B. „durability low" → Ersatzwerkzeug
  craften, „target died" → Ressourcen ernten) statt sie nur in den State zu heben.
- **M3**: Logs -> Datensatz -> RLCD-Kopf-Fine-Tuning; Checkpoint via `LAYA_EXTRA_MODELS` registrieren.
- **M4**: Parameter-Layer ausbauen (Zielwahl über `view_blocks`-Kandidaten, `predict_shortlist`).
- **Offen aus M2**: Nether-Gate (Obsidian/Feuerstein) und Diamant-Kette.
