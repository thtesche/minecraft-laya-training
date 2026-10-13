# minecraft-laya-training

Ein Minecraft-Agent, dessen **System-1-Entscheidungen von
[Laya](https://github.com/NandhaKishorM/laya)** getroffen werden — einer nicht-autoregressiven
Typed-Decision-Engine (`choice` / `score` / `noul`) — und der über den
**mine-ai MCP-Server** handelt. Ein generatives LLM (OpenRouter) dient als **System 2**
für seltene Eskalationen.

Laya erzeugt keinen Text und keine Tool-Argumente. Deshalb filtert der Code die aktuell
**gültigen Optionen** (Vorbedingungen/Tech-Tree), und Laya entscheidet in **einem** Forward-Pass
über ein binäres Gefahr-Gate plus die Wahl unter diesen Optionen; ein separater Python-**Executor**
ruft die MCP-Tools auf und respektiert das Foreground-Gating des Servers.

Aktueller Stand (**M2/M2b/M2c/M2d/M3/M4 + Homebase/Nether**): Die Progression läuft von Holz über einen Werktisch und Steinwerkzeuge
bis zur **Eisen-Spitzhacke** (Ofen + Schmelzen) und weiter über `mine_diamond` zur **Diamant-Spitzhacke**
(+ Diamant-Schwert und **kompletter Diamant-Rüstung**: Helm/Brustplatte/Hose/Schuhe). `view_blocks`
liefert die Ressourcen in der Nähe
(Holz/Stein/Eisen/Kohle/Diamant mit Distanz) in den Zustand. Auf der Oberfläche legt der Agent eine
**Homebase** an (Anker für Bett, Portal und Truhe): Er baut ein **Bett** (aus Schafwolle oder
Spinnenfäden), **schläft nachts** darin (überspringt die Nacht, setzt den Respawn-Punkt), baut aus
einem Wassereimer **Obsidian**, entzündet damit ein **Nether-Portal** und hält in einer **Truhe**
Überschuss fest. Er liest die anstehenden **Notifications** mit und setzt beim Start die
**Server-Survival-Policy** (Scaffold-Blöcke, vorsichtiges Kampf-Fenster, `food.raw.allow: always`);
gegessen wird über den Server-Reflex, Essen wird über `hunt_food`/`cook_food` beschafft, das einen
**Mindestvorrat von 16** essbaren Items hält. Die Policy läuft in zwei **Stufen** (`base` für Reisen/
Progression, `hunt` für eine feindliche Jagd). Der Agent **reagiert auf Event-Typen**: eine fast
abgenutzte Spitzhacke wird ersetzt (`replace_tool`), Drops von einem vom Reflex getöteten Mob werden
eingesammelt (`collect_drops`), und frisch verlorene Ausrüstung wird an der Todesstelle geborgen
(`recover_drops`). Seit **M3** lässt sich aus dem Run-Log ein Laya-Trainingsdatensatz bauen und der
`choice`-Kopf nachtrainieren (RLCD); der Checkpoint wird über `LAYA_EXTRA_MODELS` unter `english`
registriert — ohne Codeänderung. **M4** ergänzt einen **Parameter-Layer**: wo ein Label nur sagt
*was* (z. B. `explore`), wählt Laya das *wo* aus den `view_blocks`-Kandidaten — echten Standzellen
mit `above: air` — und bindet sie an ein `navigate`-Ziel; bei vielen Kandidaten engt
`predict_shortlist` sie vorher ein.

Die M2/M2b/M2c/M2d-Pfade wurden live getestet: `replace_tool` (abgenutzte Hacke → Ersatz), `hunt_food`
(Tier gejagt, Vorrat aufgefüllt), `collect_drops` (Reflex-Kill → Drops eingesammelt),
`recover_drops` (Tod in Place → Ausrüstung zurückgeholt) sowie die **Bett-Kette** `craft_table` →
`hunt_wool` (Schaf) → `craft_bed` → nachts `sleep` (Nacht wurde auf Tag übersprungen) laufen.

Die **Eisen-Kette** ist ebenfalls live durchgelaufen (Tier 0 → 3): `craft_pickaxe` → `mine_stone` →
`craft_stone_pickaxe` → `mine_iron` → `mine_coal` → `craft_furnace` → `smelt_iron` → `craft_iron_pickaxe`.
Dabei einen Blocker gefunden und behoben: in einem 1×1-Schacht nach dem Erz-Abbau scheiterte
`craft_item` mit `[WORKSTATION_PLACEMENT_FAILED] … no_cell` (kein Platz für die getragene Werkbank) —
in der Schleife ein **Endlos-Retry**. Der `graph.Agent` merkt sich das blockierte Label und schiebt
einen **Unblock** ein: zuerst `make_room` (gräbt mit `build_structure` vier Zellen frei, direkt vor
Ort), dann erst `explore` auf offenes Gelände. Live in einem tiefen Schacht reproduziert und behoben
(`no_cell` → `make_room` → derselbe Craft geht durch). Gepinnt in `tests/test_graph.py`.
Derselbe Mechanismus deckt jetzt auch den **Bett-Platz** ab: im Rüstungslauf antwortete `sleep`
im 1×1-Schacht dreimal mit `[BED_PLACEMENT_FAILED] … no … flat ground` (drei verschwendete Ticks).
`BLOCKED_CODES` enthält daher auch `BED_PLACEMENT_FAILED` → der Unblock gräbt zuerst ein Pocket
(`make_room`), in dem das Bett zwei gestützte Zellen findet.

Die **Diamant-Kette** ist jetzt ebenfalls live gelaufen: `mine_iron`/`mine_coal`/`smelt_iron` →
`craft_iron_pickaxe` (Ersatz, nachdem die Eisen-Hacke im Tunnel zerbrochen war) → `mine_diamond` →
`craft_diamond_pickaxe` → zwei weitere Diamanten → `craft_diamond_sword` (Endstand Tier 4,
Hacke **und** Schwert aus Diamant). Dabei einen **Deadlock** gefunden und behoben: `mine_diamond` war
auf `tier < Diamant` begrenzt und durfte damit nicht mit der Diamant-Hacke selbst die zwei Diamanten
fürs Schwert abbauen — das Schwert war so unerreichbar. Die Grenze ist entfernt (Regressions-Tests in
`tests/test_actions.py::DiamondChainTest`). Zweiter Befund: die Eisen-Hacke kann *innerhalb* eines
langen `collect_block` brechen (die `replace_tool`-Reaktion prüft nur zwischen zwei Ticks). **Härtung:**
`replace_tool` reserviert jetzt zusätzlich bei ≤ `MINING_HEADROOM = 80` Rest-Nutzungen einen frischen
Ersatz derselben Stufe; bricht die aktive Hacke, schiebt der Server den Ersatz nach, die Stufe bleibt.
Ein Guard verhindert Mehrfach-Crafting desselben Ersatzes. Dazu eine explizite Erz-Regel
(`ORE_MIN_TIER`): Bruchstein/Kohle brauchen jede Hacke, Eisen eine Stein-, Diamant eine
Eisen-Spitzhacke (Gold zählt als Tier 1, zerstört also beides). `_can_mine` startet einen Grabungslauf
nur, wenn auch ein Bruch *mittendrin* die Mindeststufe nicht unterschreitet — sonst zerstört die
nachgeschobene niedrigere Stufe das restliche Erz ohne Drop. Stein und Kohle sind ausgenommen, sonst
blockiert der Bootstrap (Holzhacke müsste den Bruchstein für den Stein-Ersatz sammeln dürfen). Und die
frühere Obergrenze `tier < IRON_PICK_TIER` für `mine_iron` ist weg: eine abgenutzte Eisenhacke kann
jetzt das Eisen für ihren eigenen Ersatz nachschürfen (eine frische schürft weiterhin kein Eisen).

**Diamant-Rüstung (live validiert):** `_next_diamond_item` erweitert die
Diamant-Ziele über Werkzeugkette + Schwert hinaus um das komplette Rüstungsset
(`DIAMOND_ARMOR`): Brustplatte (8), Hose (7), Schuhe (4), Helm (5) = **24 Diamanten**,
geordnet nach **Schutz pro Diamant** (Brustplatte zuerst), damit ein Teil-Set schon so nützlich
wie möglich ist. Es wird immer nur **das nächste fehlende Teil** angeboten (`mine_diamond` mit
`count = Kosten − getragene Diamanten`, sonst `craft_diamond_<slot>`); ein `equip_armor`
(ein `equip`-Batch, ohne Ziel) rüstet gecraftete Teile an. Rüstung braucht **keine Stöcke**,
nur den Werktisch, und zählt als „vorhanden", ob **getragen** oder noch im Inventar — so wird kein
Teil doppelt gecraftet. **Live durchgelaufen** (25 Ticks): `mine_diamond`
(8/7/4/5 = exakt 24) → `craft_diamond_chestplate` → `leggings` → `boots` → `helmet`, Endstand
alle vier Teile im Inventar. **Korrektur:** der Server rüstet **nicht** selbst an — die Teile
bleiben zunächst im Inventar (`location: main`). Getragen wird nur, was ein `equip` (`equip_armor`)
in einen Rüstungsslot verschiebt. `situation.tools.armour` listet **mitgeführte** Rüstung pro Klasse
(ob an oder nicht); der **getragene** Satz kommt aus der Inventar-`location`
(`head`/`torso`/`legs`/`feet`).

**Schild:** Der Schild (6 Planken + 1 Eisenbarren, 3×3 → Werktisch) ist eine eigene kleine Tier:
`craft_shield` → `equip_shield`. Er blockt nur **in der Off-Hand**, und ein gecrafteter Schild
bleibt — wie die Rüstung — zunächst im Rucksack; erst ein `equip` mit `destination: "off-hand"`
legt ihn dorthin. `view_status` meldet den Off-Hand-Gegenstand als Inventar-Stack mit
`location: "off-hand"`; `GameState.offhand` liest das, und `verify.diff` zählt `offhand_delta`
als Fortschritt von `equip_shield`. Weil sonst nichts mehr Eisen brauchte (die Diamant-Spitzhacke
steht), hätte der Loop nie wieder Eisen geschürft — `_iron_goal` macht das Schild zum
Eisen-Grund: 3 Barren für die (Ersatz-)Eisenhacke, danach genau **1** für den Schild, und **0**,
sobald beides steht. Der Craft wartet auf die Eisen-Spitzhacke, damit das Schild nicht die ersten
Barren stiehlt.

**Homebase & Nether-Portal:** Sobald das Diamant-Kit steht, sucht `establish_base` einen ebenen,
freien Platz für den Portalrahmen: zwei `view_blocks box`-Reads um den Bot — seine eigene Ebene und
vier darüber, damit auch ebenes Gelände über einer Mulde gefunden wird — (`perception.scan_ground`
→ `targets.flat_build_site`) verlangen eine durchgehende Bodenreihe auf gleicher Höhe mit Luft
darüber, weil der Rahmen auf Hanggrund sonst in Stein stecken bleibt. Findet der Scan nichts, fällt
die Wahl auf die nächste offene Oberfläche über dem Bot (`targets.base_target`) zurück. `go_home`
läuft nur zurück, wenn dort Arbeit wartet (`_base_steps_ready`). Bett, Portalrahmen und Truhe stehen
alle an diesem Anker (`build_portal`/`place_bed` nur bei `_at_base`). Der Nether-Pfad:
`craft_bucket` (3 Eisenbarren) → `fill_bucket` → `mine_obsidian` → `get_flint` (Gravel) →
`craft_flint_and_steel` (2×2) → `build_portal` → `light_portal` → `place_bed`.
`mine_obsidian` **erzeugt** das Obsidian: der Lauf gießt den Wassereimer auf die nächste Lava
(`use_bucket pour` auf die Lava-Zelle, notfalls erst dorthin laufen) und baut genau die entstandenen
Blöcke ab. Natürlicher Obsidian liegt nur tief unter denselben Lavaseen (`y ≈ -55`) und wird bewusst
nicht angegraben — ein Tiefbohrversuch endete in der **Wasser-Falle** (schwimmend ohne Grund, jedes
folgende `collect_block` lief leer). Dafür sorgt jetzt der **Wasser-Escape-Reflex**: `in_water and
not on_ground` macht `escape_water` zum einzigen Kandidaten und navigiert zur Basis bzw. auf den
nächsten trockenen Boden zurück. Schlüssel der Kette bleibt der **Obsidian-Latch**: Obsidian muss vor
dem Flint&Steel gebankt sein, damit `mine_obsidian` nach dem Bau (der die 10 Blöcke ausgibt) nicht
erneut gräbt. Das Bett wird **nur nachts** gelegt — nur Schlafen setzt den Respawn-Punkt. An der
Basis wird der generische Nachtschlaf vor dem angezündeten Portal **unterdrückt**: der Server legt
das Bett „neben den Bot", und dort wäre das das Portal-Innere; der Rahmenbau kann ein Bett nicht
entfernen, also blieb live `light_portal` in jedem Tick bei `[PORTAL_NO_FRAME]` hängen. Das Basisbett
legt deshalb `place_bed`, nachdem das Portal brennt.
Für Überschuss sorgt eine **Truhe** (`craft_chest` aus 8 Planken → `place_chest` → `stow_surplus`) ab
33/36 Slots; `drop_junk` wirft nur als **Notnagel**, wenn keine Truhe da ist oder sie voll ist. Ein
`[INVENTORY_FULL]`-Refusal löst eine eigene Relief-Stufe aus (`inventory_relief`). Base und Truhe
gelten **pro Lauf** und werden vom Loop in jeden `GameState` injiziert; steht beim Start schon ein
Bett (und eine Truhe) in Reichweite, werden sie übernommen, damit ein Neustart an derselben Homebase
weiterläuft statt eine zweite zu bauen. Kann die Truhe am gewählten Platz nicht stehen — der
Server antwortet `[PLACE_NO_SUPPORT]`, wenn kein fester Block an die Zelle grenzt (live stand die
Basis auf einer schmalen Kante, und der Lauf hing in jedem Tick auf `place_chest`, statt das Portal
zu bauen) —, lässt der Loop die Truhen-Kette fallen: das Relief wirft den Überschuss dann einfach
auf den Boden (`drop_junk`), was dieselben Slots freimacht und nicht scheitern kann.

**Entschärft:** `hunt_food` war ungeschützt zur nächsten Kreatur geroamt (ein Nachtlauf wanderte
159–190 m und wurde zweimal getötet, Totalverlust des Inventars). Jetzt ist es an eine
**Tageslicht-Schranke** gebunden (`FOOD_HUNT_RANGE_DAY = 64` / `FOOD_HUNT_RANGE_NIGHT = 16`): nachts
nur Tiere in Reichweite, tagsüber auch weiter entfernte. Weiterhin gilt: nachts hält der
Server-`hostile_reflex` zeitweise den Körper („body owner: …"), und Mobs in der Liste (≤ ~150 m)
müssen erst **geladen** sein, um jagbar zu sein (`[HUNT_TARGET_NOT_LOADED]`).
Details + 22 Determinismus-Tests: [`docs/architecture.md`](docs/architecture.md#m2d--erreicht-auf-event-notification-typen-reagieren).

Architektur + Zero-Shot-Befunde: [`docs/architecture.md`](docs/architecture.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env        # OPENROUTER_API_KEY eintragen (nur für System-2-Eskalation)
```

`.env` wird bei jedem CLI-Lauf automatisch geladen (echte Umgebungsvariablen haben Vorrang)
und ist gitignoriert. Der Status lässt sich mit `probe` prüfen.

Voraussetzung: Der mine-ai MCP-Server läuft unter `http://localhost:25575/mcp`.

## Nutzung

```bash
.venv/bin/python -m mlaya.cli probe      # health + tools/list
.venv/bin/python -m mlaya.cli decide     # Zustand lesen + einmal entscheiden (dry run)
.venv/bin/python -m mlaya.cli policy     # Server-Survival-Policy anzeigen
.venv/bin/python -m mlaya.cli policy --apply              # Stufe `base` setzen
.venv/bin/python -m mlaya.cli policy --apply --preset hunt  # Stufe `hunt` setzen
.venv/bin/python -m mlaya.cli run -n 10  # Entscheidungs-Loop (führt echte Aktionen aus)
.venv/bin/python -m mlaya.cli run --ignore-goal  # über das Tech-Tree-Ziel hinaus weiterlaufen
```

Konfiguration: [`config/settings.toml`](config/settings.toml).
Jede Entscheidung wird nach `runs/decisions.jsonl` geloggt.

Der `run`-Loop **stoppt von selbst**, sobald das Tech-Tree-Ziel steht: Diamant-Spitzhacke +
Diamantschwert, alle vier Diamant-Rüstungsteile *getragen*, der Schild in der Off-Hand, **das
Nether-Portal angezündet und ein Bett an der Homebase platziert**
(`actions.tech_tree_complete`). Er meldet dann `"stop": "goal"`; ohne Ziel endet er am
Iterationslimit mit `"stop": "limit"`. Ausschalten (z. B. für die Datensatz-Sammlung über das
Ziel hinaus) geht über `[loop] stop_when_complete = false` oder `run --ignore-goal`.

## M3: Logs → Datensatz → Fine-Tuning

```bash
# 1) Aus dem Run-Log einen Laya-Trainingsdatensatz bauen (train + held-out Eval)
.venv/bin/python -m mlaya.cli dataset            # -> data/train.jsonl + data/eval.jsonl
.venv/bin/python -m mlaya.cli dataset --include-failed   # auch Ticks ohne Fortschritt behalten
.venv/bin/python -m mlaya.cli dataset --no-danger        # ohne `in_danger`-Zeilen (s. u.)

# 2) Datensatz-Format prüfen (lädt keine Gewichte), dann Kopf nachtrainieren (RLCD)
.venv/bin/python -m mlaya.cli train --dry-run
.venv/bin/python -m mlaya.cli train              # -> runs/checkpoints/minecraft-<zeit>

# 3) Checkpoint unter `english` registrieren und den Loop damit fahren
export LAYA_EXTRA_MODELS='{"english": "'"$PWD"'/runs/checkpoints/minecraft-<zeit>"}'
.venv/bin/python -m mlaya.cli run -n 10
```

`LAYA_EXTRA_MODELS` ist ein JSON-Objekt `{name: quelle}`; ein Name, der ein eingebautes Checkpoint
trifft (`english`), **zeigt es um** — genau so wird das Fine-Tune ohne Codeänderung serviert.
Der Reward ist der Verifier: nur Ticks, deren Aktion nachweislich Fortschritt brachte, werden zu
Positivbeispielen (`--include-failed` für reine Imitation). Tests:
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`.

**Ergebnis des frischen Welt-Laufs** (48 Ticks, Tech-Tree aus dem Nichts → `stop: "goal"`):
35 Datensatzzeilen (26 Train / 9 Held-out). Der Held-out-Vergleich zeigt **keinen** Accuracy-Gewinn
(1.000 → 1.000) — der Split ist in-distribution, jede Zeile ist ein Zustand, den Laya im Lauf selbst
gelöst hat; der Fine-Tune zeigt sich in **Loss 0.734 → 0.532**, **ECE 0.501 → 0.392** und
**Brier 0.412 → 0.284** (besser kalibrierte Konfidenz). Ein echter Accuracy-Vergleich braucht einen
**out-of-distribution**-Split.

**Gefahr-Zeilen:** Ein `in_danger = yes`-Tick endet als **`defer`** (keine Aktion) — genau dort
steckt der echte `yes`-Fall. Der Builder nimmt solche Defer-Ticks jetzt als reine `in_danger`-Zeile
mit. Solange noch kein `yes`-Fall im Log liegt, mit `--no-danger` bauen, sonst zieht das Training
den geteilten `choice`-Kopf auf „immer no".

## M4: Parameter-Layer (Zielwahl)

`src/mlaya/targets.py` baut aus den `view_blocks`-Treffern eine Kandidatenliste von
**Standzellen** (`above: air`), `brain.choose_target` lässt Laya daraus wählen (bei mehr als
`TARGET_K = 6` Kandidaten via `predict_shortlist`), und `actions.resolve(label, gs, target)`
bindet das Ziel an die Werkzeug-Argumente. Erster Nutznießer ist `explore` → `navigate` mit
echter `(x, y, z)`-Position statt eines festen `x + 16`-Offsets. Neu ist außerdem, dass die
`view_blocks`-Abfrage zusätzlich nach `grass_block`/`sand`/`gravel`/`dirt` sucht und
`perception.BlockHit` die `above`/`below`-Annotation der Treffer behält.
