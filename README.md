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

Aktueller Stand (**M2/M2b/M2c/M2d/M3/M4**): Die Progression läuft von Holz über einen Werktisch und Steinwerkzeuge
bis zur **Eisen-Spitzhacke** (Ofen + Schmelzen) und weiter über `mine_diamond` zur **Diamant-Spitzhacke**
(+ optional Diamant-Schwert). `view_blocks` liefert die Ressourcen in der Nähe
(Holz/Stein/Eisen/Kohle/Diamant mit Distanz) in den Zustand. Zusätzlich baut der Agent ein **Bett** (aus
Schafwolle oder Spinnenfäden), **schläft nachts** darin (überspringt die Nacht, setzt den
Respawn-Punkt), liest die anstehenden **Notifications** mit und setzt beim Start die
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
```

Konfiguration: [`config/settings.toml`](config/settings.toml).
Jede Entscheidung wird nach `runs/decisions.jsonl` geloggt.

## M3: Logs → Datensatz → Fine-Tuning

```bash
# 1) Aus dem Run-Log einen Laya-Trainingsdatensatz bauen (train + held-out Eval)
.venv/bin/python -m mlaya.cli dataset            # -> data/train.jsonl + data/eval.jsonl
.venv/bin/python -m mlaya.cli dataset --include-failed   # auch Ticks ohne Fortschritt behalten

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

## M4: Parameter-Layer (Zielwahl)

`src/mlaya/targets.py` baut aus den `view_blocks`-Treffern eine Kandidatenliste von
**Standzellen** (`above: air`), `brain.choose_target` lässt Laya daraus wählen (bei mehr als
`TARGET_K = 6` Kandidaten via `predict_shortlist`), und `actions.resolve(label, gs, target)`
bindet das Ziel an die Werkzeug-Argumente. Erster Nutznießer ist `explore` → `navigate` mit
echter `(x, y, z)`-Position statt eines festen `x + 16`-Offsets. Neu ist außerdem, dass die
`view_blocks`-Abfrage zusätzlich nach `grass_block`/`sand`/`gravel`/`dirt` sucht und
`perception.BlockHit` die `above`/`below`-Annotation der Treffer behält.
