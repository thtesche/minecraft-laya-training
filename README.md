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

Aktueller Stand (**M2/M2b/M2c**): Die Progression läuft von Holz über einen Werktisch und Steinwerkzeuge
bis zur **Eisen-Spitzhacke** (Ofen + Schmelzen). `view_blocks` liefert die Ressourcen in der Nähe
(Holz/Stein/Eisen/Kohle mit Distanz) in den Zustand. Zusätzlich baut der Agent ein **Bett** (aus
Schafwolle oder Spinnenfäden), **schläft nachts** darin (überspringt die Nacht, setzt den
Respawn-Punkt), liest die anstehenden **Notifications** mit und setzt beim Start die
**Server-Survival-Policy** (Scaffold-Blöcke, vorsichtiges Kampf-Fenster, `food.raw.allow: always`);
gegessen wird über den Server-Reflex, Essen wird über `hunt_food`/`cook_food` beschafft, das einen
**Mindestvorrat von 16** essbaren Items hält. Die Policy läuft in zwei **Stufen** (`base` für Reisen/
Progression, `hunt` für eine feindliche Jagd).

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
