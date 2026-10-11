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
    view_status + read_recent_events (frei) -> GameState -> kompakter NL-State (`narrative()`)
[3] Brain (S1)    (src/mlaya/brain.py)
    Laya-Router, EIN predict()-Pass:
      in_danger  (binäres Gate)      -> defer an die Survival-Policy
      danger     (score, nur Logging)
      intent     (choice)            -> unter den gefilterten Kandidaten
[4] Actions       (src/mlaya/actions.py)
    candidates(gs): gültige Labels nach Vorbedingungen (Tech-Tree-Reihenfolge)
    resolve(label, gs): Label -> echtes MCP-Tool + Argumente
[5] Verify        (src/mlaya/verify.py)
    view_status erneut lesen; Soll/Ist-Delta; Fortschritt je Aktion
[6] Memory        (src/mlaya/memory.py)
    runs/decisions.jsonl (jede Entscheidung + Outcome)
[7] System 2      (brain.escalate_to_system2)
    struktureller Trigger (Label nicht auflösbar) -> OpenRouter wählt eine gültige Alternative
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

## Nächste Meilensteine

- **M2**: Progression-Kaskade vertiefen (Planks/Sticks/Öfen, Eisen, Nether-Gate) und `view_blocks`
  zur Ressourcen-Erkennung nutzen (Holz/Stein in der Nähe in den State).
- **M3**: Logs -> Datensatz -> RLCD-Kopf-Fine-Tuning; Checkpoint via `LAYA_EXTRA_MODELS` registrieren.
- **M4**: Parameter-Layer ausbauen (Zielwahl über `view_blocks`-Kandidaten, `predict_shortlist`).
