# Nether-Vorbereitung (Post-Portal-Schritt)

Status: **Phase 1 + 2 implementiert** (Bogen + Pfeile, Bruchstein-Vorrat, Goldhelm; Sicherheits-
Gates). Phase 3 (der eigentliche Nether-Besuch) ist noch offen. Details unten.

## Idee

Zwischen „Portal brennt" (`light_portal`) und „Bot betritt den Nether" fehlt eine
**Vorbereitungsstufe**: der Bot soll erst dann durch das Tor gehen, wenn Rüstung, Werkzeug,
Nahrung und Baumaterial wirklich stimmen. Genau wie beim Essen (siehe `NETHER_FOOD_RESERVE = 64`
in `README.md`) gilt: lieber ordentlich vorbereitet als schnell hinein und alles verlieren.

Die Vorbereitung läuft in **derselben Tages-Lull** wie das Essen (siehe `_nether_idle`) und ist
jetzt Teil des **Auto-Stops**: `tech_tree_complete` verlangt zusätzlich, dass kein Vorbereitungs-
Schritt mehr offen ist (`_all_nether_prep_steps`). Damit bereitet sich der Lauf fertig vor,
statt mit dem Bett in der ersten Nacht zu enden. Jeder Schritt fällt aus der Liste, sobald er
nicht mehr weiterkommt (kein Spawn, kein Erz) — ein unerreichbares Item blockiert den Lauf also
nie.

Der Nether ist ein anderes Spiel als die Oberwelt:

- **Hunger fällt schneller** (Sprinten, Kämpfe), und der **Tod kostet das ganze Inventar** —
  deshalb der große Essensvorrat.
- **Ghasts zerstören Blöcke**, auch den Portalrahmen → Feuerzeug zum Neu-Anzünden, Bruchstein
  als Schutz.
- **Wasser verdunstet**, **Betten explodieren**, **Uhren/Kompass spinnen**.

Der Schritt hat zwei Hälften: **(A) Inventar vorbereiten** (vor dem Durchgang) und **(B) nach der
Ankunft** (Schutzraum, Rückweg). Dieser Entwurf deckt beides ab, priorisiert aber die
Inventar-Vorbereitung, weil sie an der bestehenden Kette hängt.

## (A) Optimales Nether-Inventar — Abgleich mit dem, was der Bot kann

Legende: ✅ vorhanden · 🟡 machbar (Vorschlag) · ⛔ außerhalb der Reichweite dieses Bots.

| Empfehlung (Hotbar) | Bot-Stand | Bewertung |
|---|---|---|
| Schwert (Diamant) | ✅ `craft_diamond_sword` (getragen) | erledigt |
| Spitzhacke | ✅ `craft_diamond_pickaxe` (getragen) | erledigt |
| Schild | ✅ `craft_shield` + `equip_shield` (Offhand) | erledigt |
| Nahrung (Steak) | ✅ 64 × gebratenes Fleisch (`NETHER_FOOD_RESERVE`) | erledigt |
| Baublöcke (Bruchstein) | ✅ `NETHER_COBBLE = 64` im Lull | erledigt |
| Bogen + Pfeile | ✅ `craft_bow` + `craft_arrows` (≥ 16) | erledigt — Server feuert |
| Axt / Elytra | – / ⛔ | Axt unnötig (kein Holzhacken im Nether), Elytra = End |
| Feuerschutztrank / Wassereimer | ⛔ | Henne-Ei bzw. verdunstet — siehe unten |
| Feuerwerk | ⛔ | braucht Elytra + Gunpowder |

| Empfehlung (Offhand/Inventar) | Bot-Stand | Bewertung |
|---|---|---|
| Goldrüstung gegen Piglins | ✅ `mine_gold` → `smelt_gold` → `craft_gold_helmet` (bereit) | erledigt (Tragen erst im Nether) |
| Feuerzeug (Portal neu anzünden) | ✅ `craft_flint_and_steel` (getragen) | erledigt |
| Totem der Unsterblichkeit | ⛔ | Evoker/Raid |
| Shulker-Kiste | ⛔ | End-Stadt |
| Volle Rüstung (Diamant) | ✅ alle 4 Teile getragen | erledigt (Verzauberung ⛔) |

### Der Bogen ist die eigentliche Lücke

Der **Server-Kampf feuert den Bogen schon selbst** (`survival.BASE_CHANGES["combat"]["bow"] = True`);
der Bot muss ihn also nur *haben* und **Pfeile** dabei haben. Damit ist das keine neue
Kampf-KI, sondern reine Beschaffung — genau wie beim Essen:

- **Bogen** = 3 Stöcke + 3 Fäden. Fäden liefert die Spinnenjagd (`hunt_string`, Ziel im Lull 3).
- **Pfeile** = Flint + Stock + Feder. Flint (`get_flint`, nach dem Feuerzeug wieder geöffnet) und
  Federn (`hunt_feather`, Hühner) sind Teil der Ketten.

Der Server-Scaffold kennt außerdem schon `netherrack`/`basalt` (`scaffold_blocks`), das
Nether-Gelände ist also im Serverreflex vorgesehen.

### Warum Feuerschutz & Wassereimer nicht gehen

- **Wassereimer** verdunstet im Nether — ein Vorrat daran ist sinnlos.
- **Feuerschutztrank** braucht einen Braustand und **Blaze Powder aus dem Nether** — das ist ein
  **Henne-Ei**: den Trank kann es erst *nach* dem ersten Nether-Besuch geben. Also kein Pre-Nether-
  Item; er wird frühestens nach der ersten Blaze-Rute relevant.
- **Verzauberung** (Feuerschutz/Protection) bräuchte Verzauberungstisch + Lapis + XP — der Bot
  verzaubert nie.

## (B) Nach der Ankunft

1. **Schutzraum** aus Bruchstein um das Ankunftsportal (Hülle, das Portal als Tür), damit Ghasts
   den Rahmen nicht zerschießen. Bruchstein hat Blast-Resistance 6 und hält Ghast-Feuerbällen stand.
2. **Rückweg** offenhalten (Portal zuerst anzünden/absichern).
3. **Sicherheits-Gate im Nether** (siehe unten).

## Sicherheits-Gates (mit dem Schritt umgesetzt)

- **Nie im Nether schlafen / Bett legen.** Betten explodieren dort. `sleep`, `place_bed` (und
  `fill_bucket`, da Wasser verdunstet) sind über `_overworld(gs)` auf die Oberwelt beschränkt
  (`gs.dimension`; `apply_policy` wendet die Server-Policy bei Dimensionswechsel automatisch neu an).
- **Nahrung und Bruchstein** vor dem Durchgang prüfen (Mindestvorrat), damit der Bot nicht „aus
  Versehen" mit halbem Inventar hineinläuft.
- **Gold-Grab nur am Tag:** der Tiefen-Grab nach Gold ist nachts nicht im Angebot (der Lauf
  schläft dann), bleibt aber für den Auto-Stop offen — sonst würde die Nacht den Lauf zu früh beenden.

## Umsetzung in Stufen

**Phase 0 — erledigt (Referenz):** Diamant-Kit, Schild, Feuerzeug, 64 Essen, Bruchstein.

**Phase 1 — erledigt:** 
- **Bogen + Pfeile** (`craft_bow`, `craft_arrows`, `hunt_feather`): Bogen = 3 Fäden (Spinnen,
  `hunt_string`) + 3 Stöcke; Pfeile = Flint (`get_flint`, nach dem Feuerzeug wieder geöffnet) +
  Stock + Feder (Hühner). Ziel: Bogen vorhanden ∧ ≥ 16 Pfeile (`NETHER_ARROWS`). Der Server feuert
  den Bogen selbst (`combat.bow`).
- **Bruchstein-Mindestvorrat** im Lull auf 64 angehoben (`NETHER_COBBLE`); die leichte Schwelle 12
  gilt weiter außerhalb des Lulls. Vor dem Durchgang trägt der Bot einen Ersatz-Spitzhacken-Reflex
  (`replace_tool`).

**Phase 2 — erledigt (Goldhelm, getragen):** 
- **Goldhelm** (`mine_gold` → `smelt_gold` → `craft_gold_helmet`), ein Stück goldene Rüstung hält
  Piglins neutral. Golderz wird wie Diamant über `resource_found` gefunden (`gold_ore` in
  `perception.CHAIN_NAMES`) und per `collect_block`-Tunnel abgebaut; Eisen-Spitzhacke genügt.
- **Offen:** der Helm wird **getragen erst im Nether** — er belegt denselben Slot wie der
  Diamanthelm (siehe offene Frage unten). Bis dahin nur im Gepäck.

**Phase 3 — offen (der eigentliche Nether-Besuch):**
- **`enter_portal`**: hinein navigieren (`dimension → nether`).
- **Schutzraum** bauen, **Rückweg** sichern, dann die erste Nether-Ressource (Blaze/Neterrack) als
  Ziel der darauffolgenden Kette.

## Offene Frage: Goldhelm vs. voller Diamant-Satz

Ein Stück goldene Rüstung hält Piglins neutral, aber der Bot trägt einen **vollen Diamant-Satz** —
der Goldhelm würde den **Diamanthelm ersetzen** (ein Rüstungsslot ist belegt). Das ist eine
Abwägung: Piglin-Neutralität gegen etwas weniger Schutz. Der Schritt legt den Helm deshalb nur an
(noch nicht getragen) bereit; die Entscheidung „im Nether Goldhelm statt Diamanthelm" gehört zu
Phase 3 (oder der Bot trägt goldene Stiefel statt Diamantstiefel).

## Außerhalb der Reichweite (bewusst nicht gebaut)

Elytra + Feuerwerk (End/Gunpowder), Totem (Evoker/Raid), Shulker-Kiste (End), Feuerschutztrank
(Nether — Henne-Ei), Verzauberung (Tisch/Lapis/XP), goldene Karotten (Anbau + Gold),
Armbrust (wie Bogen, aber aufwändiger — Bogen reicht).
