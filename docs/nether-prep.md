# Nether-Vorbereitung (Post-Portal-Schritt)

Status: **Entwurf / geplant** — dokumentiert, noch nicht implementiert (Stand: siehe Git-Historie
`docs/nether-prep.md`).

## Idee

Zwischen „Portal brennt" (`light_portal`) und „Bot betritt den Nether" fehlt eine
**Vorbereitungsstufe**: der Bot soll erst dann durch das Tor gehen, wenn Rüstung, Werkzeug,
Nahrung und Baumaterial wirklich stimmen. Genau wie beim Essen (siehe `NETHER_FOOD_RESERVE = 64`
in `README.md`) gilt: lieber ordentlich vorbereitet als schnell hinein und alles verlieren.

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
| Baublöcke (Bruchstein) | ✅ trägt reichlich; `mine_stone` hält ≥ 12 | 🟡 Mindestvorrat für den Nether anheben |
| Bogen + Pfeile | ⛔ noch nicht gebaut | 🟡 **wichtigste Lücke** (siehe unten) |
| Axt / Elytra | – / ⛔ | Axt unnötig (kein Holzhacken im Nether), Elytra = End |
| Feuerschutztrank / Wassereimer | ⛔ | Henne-Ei bzw. verdunstet — siehe unten |
| Feuerwerk | ⛔ | braucht Elytra + Gunpowder |

| Empfehlung (Offhand/Inventar) | Bot-Stand | Bewertung |
|---|---|---|
| Goldrüstung gegen Piglins | ⛔ kein Gold | 🟡 machbar mit neuer Kette (`mine_gold`) |
| Feuerzeug (Portal neu anzünden) | ✅ `craft_flint_and_steel` (getragen) | erledigt |
| Totem der Unsterblichkeit | ⛔ | Evoker/Raid |
| Shulker-Kiste | ⛔ | End-Stadt |
| Volle Rüstung (Diamant) | ✅ alle 4 Teile getragen | erledigt (Verzauberung ⛔) |

### Der Bogen ist die eigentliche Lücke

Der **Server-Kampf feuert den Bogen schon selbst** (`survival.BASE_CHANGES["combat"]["bow"] = True`);
der Bot muss ihn also nur *haben* und **Pfeile** dabei haben. Damit ist das keine neue
Kampf-KI, sondern reine Beschaffung — genau wie beim Essen:

- **Bogen** = 3 Stöcke + 3 Fäden. Fäden liefert die Spinnenjagd (`hunt_string` existiert bereits).
- **Pfeile** = Flint + Stock + Feder. Flint (`get_flint`) und Federn (Hühner aus `hunt_food`)
  sind bereits Teil der Ketten.

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

## Sicherheits-Gates (müssen mit dem Schritt kommen)

- **Nie im Nether schlafen / Bett legen.** Betten explodieren dort. `sleep` und `place_bed` müssen
  auf `gs.dimension == overworld` beschränkt werden (Feld `perception.GameState.dimension` ist da;
  `apply_policy` wendet die Server-Policy bei Dimensionswechsel automatisch neu an).
- **Kein `fill_bucket` im Nether** (Wasser verdunstet).
- **Nahrung und Bruchstein** vor dem Durchgang prüfen (Mindestvorrat), damit der Bot nicht „aus
  Versehen" mit halbem Inventar hineinläuft.

## Vorgeschlagene Umsetzung in Stufen

**Phase 0 — erledigt (Referenz):** Diamant-Kit, Schild, Feuerzeug, 64 Essen, Bruchstein.

**Phase 1 — günstig, hoher Nutzen (Empfehlung als Nächstes):**
- **Bogen + Pfeile** beschaffen (`craft_bow`, `craft_arrows`) mit eigenem Mindestvorrat (z. B.
  Bogen vorhanden ∧ ≥ 16 Pfeile) — Server feuert.
- **Bruchstein-Mindestvorrat** für den Nether anheben (eigener Zielwert, z. B. 64).
- Der Bot hat schon eine Spare-Pickaxe-Reflex (`replace_tool`) — vor dem Durchgang sicherstellen.

**Phase 2 — größer, optional:**
- **Goldhelm** (ein Stück goldene Rüstung hält Piglins neutral): neue Kette
  `mine_gold` → `craft_gold_helmet`. Golderz braucht nur eine Eisen-Spitzhacke (hat der Bot).
  Erfordert Golderz im `view_blocks`-Scan und ein Rezept.

**Phase 3 — der eigentliche Nether-Besuch (nächste große Kette):**
- **`enter_portal`**: hinein navigieren (`dimension → nether`).
- **Schutzraum** bauen, **Rückweg** sichern, dann die erste Nether-Ressource (Blaze/Neterrack) als
  Ziel der darauffolgenden Kette.

## Außerhalb der Reichweite (bewusst nicht gebaut)

Elytra + Feuerwerk (End/Gunpowder), Totem (Evoker/Raid), Shulker-Kiste (End), Feuerschutztrank
(Nether — Henne-Ei), Verzauberung (Tisch/Lapis/XP), goldene Karotten (Anbau + Gold),
Armbrust (wie Bogen, aber aufwändiger — Bogen reicht).
