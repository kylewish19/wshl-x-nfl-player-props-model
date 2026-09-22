# 2026 Week 2 — Giants at Rams postgame

Final: Rams 28, Giants 6.

## Official card
- Betting record: **5-5 (50.0%)**.
- Jaxson Dart's two Under wins remain in betting W-L but are tagged `EARLY_GAME_INJURY` and excluded from probability-calibration learning.
- Clean/non-Dart model misses: Stafford U1.5 pass TDs, Singletary O11.5 rush yards, Corum U47.5 rush yards, Kyren U63.5 rush yards, Beckham O5.5 receiving yards.
- Hits: Stafford U0.5 rush yards, Skattebo U51.5 rush yards, Mooney O19.5 receiving yards, plus both Dart Unders.

## Shadow/add-on
- Full receptions + TE-yardage raw shadow card: **10-8 (55.6%)**.
- Highlighted add-on subset from the pregame summary: **2-4**.
- Game-market shadow bet NYG +6.5 lost (Rams won by 22). Keep game markets shadow-only.
- Abdul Carter sack +198 shadow hit. Do not lower the sack probability floor from one result.

## Manual injury gate audit
The pregame vacancy gate removed five Rams receiving-yard model BETs:
- Davante Adams U62.5 — would have lost (195)
- Blake Corum U3.5 — would have lost (13)
- Kyren Williams U16.5 — would have won (12)
- Konata Mumpfield U18.5 — would have won (12)
- Xavier Smith U17.5 — would have won (0)

The gate avoided two severe context-invalid losses but also passed three winners. Keep the gate because it protects against a known model-input limitation, not because of this one game's record.

## Lessons / model actions
1. **Do not learn from early-injury player outcomes as normal calibration observations.** Store betting result, but mark them calibration-ineligible.
2. **Rams RB miss pattern was opportunity + efficiency, not simply one bad player projection.** Both Kyren Williams and Blake Corum cleared their Under lines by large margins. Do not change official coefficients from one game; require v0.5 opportunity×efficiency shadow predictions on the next live card.
3. **Target-vacancy redistribution needs a historical challenger.** Puka Nacua/Jordan Whittington absences concentrated targets into Davante Adams and other Rams pass-catchers. Build/test a teammate-availability/vacated-target-share feature before promotion.
4. **Stafford passing-TD ceiling was badly undercalled.** Track red-zone target concentration and opponent defensive starter absences in the next challenger; no immediate official Poisson coefficient change after one game.
5. **Backup/secondary role uncertainty needs a stronger gate.** Singletary's actual workload (3 carries) and Beckham's zero receiving yards show why low-line props can still be fragile when role expectation is uncertain.
6. **No official champion promotion after Week 2.** Preserve v0.3 until prospective evidence accumulates. Refit on completed Week 2 data for the next card.
