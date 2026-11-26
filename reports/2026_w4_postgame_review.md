# Week 4 postgame review / November 26 restart

Retrospective grading on 2026-11-26, using only the original locked cards for PIT@CLE (Oct 1) and DET@CAR (Oct 4). No predictions or lines reconstructed for missed weeks.

## Results
- Official 5-0 (TNF 4-0, SNF 1-0), +4.5014 units at one unit risk per play.
- Top-10 as locked 14-6 (8-2, 6-4).
- Raw BET as locked 15-5 (9-2, 6-3).
- Cumulative official record through Week 4: 14-14; the two injury-affected Dart wins remain excluded from probability calibration.
- Among 15 raw signals rejected by context gates, 5 losses were avoided and 10 winners filtered out. Do not infer superiority from this two-game sample.
- v0.5 on the 20 raw BET rows: 14-6, including its opposite TeSlaa direction; v0.3 15-5. No promotion.
- Official five all won in both directional models.
- SNF game-market shadow: Detroit -3.5 lost, Detroit ML lost, Under 51.5 lost (CAR 32 DET 26). Total was the only raw BET tag; all game markets remain shadow-only.

## Lessons / audit findings
- Target vacancy remains important: McMillan Under 71.5 lost (192 yards, 16 targets); Metchie Over 17.5 won (37 yards).
- Warren's two Unders lost (93 rush, 33 receiving); the role-change filter protected both.
- Hubbard rushing Under lost (122); Jameson Williams receiving Under lost (102).
- Over filters were conservative: Jeudy, Sanders, Boston, Concepcion and TeSlaa all cleared their lines despite context passes. No gate changes without OOS testing.
- SNF Top-10 was locked with Hutchinson 57.36% and did not include Goff Over 1.5 passing TDs 60.75%; it was not strictly sorted across all supported markets. Preserve original record, fix selection automation prospectively. 
- Report actual break-even probability alongside de-vig edge; do not confuse the latter with expected return. 
- Weeks 5-11 have no locked cards; no W-L backfill. November 25 Week 12 game has no locked card.
- Frozen artifacts for champion, v0.5 and game family downloadable as of restart. Current data completeness not yet verified; refresh and verify through completed games before forecasting. Candidate tests remain pending.

## Sources
- https://www.steelers.com/game-day/2026/week4/steelers-at-browns/
- https://www.statmuse.com/nfl/game/10-4-2026-car-vs-det-19127
- https://www.cbssports.com/nfl/gametracker/boxscore/NFL_20261004_DET%40CAR/
