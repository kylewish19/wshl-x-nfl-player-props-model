# First Game Preflight

Before an official player-prop card is locked:

1. Refresh nflverse data and confirm completed 2026 games are present.
2. Use the frozen v0.3 official champions. Shadow challengers never replace official picks automatically.
3. Validate every player, prop name, line and FanDuel price with `scripts/preflight_game.py`.
4. Check final injury/inactive news, role/depth-chart changes and major offensive-line absences close to kickoff.
5. Check weather/roof for outdoor games and confirm current spread/total.
6. For sacks, FanDuel settlement treats an official half-sack as YES; our `def_sacks > 0` target matches that rule.
7. Lock the CSV before kickoff with timestamp/model version. Do not edit a locked card after results begin.
8. Generate the v0.3 official card and shadow cards side-by-side.
9. Grade all official BETs and retain PASS/shadow outcomes for calibration research.
10. Do not promote a challenger because of one week. Require prospective evidence plus historical walk-forward support.

## Current experimental status

- v0.3: OFFICIAL
- v0.4 matchup context: SHADOW; historical results did not justify promotion
- v0.5 opportunity x efficiency:
  - QB pass yards: +0.47% historical MAE improvement; shadow only
  - RB rush yards: +2.04%; historical promotion candidate, but shadow first
  - Receiving yards: +1.14%; historical promotion candidate, but shadow first
