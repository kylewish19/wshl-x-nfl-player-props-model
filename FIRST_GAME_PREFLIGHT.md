# First Game Preflight

Before an official player-prop card is locked:

1. Refresh nflverse data and confirm completed 2026 games are present.
2. Use the frozen v0.3 official champions. Shadow challengers never replace official picks automatically.
3. Validate every player, prop name, line and FanDuel price with `scripts/preflight_game.py`.
4. Check final injury/inactive news, role/depth-chart changes and major offensive-line absences close to kickoff.
5. If one or more high-volume teammates are ruled out, flag all affected receiving/reception props as **target-vacancy risk** unless a vacancy-aware challenger explicitly handles the redistribution.
6. Check weather/roof for outdoor games and confirm current spread/total.
7. For sacks, FanDuel settlement treats an official half-sack as YES; our `def_sacks > 0` target matches that rule.
8. Lock the CSV before kickoff with timestamp/model version. Do not edit a locked card after results begin.
9. Generate the v0.3 official card and shadow cards side-by-side. For the next live card, v0.5 opportunity×efficiency must be logged prospectively for RB rushing and receiving yards.
10. Grade all official BETs and retain PASS/shadow outcomes for calibration research.
11. If a player exits very early because of an in-game injury, keep the sportsbook result in W-L but mark that row **calibration-ineligible**. Do not teach the probability model from an outcome created primarily by lost playing time.
12. Do not promote a challenger because of one week. Require prospective evidence plus historical walk-forward support.

## Current experimental status

- v0.3: OFFICIAL
- v0.4 matchup context: SHADOW; historical results did not justify promotion
- v0.5 opportunity x efficiency:
  - QB pass yards: +0.47% historical MAE improvement; shadow only
  - RB rush yards: +2.04%; historical promotion candidate; mandatory prospective shadow logging
  - Receiving yards: +1.14%; historical promotion candidate; mandatory prospective shadow logging
- Receptions / TE receiving yards add-on: SHADOW only after first live raw shadow card finished 10-8.
- Game markets: SHADOW only; current game-market model has not beaten the historical closing market.
