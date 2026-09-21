# NFL Game Markets Status

Spread, Moneyline and Game Total are now part of the NFL project as a separate model family.

## Current status

**Player props:** official model workflow.

**Game markets (Spread / ML / O-U): prospective shadow workflow.**

Historical testing did not show a reliable edge over the NFL closing market. The shrunken v0.3 blend explicitly allowed the model to receive 0%, 25%, 50%, 75% or 100% weight. On 2024 selection data, **0% model weight won for all three markets**. That choice was frozen before checking 2025.

Therefore we will not manufacture official game-market bets from a model that has not earned them.

## Live 2026 collection

For every game we analyze, record:
- FanDuel spread and both prices
- FanDuel moneylines
- FanDuel total and both prices
- timestamp / whether line is opening, current or closing
- independent model margin, total and win probability
- market-informed/shadow outputs
- final score and ATS/ML/O-U results
- closing-line value when an earlier line was captured

This prospective dataset will determine whether specific disagreement sizes, matchup types or model versions contain actionable edge.

## Promotion rule

A game-market challenger must outperform the market baseline prospectively and remain supported by walk-forward historical validation. One hot week is not enough.
