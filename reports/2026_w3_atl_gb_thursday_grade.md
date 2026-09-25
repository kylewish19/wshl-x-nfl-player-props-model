# 2026 Week 3 Thursday — Falcons at Packers grading

Final: **Atlanta 35, Green Bay 14**.

## Official locked card
**2-3 (40.0%)**

- Kaleb Johnson U37.5 rushing — WIN (6)
- Michael Penix Jr. U5.5 rushing — WIN (-2)
- Jordan Love U1.5 passing TDs — LOSS (2)
- Chris Brooks O14.5 rushing — LOSS (0)
- Jordan Love U231.5 passing yards — LOSS (312)

All five outcomes are calibration-eligible; there was no early-injury distortion like the Week 2 Jaxson Dart game.

## Thursday-only diagnostic notes — NO MODEL CHANGES YET
1. Green Bay's negative game script was the dominant miss driver. Love attempted 53 passes after Atlanta built a large lead, producing 312 yards and 2 TDs. The current pregame model did not capture the tail risk of a 50+ attempt comeback script strongly enough.
2. Chris Brooks' rushing miss was primarily opportunity: zero carries. The Week 2 lesson about low-volume secondary-back role uncertainty remains important.
3. Kaleb Johnson's Under was a clean hit: 4 carries, 6 yards. Both v0.3 and v0.5 agreed.
4. Penix U5.5 rushing was a clean hit at -2 yards in his return.
5. The Week 2 context gate worked extremely well here. Of nine raw v0.3 receiving signals removed because of Jayden Reed's absence / Penix's return, eight would have lost. Only Chris Brooks U6.5 receiving would have won.
6. Packers target-vacancy redistribution was real: Matthew Golden 100 yards, Christian Watson 96, Skyy Moore 31. The removed Unders on all three would have lost.
7. Atlanta QB-change protection was also justified: Drake London exploded for 194 yards; removed London U65.5 lost by 128.5 yards.
8. The add-on reception/TE model had a better night: 7-5 on rows it tagged BET and 10-7 on all directional predictions. Keep shadow-only until Tuesday/full Week 3 review.
9. The game-market shadow went 2-1: ATL +4.5 and ATL ML +205 won; Under 43.5 lost. Do not promote — historical validation still says the game-market family has not beaten the closing market.
10. The two highlighted MarShawn Lloyd alternate-rushing shadows both lost; he rushed for 11 yards.
11. Generic alternate-line logic still needs a Tuesday audit: rows with only an Over price can be labeled an UNDER BET when the Under side is not actually offered. Do not change code midweek; add this to Tuesday's fix list.

## Week 3 freeze
Do **not** retrain, change thresholds, alter feature sets, or promote a challenger before Sunday/Monday. Sunday and Monday use the same frozen Week 3 model state. Full Week 3 diagnosis and any code changes happen Tuesday.
