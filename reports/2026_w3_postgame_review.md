# 2026 Week 3 postgame review

## Official result
The exact locked Week 3 player-prop card finished **4-9 (30.8%)**:
- TNF ATL@GB: 2-3
- Sunday daytime: 0-4
- SNF LAR@DEN: 2-1
- MNF PHI@CHI: 0-1

The v0.5 opportunity×efficiency challenger went **3-8** on the 11 official markets it supports, the same directional W-L as v0.3 on that subset. **Do not promote v0.5.**

## Main failure modes

### 1. Opportunity / role forecasts were the biggest Over failure
Three official Overs failed because the player never received anything close to the modeled opportunity:
- Chris Brooks O14.5 rush: model 24.9, actual 0 rush yards / 0 carries.
- RJ Harvey O19.5 rush: model 41.4, actual 6 on 2 carries.
- Darius Cooper O5.5 rec: model 17.4, actual 0 on 0 targets.

Pat Bryant O24.5 receiving was the only official Over winner.

**Lesson:** an Over should not be promoted from yardage projection alone when its opportunity distribution is unstable. We need a live opportunity-reliability gate using current-season snaps, routes/targets/carries, depth-chart role, and injury-driven role changes.

### 2. Early-season probabilities were too confident
Several 77-85% official signals failed:
- CeeDee Lamb U82.5: 83.5% -> actual 112.
- Jahmyr Gibbs U89.5: 80.2% -> actual 99.
- James Cook U79.5: 77.3% -> actual 154.
- RJ Harvey O19.5: 85.5% -> actual 6.
- Jordan Love U1.5 pass TD: 76.3% -> actual 2.

**Lesson:** Week 1-4 live probabilities need stronger shrinkage/calibration. Current-season sample size is small and role/game-script tails are materially wider than the model's displayed confidence implies.

### 3. Game-script / team-volume tails were under-modeled
- Jordan Love was projected 205.7 passing yards and finished with 312 on 53 attempts after Green Bay fell behind.
- James Cook was projected 66.2 rushing yards and finished with 154.
- CeeDee Lamb and Jahmyr Gibbs both beat high Under thresholds.

**Lesson:** point projections can be reasonable while side probabilities are overconfident. Team play volume and game-state distributions need more variance, not simply a global point-projection adjustment.

### 4. The context gates were useful, but MNF showed over-pruning
Good MNF protections:
- Keenum U167.5 passing would have lost badly (247).
- Keenum U0.5 passing TD would have lost (2 TD).
- Bigsby O12.5 rushing would have lost (0 rushing yards).
- Barkley U72.5 rushing would have lost (82).
- Swift U60.5 rushing would have lost (84).

But the blanket QB-change gate also removed three strong Bears receiving Overs that all won:
- Rome Odunze O24.5 -> 44.
- Luther Burden III O35.5 -> 61.
- Kalif Raymond O22.5 -> 90.

And the prior raw MNF top-10 model list finished **6-4**, while the only official MNF promotion (Darius Cooper O5.5) lost.

**Lesson:** keep stale-QB and unstable-RB-role gates, but do not automatically veto every receiver because of a QB change. Test a probability haircut / uncertainty penalty rather than a binary pass gate.

## Development decision
- **v0.3 remains champion for now only because no challenger has earned promotion.**
- **v0.5 is not promoted.**
- Do not retrain blindly on Week 3.
- Before Week 4, test:
  1. an early-season probability calibration/shrinkage layer;
  2. an opportunity-reliability gate for Overs;
  3. stronger current-season snap/route/carry/target role features;
  4. wider game-script/team-volume variance;
  5. a continuous QB-change uncertainty penalty instead of the current blanket receiver veto.
- Any change must be backtested chronologically before it can affect Week 4 official picks.
