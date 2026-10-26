# Model experiments

Every change is tested the same way before it's adopted: same games, same
date-based split (train through 2025-11-15, test on the 1,215 newer games the
models never saw), same model settings, scored on the held-out test set and
on real graded QB prop lines. "Resamples better" is the share of 2,000
bootstrap resamples of the test games in which the change had the lower MAE
(near 100% = real gain, near 50% = no difference). The experiment ladder
that produced these numbers is `python -m scripts.compare_features` on the
`experiments/model-ideas` branch.

## Adopted

| Change | XGBoost test MAE | RMSE | Resamples better | Notes |
|---|---|---|---|---|
| Stats only (17 features) | 63.80 | 79.57 | — | starting point |
| **+ market features** (CFBD game total, team spread, implied team points) | **62.52** | **77.21** | **99.8%** | biggest gain since fixing leakage; in production |

## Tested, not adopted (October 2026)

Starting from the production model above (62.52 MAE / 77.21 RMSE):

| Change | XGBoost test MAE | RMSE | Resamples better | Verdict |
|---|---|---|---|---|
| Predict the median instead of the average | 62.84 | 77.46 | 0.6% | worse |
| ... + XGBoost re-tuned for the median (Optuna, 100 trials) | 62.56 | 77.23 | 37.5% | ties current |
| + per-attempt opponent defense (yards/attempt allowed, passes faced) | 62.52 | 77.24 | 49.3% | no change |
| + team pass rate & pace (CFBD team box scores) | 62.59 | 77.28 | 25.0% | no change |
| + prior-season QB stats (yards/game, yards/attempt, games) | 62.45 | 77.18 | 79.0% | tiny, not convincing |
| + all three groups | 62.56 | 77.28 | 37.6% | no change |

Without market features the new groups didn't help either (stats only 63.80
vs 63.85 with pass rate & pace, 63.96 with all three).

**Why the median didn't help:** the average model was already balanced: the
actual result beat its prediction 49.4% of the time, so it wasn't biased high
by big passing games the way we expected.

**Why the new features didn't help:** they're redundant with what the model
already has. Team pass rate and pace correlate with passing yards (+0.23 for
pass rate), but the QB's own recent attempts and yards, and his team's recent
passing volume, already capture how pass-heavy and fast his team is.

Only changes that improve results go into `main`, so none of this code is
here. It's preserved on the `experiments/model-ideas` branch (commit
`5c70813`): the `PREDICT_MEDIAN` switch and `xgboost_reg_tune.py --median`,
the three feature groups, the CFBD team box-score ingestion, and the
experiment ladder, so any of it can be re-tested as more prop-line games
come in. The team box scores already loaded (2022 to now) are still in the
shared database's `team_game_stats` table.

## Still the open question

None of these changes beat the sportsbook line on the 57 graded prop-line
games so far (line MAE 54.9 vs ~58 for every model version). More weeks of
prop lines, or a month of paid historical props, are what can answer that.
