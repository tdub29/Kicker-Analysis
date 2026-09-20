# Superseded code, kept for the record

Nothing here runs in the current pipeline. It is the version that was replaced,
kept so the commit that replaced it can be read against what it replaced rather
than against a blank.

| file | why it was retired |
|---|---|
| `feature_engineering.py`, `modeling.py`, `evaluation.py`, `pipeline.py`, `data_ingestion.py` | The original pipeline. Its modelling layer was 17 lines: a three-feature linear regression predicting `homefppg`, the season-to-date fantasy AVERAGE. That predicts a rolling mean from three other rolling means, so it scores well and forecasts nothing. It also trained on home rows only, and `evaluate_predictions` was never called on a holdout, so no out-of-sample number existed anywhere. |
| `features_v1.py` | Correctly lagged, and still not deployable. It fed the model observed temperature and wind, which rank 3rd and 7th by importance and do not exist before kickoff: of 255 unplayed games in the schedule file, 0 carry either, while rest, division game, surface and home field carry all 255. It also admitted the game-day roof state and two columns that correlate 0.992 with the season, which is a calendar clock wearing a feature's name. |
| `evaluate_v1.py` | Sorted a table by MAE and read a winner off the top. On this target that is nearly meaningless: predicting a constant 8.0 every week beat both gradient-boosted models. Its `top5_hit` metric was an artifact (constant predictions made it return the alphabetically first five player ids) and its feature-importance block ran on 34 rows. |

The replacements are `features.py`, `model.py`, `backtest.py` and `predict_week.py`.
What each fix was worth, with numbers, is in `../METHOD.md`.
