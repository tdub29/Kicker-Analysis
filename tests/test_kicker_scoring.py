"""Scoring and simulation must describe the same kick outcomes."""
import numpy as np
import pandas as pd
import pytest

from kicker_analysis.scoring import BASE_SCORING, KickerScoring, MADE_BUCKETS
from kicker_analysis.model import CompoundKickerModel
from kicker_analysis.evaluate_rebuild import empirical_crps


PHI_KAP = {"fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3,
           "fgm_40_49": 4, "fgm_50p": 5, "xpm": 1, "fgmiss": -1, "xpmiss": -1}


def test_blocked_kicks_are_misses_and_zero_opportunity_scores_zero():
    row = {c: 0 for c in MADE_BUCKETS}
    row.update(fg_made_40_49=1, fg_att=3, fg_made=1, fg_missed=1, fg_blocked=1,
               pat_att=3, pat_made=1, pat_missed=1, pat_blocked=1)
    zero = {k: 0 for k in row}
    data = pd.DataFrame([row, zero])
    scoring = KickerScoring.from_sleeper(PHI_KAP)
    assert scoring.points(data).tolist() == [1, 0]  # 4+1-2-2
    assert BASE_SCORING.points(data).tolist() == [5, 0]
    # Missing/aborted outcomes must not be quietly charged as misses.
    data.loc[0, "pat_att"] += 1
    with pytest.raises(ValueError, match="unclassified pat"):
        scoring.points(data)
    with pytest.raises(ValueError, match="unsupported kicker"):
        KickerScoring.from_sleeper({**PHI_KAP, "fgm_yds": .1})
    assert scoring.identity == KickerScoring.from_sleeper(dict(reversed(list(PHI_KAP.items())))).identity


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value)


def controlled_model(scoring, fg=2., pat=3., fg_success=.8, pat_success=.9):
    model = CompoundKickerModel(features=["a"], n_sims=100000, scoring=scoring)
    model.fg_, model.pat_ = Constant(fg), Constant(pat)
    model.make_rate_, model.pat_make_rate_ = fg_success, pat_success
    model.mix_ = np.array([0., 0., .5, .25, .25, 0.])
    model.imp_ = type("Identity", (), {"transform": staticmethod(np.asarray)})()
    return model


def test_scoring_identity_conversion_and_categorical_allocation():
    # E = 2*.8*(.5*3+.25*4+.25*5) + 3*.9 - 2*.2 - 3*.1 = 8.
    model = controlled_model(KickerScoring.from_sleeper(PHI_KAP))
    result = model.predict(pd.DataFrame({"a": [0.]})).iloc[0]
    assert result.exp_points == pytest.approx(8.)
    assert result.simulation_mean_points == pytest.approx(8., abs=.04)
    assert result.floor_p20 <= result.median_points <= result.ceiling_p90
    assert result.p_ge_8 >= result.p_ge_10 >= result.p_ge_12 >= result.p_ge_15
    # Every certain 50-yard make is exactly five points, never a rounding bonus.
    model = controlled_model(BASE_SCORING, pat=0, fg_success=1)
    model.mix_ = np.array([0., 0., 0., 0., 1., 0.])
    draws = model._draw(np.zeros((1, 1)))
    assert np.all(draws % 5 == 0)
    # Failed PATs score penalties, not phantom conversions; negatives are valid.
    model = controlled_model(KickerScoring.from_sleeper(PHI_KAP), fg=0,
                             pat_success=0)
    result = model.predict(pd.DataFrame({"a": [0.]})).iloc[0]
    assert result.exp_points == -3
    assert result.ceiling_p90 <= 0
    assert empirical_crps([0, 2], [0, 1, 2]).tolist() == [.5, .5, .5]
