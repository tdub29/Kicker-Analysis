"""Current app exports reject legacy or mixed-time forecast records."""
import json
import pandas as pd
import pytest

from kicker_analysis.predict_week import write_site, board_identity
from kicker_analysis.scoring import BASE_SCORING


def test_export_identity_scoring_and_separate_means(tmp_path):
    row = dict(player_id='gsis',player_display_name='A.Kicker',team='LA',opponent='KC',
        is_home=1,rank_points=1,rank_ceiling=1,exp_points=8,distribution_mean_points=7.5,
        ridge_points=8.2,floor_p20=2,ceiling_p90=14,p_ge_15=.09,p_ge_10=.3,
        implied_total=24,exp_fg_att=2,exp_pat_att=3,is_indoor_certain=0,stale_weeks=1,
        season=2026,week=10,model_id='reference',scoring_spec=json.dumps(BASE_SCORING.__dict__),
        scoring_id=BASE_SCORING.identity,source_id='source',input_id='inputs',board_id='board',
        created_utc='2026-11-15T00:00:00Z',training_cutoff='2026-W9',
        market_intercept=4,market_coefficient=.1667,starter_status='inferred')
    path = tmp_path/'board.js'
    board = pd.DataFrame([row])
    board["board_id"] = board_identity(board)
    write_site(board,2026,10,path)
    payload = json.loads(path.read_text().split('=',1)[1].strip().rstrip(';'))
    assert payload['schemaVersion'] == 2
    assert payload['kickers'][0]['exp'] == 8
    assert payload['kickers'][0]['distributionMean'] == 7.5
    assert payload['scoring']['fg_missed'] == 0
    with pytest.raises(ValueError,match='season/week'):
        write_site(board,2026,11,path)
    with pytest.raises(ValueError,match='legacy'):
        write_site(board.drop(columns=['board_id']),2026,10,path)
    tampered = board.copy()
    tampered["exp_points"] = 99
    with pytest.raises(ValueError,match='board content'):
        write_site(tampered,2026,10,path)
    tampered = board.copy()
    tampered["scoring_spec"] = json.dumps({**BASE_SCORING.__dict__, "fg_missed":-1})
    with pytest.raises(ValueError,match='scoring specification'):
        write_site(tampered,2026,10,path)
