"""Known role and timing failures must stay fixed before forecast evaluation."""
import numpy as np
import pandas as pd

from kicker_analysis.prepare import participation, coach_history
from kicker_analysis.build_dataset import KEEP, ATT_BUCKETS, LONG_COLS
from kicker_analysis.features import build_features


def test_placekicker_role_excludes_punter_and_kickoff_specialist_without_dropping_starter():
    context = pd.DataFrame([dict(season=2020,week=w,game_id=f'g{w}',team='LAC') for w in (1,2)])
    outcome = {c:0 for c in KEEP + ATT_BUCKETS + LONG_COLS}
    outcome.update(season=2020,week=1,season_type='REG',player_id='k',
                   player_display_name='Place Kicker',team='LAC',pat_att=1,pat_made=1)
    kicks = pd.DataFrame([outcome])
    snaps = pd.DataFrame([dict(game_id=f'g{w}',season=2020,week=w,game_type='REG',
        team='LAC',player=pid,pfr_player_id=pid,position='K',st_snaps=4)
        for w in (1,2) for pid in ('k','punter','kickoff')])
    players = pd.DataFrame([dict(pfr_id=pid,gsis_id=pid,position=pos)
                            for pid,pos in [('k','K'),('punter','P'),('kickoff','K')]])
    result,ledger,_ = participation(kicks,snaps,players,context)
    assert 'punter' not in set(result.player_id)
    assert result[result.player_id.eq('k')].model_eligible.all()
    assert result[result.player_id.eq('kickoff')].model_eligible.eq(False).all()
    assert result.loc[result.player_id.eq('k') & result.week.eq(2),'participation_status'].item() == 'verified_active_zero'
    assert ledger.eligible_rows.tolist() == [1,1]


def test_coach_history_uses_only_previous_weeks_and_counts_declined_long_kicks():
    base = dict(season=2025,season_type='REG',posteam='KC',home_team='KC',
        home_coach='Coach',away_coach='Other',down=4,ydstogo=3,yardline_100=40,
        wp=.5,half_seconds_remaining=600,score_differential=0,play_type='run',
        field_goal_attempt=0,punt_attempt=0,rush_attempt=1,pass_attempt=0,kick_distance=np.nan)
    plays = pd.DataFrame([{**base,'week':1}, {**base,'week':2,'field_goal_attempt':1,
                         'rush_attempt':0,'kick_distance':57,'play_type':'field_goal'}])
    context = pd.DataFrame([dict(season=2025,week=w,team='KC',coach='Coach') for w in (1,2)])
    history, choices = coach_history(plays,context)
    first, second = history.iloc[0], history.iloc[1]
    assert first.coach_long_menu_prior == 0
    assert second.coach_long_menu_prior == 1
    assert second.coach_long_kick_prior == 0  # week2 make cannot enter its own feature
    assert len(choices) == 2 and choices.is_go.sum() == 1


def test_missing_climate_is_unknown_and_authoritative_munich_is_not_a_dome():
    row = dict(season=2026,week=10,player_id='k',game_id='g',team='DET',opponent='NE',
        fantasy_points=0,fg_att=0,fg_made=0,pat_att=0,fg_long=0,team_score=0,
        roof='dome',team_spread=0,total_line=40,is_home=1)
    schedule = pd.DataFrame([dict(season=2026,game_id='g',gameday='2026-11-15',
        stadium_id='MUN01',stadium='Allianz Arena',home_team='DET',roof='dome',temp=np.nan,wind=np.nan)])
    result = build_features(pd.DataFrame([row]),schedule).iloc[0]
    assert result.is_indoor_certain == 0
    assert pd.isna(result.climate_wind_exposed) and result.climate_wind_missing == 1
