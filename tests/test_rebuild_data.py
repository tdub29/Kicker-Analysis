"""Known role and timing failures must stay fixed before forecast evaluation."""
import numpy as np
import pandas as pd
import json
import hashlib
import pytest

from kicker_analysis.prepare import participation, coach_history, prepared_snapshot
from kicker_analysis.build_dataset import KEEP, ATT_BUCKETS, LONG_COLS
from kicker_analysis.features import build_features
from kicker_analysis.features import stadium_climate
from kicker_analysis.fourth_down import fourth_down_plays, team_game_rates
from kicker_analysis.predict_week import drop_released


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


def test_roster_guard_accepts_name_formats_but_never_released_or_wrong_team(tmp_path):
    roster = tmp_path/'players.json'
    roster.write_text(json.dumps({'1':{'position':'K','first_name':'Evan','last_name':'McPherson','team':None},
                                  '2':{'position':'K','first_name':'Jake','last_name':'Elliott','team':'LAR'}}))
    rows = pd.DataFrame([dict(player_id='k',player_display_name=name,team='CIN')
                         for name in ['E.McPherson','Evan McPherson']]
                        + [dict(player_id='j',player_display_name='Jake Elliott',team='LA')])
    assert drop_released(rows,roster).player_id.tolist() == ['j']


def test_veteran_new_team_zero_role_does_not_require_prior_same_team():
    context = pd.DataFrame([dict(season=2020,week=w,game_id=f'g{w}',team=team)
                            for w,team in [(1,'DAL'),(2,'MIN')]])
    outcome = {c:0 for c in KEEP + ATT_BUCKETS + LONG_COLS}
    outcome.update(season=2020,week=1,season_type='REG',player_id='k',
                   player_display_name='D.Bailey',team='DAL',pat_att=1,pat_made=1)
    snaps = pd.DataFrame([dict(game_id='g2',season=2020,week=2,game_type='REG',team='MIN',
        player='Dan Bailey',pfr_player_id='k',position='K',st_snaps=2)])
    players = pd.DataFrame([dict(pfr_id='k',gsis_id='k',position='K')])
    out,_,_ = participation(pd.DataFrame([outcome]),snaps,players,context)
    assert out.loc[out.week.eq(2),'participation_status'].item() == 'verified_active_zero'


def test_aborted_formations_do_not_become_go_and_blocked_recovery_pass_stays_fg():
    base = dict(season=2025,week=1,season_type='REG',posteam='LA',down=4,ydstogo=3,
        yardline_100=30,wp=.5,half_seconds_remaining=600,play_type='pass',
        field_goal_attempt=0,punt_attempt=0,rush_attempt=0,pass_attempt=1,
        field_goal_result=None,kick_distance=np.nan,aborted_play=1)
    frame = pd.DataFrame([base,{**base,'aborted_play':0,'field_goal_attempt':1,
                                  'field_goal_result':'blocked','kick_distance':48},
        {**base,'week':2,'aborted_play':0,'down':3}])
    selected = fourth_down_plays(frame)
    assert len(selected) == 1 and selected.is_fg.item() == 1 and selected.is_go.item() == 0
    counts = team_game_rates(frame).set_index('week')
    assert counts.loc[2,'fourth_in_range'] == 0
    assert pd.isna(counts.loc[2,'go_rate'])


def test_weather_support_counts_observations_not_missing_games():
    g = pd.DataFrame([dict(season=2025,gameday='2025-11-10',stadium_id='PHO00',
        stadium='Arizona',home_team='ARI',temp=np.nan,wind=np.nan)]*32
        + [dict(season=2026,gameday='2026-11-10',stadium_id='PHO00',
        stadium='Arizona',home_team='ARI',temp=np.nan,wind=np.nan)])
    r = stadium_climate(g).query('season == 2026').iloc[0]
    assert r.venue_games_prior == 32 and r.venue_wind_observations_prior == 0
    assert pd.isna(r.venue_wind_climate)


def test_only_completed_matching_generation_is_consumed(tmp_path):
    generation = tmp_path/'generations'/'complete'
    generation.mkdir(parents=True)
    names = ['kicker_games.parquet','fourth_down.parquet','coach_history.parquet']
    for name in names: (generation/name).write_bytes(b'complete')
    meta = {'generation':'generations/complete','prepared_sha256':
            {name:hashlib.sha256(b'complete').hexdigest() for name in names}}
    (tmp_path/'quality.json').write_text(json.dumps(meta))
    (tmp_path/'generations'/'unfinished').mkdir()
    assert prepared_snapshot(tmp_path)['kicker_games.parquet'] == b'complete'
    (generation/'fourth_down.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='modified data generation'):
        prepared_snapshot(tmp_path)
