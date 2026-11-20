"""Descriptive tracking metrics v2: event rates + split-half stability + RAPM-harness test.

  python metrics2.py stability   # Jan-Feb vs March reliability of every metric
  python metrics2.py eval        # add the new metrics to the embed_eval harness
"""
import json, math, pickle, sys
from collections import defaultdict
import numpy as np
import embed_eval as ee
from masked_paths import ROOT

EV = pickle.load(open(ROOT / 'results/metrics/event_counts.pkl', 'rb'))
import os
HOLES = os.environ.get('HOLES') == '1'
if HOLES:  # holes-field sums (holes_metrics.py) merged into the per-game event counts
    for g, c in pickle.load(open(ROOT / 'results/holes/player_game.pkl', 'rb')).items():
        for p, v in c.items():
            EV.setdefault(g, {}).setdefault(p, {}).update(v)

NEW_OFF = ['pass_rate', 'pot_ast36', 'pot_ast_quality', 'lost_pass_rate', 'hold_per_touch', 'drives36',
           'drive_finish', 'drive_kick_rate', 'drive_tov_rate', 'collapse', 'cut_rate', 'cut_fed_rate',
           'screen_rate', 'crash_rate', 'trans_share', 'cs_share']
NEW_DEF = ['closeout_speed', 'closeout_d1', 'closeout_res', 'beaten_rate', 'help', 'rim_deter',
           'switch_entropy', 'iso_def_adj']
HOLES_OFF = ['h_open_rate', 'h_reach_rate', 'h_spot', 'h_teamA', 'h_hold_A', 'h_create']
HOLES_DEF = ['h_cover_rate', 'h_oppA', 'h_onball_A']
if HOLES:
    NEW_OFF = NEW_OFF + HOLES_OFF
    NEW_DEF = NEW_DEF + HOLES_DEF


def shrink(num, den, prior, k):
    return (num + k * prior) / (den + k)


def describe2(G, gids, shots):
    d = ee.describe(G, gids, shots)
    gs = set(gids)
    S = defaultdict(lambda: defaultdict(float))
    for g in gs:
        for p, c in EV.get(g, {}).items():
            for k, v in c.items():
                S[p][k] += float(v)
    L = defaultdict(float)
    for c in S.values():
        for k, v in c.items():
            L[k] += v
    lg = lambda a, b: L[a] / max(L[b], 1)
    # shot-derived pieces
    faced, rim_faced, cs, own = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    iso_num, iso_n = defaultdict(float), defaultdict(float)
    for s in shots:
        if s['game_id'] not in gs or not s['trk_ok']:
            continue
        own[s['shooter']] += 1; cs[s['shooter']] += s['touch_s'] <= 1.2
        for q in s['def_ids']:
            faced[q] += 1; rim_faced[q] += s['dist'] < 6
        skill = d.get(s['shooter'], {}).get('shot_making', 0.0)
        iso_num[s['nd_id']] += s['made'] - s['xb'] - skill; iso_n[s['nd_id']] += 1
    lg_rim = sum(rim_faced.values()) / max(sum(faced.values()), 1)
    lg_cs = sum(cs.values()) / max(sum(own.values()), 1)
    # heights of matched offensive players -> switch entropy
    height = {p: f['height'] for p, f in d.items() if np.isfinite(f.get('height', np.nan))}
    pair_h = defaultdict(lambda: np.zeros(4))
    for g in gs:
        for (q, o), c in G[g]['pairs'].items():
            h = height.get(o)
            if h is not None:
                pair_h[q][0 if h <= 75 else 1 if h <= 79 else 2 if h <= 82 else 3] += c
    for p, f in d.items():
        c = S.get(p, {})
        g_ = lambda k: c.get(k, 0.0)
        m = f['minutes']
        f.update(
            pass_rate=shrink(g_('passes'), g_('touches'), lg('passes', 'touches'), 50),
            pot_ast36=36 * g_('pot_ast') / m,
            pot_ast_quality=shrink(g_('pot_ast_x'), g_('pot_ast'), lg('pot_ast_x', 'pot_ast'), 20),
            lost_pass_rate=shrink(g_('lost_pass'), g_('touches'), lg('lost_pass', 'touches'), 100),
            hold_per_touch=shrink(g_('hold_s'), g_('touches'), lg('hold_s', 'touches'), 50),
            drives36=36 * g_('drives') / m,
            drive_finish=(g_('drive_shot_made') - g_('drive_shot_x')) / (g_('drive_shot') + 30),
            drive_kick_rate=shrink(g_('drive_kick'), g_('drives'), lg('drive_kick', 'drives'), 20),
            drive_tov_rate=shrink(g_('drive_tov'), g_('drives'), lg('drive_tov', 'drives'), 20),
            collapse=shrink(g_('collapse'), g_('drives_full'), lg('collapse', 'drives_full'), 10),
            cut_rate=1000 * shrink(g_('cuts'), g_('offball_frames'), lg('cuts', 'offball_frames'), 2000),
            cut_fed_rate=shrink(g_('cut_fed'), g_('cuts'), lg('cut_fed', 'cuts'), 20),
            screen_rate=1000 * shrink(g_('screens'), g_('offball_frames'), lg('screens', 'offball_frames'), 2000),
            crash_rate=shrink(g_('crash'), g_('crash_opp'), lg('crash', 'crash_opp'), 20),
            trans_share=shrink(g_('trans_shots'), g_('linked_shots'), lg('trans_shots', 'linked_shots'), 20),
            cs_share=shrink(cs[p], own[p], lg_cs, 20),
            closeout_speed=shrink(g_('co_speed'), g_('co_n'), lg('co_speed', 'co_n'), 10),
            closeout_d1=shrink(g_('co_d1'), g_('co_n'), lg('co_d1', 'co_n'), 10),
            closeout_res=g_('co_res') / (g_('co_n') + 60),
            beaten_rate=100 * g_('beaten') / max(f.get('minutes', 1), 1),
            help=shrink(g_('help_move'), g_('help_n'), lg('help_move', 'help_n'), 20),
            rim_deter=shrink(rim_faced[p], faced[p], lg_rim, 50),
            switch_entropy=(lambda v: float(-(v / v.sum() * np.log(v / v.sum() + 1e-12)).sum()) if v.sum() > 30 else np.nan)(pair_h[p]),
            iso_def_adj=iso_num[p] / (iso_n[p] + 100))
        if HOLES:
            f.update(
                h_open_rate=shrink(g_('h_open'), g_('h_off_n'), lg('h_open', 'h_off_n'), 2000),
                h_reach_rate=shrink(g_('h_reach'), g_('h_off_n'), lg('h_reach', 'h_off_n'), 2000),
                h_spot=shrink(g_('h_hat'), g_('h_off_n'), lg('h_hat', 'h_off_n'), 2000),
                h_teamA=shrink(g_('h_teamA'), g_('h_off_n'), lg('h_teamA', 'h_off_n'), 2000),
                h_hold_A=shrink(g_('h_hold_A'), g_('h_hold_n'), lg('h_hold_A', 'h_hold_n'), 500),
                h_create=shrink(g_('h_create'), g_('h_touches'), lg('h_create', 'h_touches'), 50),
                h_cover_rate=shrink(g_('h_cover'), g_('h_def_n'), lg('h_cover', 'h_def_n'), 2000),
                h_oppA=shrink(g_('h_oppA'), g_('h_def_n'), lg('h_oppA', 'h_def_n'), 2000),
                h_onball_A=shrink(g_('h_onball_A'), g_('h_onball_n'), lg('h_onball_A', 'h_onball_n'), 500))
    return d


def stability():
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: v for g, v in G.items() if v['stints']}
    gA = sorted(g for g, v in G.items() if v['date'] < ee.A_END)
    gB = sorted(g for g, v in G.items() if ee.A_END <= v['date'] < ee.B_END)
    dA, dB = describe2(G, gA, shots), describe2(G, gB, shots)
    both = [p for p in dA if p in dB and dA[p]['minutes'] >= 300 and dB[p]['minutes'] >= 300]
    rows = []
    groups = [('box', ee.BOX), ('trk_off', ee.TRK_OFF), ('trk_def', ee.TRK_DEF), ('new_off', NEW_OFF), ('new_def', NEW_DEF)]
    for grp, cols in groups:
        for c in cols:
            a = np.array([dA[p].get(c, np.nan) for p in both], float); b = np.array([dB[p].get(c, np.nan) for p in both], float)
            ok = np.isfinite(a) & np.isfinite(b)
            r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 20 and a[ok].std() > 0 else np.nan
            rows.append((grp, c, float(r), int(ok.sum())))
    json.dump(rows, open(ROOT / 'results/metrics/stability.json', 'w'), indent=1)
    print(f'{len(both)} players with >=300 min in both Jan-Feb and March')
    for grp, c, r, n in rows:
        if grp != 'box':
            print(f'{grp:8s} {c:18s} r={r:+.2f}')
    print('box median r', np.nanmedian([r for g, c, r, n in rows if g == 'box']).round(2))
    return rows


def evaluate():
    arms = {'box': ee.BOX, 'trk_v1': ee.TRK_OFF + ee.TRK_DEF, 'trk_v2': ee.TRK_OFF + ee.TRK_DEF + NEW_OFF + NEW_DEF,
            'box+trk_v1': ee.BOX + ee.TRK_OFF + ee.TRK_DEF,
            'box+trk_v2': ee.BOX + ee.TRK_OFF + ee.TRK_DEF + NEW_OFF + NEW_DEF,
            'box+def_v2': ee.BOX + ee.TRK_DEF + NEW_DEF}
    ee.main(arms=arms, describe_fn=describe2, out_name='embed_eval_v2.json')


if __name__ == '__main__':
    {'stability': stability, 'eval': evaluate}[sys.argv[1]]()
