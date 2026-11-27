"""Player page data (artifact V6Yz1KpjRAKT71TS9t5jr5): results/context/players.json (player_context.py, PC) +
results/archetypes/mix.json (archetype_mix.py) -> OUT/players.json (one record per line) and OUT/meta.json.
  python build_page_data.py OUT_DIR   (Mac)"""
import json, sys
import numpy as np

OUTD = sys.argv[1].rstrip('/') + '/'
D = json.load(open('results/context/players.json')); P = D['players']
X = json.load(open('results/archetypes/mix.json'))
OFFN = ['Spot-up wing', 'Rim-running big', 'Hub big', 'Pick-and-pop big', 'Primary creator', 'Secondary creator', 'Versatile wing']
DEFN = ['Off-ball perimeter', 'Help-side forward', 'Point-of-attack stopper', 'Rim anchor', 'Low-involvement']
STYLE = {'onball_share': 'Time on ball', 'hold_per_touch': 'Hold per touch', 'pot_ast36': 'Potential assists', 'drives36': 'Drives',
         'cs_share': 'Catch-and-shoot share', 'cut_rate': 'Cuts', 'screen_rate': 'Screens set', 'paint_share': 'Time in paint',
         'crash_rate': 'Crashes offensive glass', 'spacing': 'Spacing (distance from ball)', 'own_touch': 'Touches', 'pass_rate': 'Passes per touch',
         'rim_presence': 'Rim presence on D', 'dmatch': 'Matchup distance', 'onball_def': 'Guards the ball', 'help': 'Help rotations',
         'matchup_usage': 'Guards high-usage players', 'nd_rate': 'Nearest defender on shots'}
ARCH_OFF = {0: ('One-touch swing', 'Catches and moves it on, on the move: about one touch, spread across the floor.'),
            1: ('Second-side creator', 'Handles twice or more from the perimeter: attacks a closeout or re-handles after a swing.'),
            2: ('Interior off-ball', 'No touches, near the rim: dunker spot, roller, seal, offensive-glass position.'),
            3: ('Spacer', 'No touches, far from ball and rim: corners and above the break.'),
            4: ('Interior hub', 'Touches near the rim and handler: post-ups, short rolls, handoffs.'),
            5: ('Lead handler', 'On the ball half the time, above the break: runs the possession.')}
ARCH_DEF = {0: ('Low man / interior help', 'Near the rim off the ball, between man and basket.'),
            1: ('Ball pressure (chasing)', 'On the handler far from the rim, often trailing the play.'),
            2: ('Rim protector', 'In the paint, guarding the ball near the rim.'),
            3: ('Perimeter stay-home', 'Far from the rim, attached to a shooter.'),
            4: ('Point-of-attack contain', 'On the handler at mid-distance, staying between man and rim.')}
base = [p for p in P if p.get('trk_minutes', 0) >= 500 and 'style' in p]
dist = {k: np.array([p['style'][k] for p in base]) for k in STYLE}
gd = np.array([p['guards_off_rating'] for p in P if 'guards_off_rating' in p])
gb = np.array([p['guarded_by_def_rating'] for p in P if 'guarded_by_def_rating' in p])
names = {p['id']: p['name'] for p in P}
rows = []
for p in P:
    r = {k: p.get(k) for k in ('id', 'name', 'team', 'pos', 'age', 'off', 'deff', 'tot', 'sd', 'tot_pct', 'poss_recent', 'trk_minutes')}
    dk = p.get('darko') or {}
    r['darko'], r['darko_o'], r['darko_d'] = dk.get('dpm'), dk.get('o'), dk.get('d')
    for k in ('off_role', 'def_role'):
        if k in p:
            r[k] = p[k]
    if 'style' in p:
        r['style'] = {k: round(float((dist[k] < p['style'][k]).mean() * 100)) for k in STYLE}
    if 'guards_off_rating' in p:
        r['guards'] = [p['guards_off_rating'], round(float((gd < p['guards_off_rating']).mean() * 100))]
    if 'guarded_by_def_rating' in p:
        r['guarded_by'] = [p['guarded_by_def_rating'], round(float((gb < p['guarded_by_def_rating']).mean() * 100))]
    r['shots'], r['defended'], r['mates'] = p['shots'], p['defended'], []   # pair fit dropped (pair_validate.py)
    f = {}
    for side in ('off', 'def'):
        e = X['players'][side].get(str(p['id']))
        if e:
            e = dict(e); e['mates'] = [dict(m, name=names.get(m['mate'], str(m['mate']))) for m in e['mates']]; f[side] = e
    if f:
        r['fluid'] = f
    rows.append(r)
rows.sort(key=lambda r: -r['tot'])
with open(OUTD + 'players.json', 'w') as fh:
    fh.write('[\n' + ',\n'.join(json.dumps(r, separators=(',', ':')) for r in rows) + '\n]\n')
meta = dict(generated=D['generated'], through=D['ratings_through'], model='duncan-2',
            off_roles=[dict(name=OFFN[r['id']], **r) for r in D['roles']['off']], def_roles=[dict(name=DEFN[r['id']], **r) for r in D['roles']['def']],
            synergy=D['synergy'], style_names=STYLE,
            arch={'off': [dict(id=a['id'], name=ARCH_OFF[a['id']][0], text=ARCH_OFF[a['id']][1], share=a['share'], profile=a['profile']) for a in X['archetypes']['off']],
                  'def': [dict(id=a['id'], name=ARCH_DEF[a['id']][0], text=ARCH_DEF[a['id']][1], share=a['share'], profile=a['profile']) for a in X['archetypes']['def']],
                  'lead': X['lead_handler']})
json.dump(meta, open(OUTD + 'meta.json', 'w'))
print(len(rows), 'players;', sum('fluid' in r for r in rows), 'with BORIS mixes; top 5:', [(r['name'], r['tot']) for r in rows[:5]])
