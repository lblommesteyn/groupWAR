"""NBA player movement (trades, signings, waivers, with dates and team ids) from the public nba.com data file, for
transaction-time mover definitions (AUDIT.md I2). Output: results/latent/transactions.json   python fetch_transactions.py (PC)"""
import collections, json, urllib.request
from masked_paths import ROOT

URL = 'https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json'
req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0 (research script)', 'Referer': 'https://www.nba.com/',
                                          'Accept': 'application/json'})
d = json.load(urllib.request.urlopen(req, timeout=60))
rows = d['NBA_Player_Movement']['rows']
print(len(rows), 'rows; example', rows[0])
print(sorted(collections.Counter(r['TRANSACTION_DATE'][:4] for r in rows).items()))
print(collections.Counter(r['Transaction_Type'] for r in rows))
json.dump(rows, open(ROOT / 'results/latent/transactions.json', 'w'))
