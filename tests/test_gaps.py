import json
import unittest
from gap_distribution import analyze
from test_audit import frame


def run(missing, gap=False):
    rows=[]
    for i,miss in enumerate(missing):
        f=frame(i*200+(1000 if gap and i>=2 else 0))
        f['on_court']={'home':list(range(5)),'away':list(range(5,10))}
        if miss:f['home_players']=f['home_players'][1:]
        rows.append(json.dumps(f))
    return analyze(rows)


class GapTests(unittest.TestCase):
    def test_single_and_multiple_samples(self):
        self.assertEqual(run([False,True,False])['duration_ms_histogram']['players'],{200:1})
        self.assertEqual(run([False,True,True,False])['duration_ms_histogram']['players'],{400:1})

    def test_boundaries_censored(self):
        r=run([True,False,True])
        self.assertEqual(r['duration_ms_histogram']['players'],{})
        self.assertEqual(r['censored_episodes']['players'],2)

    def test_stream_gaps_not_imputed(self):
        r=run([False,True,True,False],gap=True)
        self.assertEqual(r['duration_ms_histogram']['players'],{})
        self.assertEqual(r['censored_episodes']['players'],2)
