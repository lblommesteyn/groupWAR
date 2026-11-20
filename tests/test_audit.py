import json
import unittest
from audit import audit_rows


def frame(t):
    return dict(game_id='0022500613', timestamp_ms=t, period=1, game_clock_ms=720000-t,
                home_players=[dict(player_id=i, on_court=True, xyz=[0, 0, 0]) for i in range(5)],
                away_players=[dict(player_id=i, on_court=True, xyz=[0, 0, 0]) for i in range(5, 10)],
                ball={'xyz': [0, 0, 4]})


class AuditTests(unittest.TestCase):
    def run_frames(self, frames):
        return audit_rows(map(json.dumps, frames))

    def test_missing_ball_breaks_run(self):
        frames = [frame(t) for t in range(0, 1000, 200)]
        frames[2]['ball'] = None
        result = self.run_frames(frames)
        self.assertEqual(result['complete_geometry_fraction'], .8)
        self.assertEqual(result['longest_complete_stable_lineup_run_frames'], 2)

    def test_duplicates_and_nonfinite_positions(self):
        f = frame(0)
        f['away_players'][0]['player_id'] = 0
        self.assertEqual(self.run_frames([f])['complete_geometry_fraction'], 0)
        f = frame(0)
        f['ball']['xyz'][0] = float('nan')
        self.assertEqual(self.run_frames([f])['complete_geometry_fraction'], 0)

    def test_period_gap_substitution_and_order(self):
        for change in ('period', 'gap', 'substitution'):
            a, b = frame(0), frame(200)
            if change == 'period': b['period'] = 2
            if change == 'gap': b['timestamp_ms'] = 2000
            if change == 'substitution': b['home_players'][0]['player_id'] = 11
            self.assertEqual(self.run_frames([a, b])['longest_complete_stable_lineup_run_frames'], 1)
        b = frame(200)
        b['home_players'].reverse()
        self.assertEqual(self.run_frames([frame(0), b])['longest_complete_stable_lineup_run_frames'], 2)

    def test_invalid_line_breaks_run(self):
        result = audit_rows([json.dumps(frame(0)), 'bad', json.dumps(frame(200))])
        self.assertEqual(result['counts']['invalid_json'], 1)
        self.assertEqual(result['longest_complete_stable_lineup_run_frames'], 1)

    def test_window_counts_do_not_cross_gaps(self):
        frames = [frame(i*200) for i in range(52)]
        result = self.run_frames(frames)
        self.assertEqual(result['nonoverlapping_windows_by_frames']['26'], 2)
        frames[25]['ball'] = None
        result = self.run_frames(frames)
        self.assertEqual(result['nonoverlapping_windows_by_frames']['26'], 1)
        self.assertEqual(sum(int(n)*v for n,v in result['complete_run_length_histogram'].items()), 51)


if __name__ == '__main__':
    unittest.main()
