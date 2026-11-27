import unittest
import numpy as np
import pandas as pd
from iggy_minutes import fit_role_scaler, project


class IggyEvaluationTests(unittest.TestCase):
    def test_future_role_values_cannot_change_preprocessing(self):
        train = {(1, pd.Timestamp('2023-11-01')): np.array([1., 2.]),
                 (2, pd.Timestamp('2024-03-01')): np.array([3., 8.])}
        mu, sd = fit_role_scaler(train)
        future = dict(train, **{})
        future[(3, pd.Timestamp('2025-01-01'))] = np.array([1e9, -1e9])
        mu2, sd2 = fit_role_scaler(future)
        np.testing.assert_array_equal(mu, mu2)
        np.testing.assert_array_equal(sd, sd2)

    def test_feasible_rotation_conserves_minutes(self):
        cap = np.full(10, 40.)
        minutes = project(np.arange(1, 11, dtype=float), cap)
        self.assertAlmostEqual(minutes.sum(), 240.)
        self.assertTrue(np.all(minutes <= cap))
        self.assertTrue(np.all(minutes >= 0))


if __name__ == '__main__':
    unittest.main()
