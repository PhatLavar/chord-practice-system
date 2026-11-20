"""Selection precision, N false positives and agreement threshold semantics."""
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/acr'))
from audit_confidence import decisions, tally


class ConfidenceTests(unittest.TestCase):
    def test_empty_selection_is_undefined_not_perfect(self):
        result = tally(np.array([0]), np.array([0]), np.array([.9]),
                       np.array([False]), np.array([True]), np.array(['song']))
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['mean_confidence'])
        self.assertEqual(result['coverage'], 0.)

    def test_n_false_positive_counts_in_operational_precision(self):
        target = np.array([0, -100, 1])
        pred = np.array([0, 0, 0])
        keep = np.array([True, True, False])
        result = tally(target, pred, np.ones(3), keep, np.ones(3, dtype=bool), np.array(['a', 'b', 'c']))
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['coverage'], 2/3)
        self.assertEqual(result['correct_coverage'], 1/3)
        self.assertEqual(result['accepted_songs'], 2)

    def test_agreement_requires_both_scores_and_chord_gates(self):
        f = [np.array([[.95, .95, .2], [.05, .05, .8]]),
             np.array([[.95, .95, .95], [.05, .05, .05]])]
        e = [np.array([[.95, .95, .95], [.05, .05, .05]]),
             np.array([[.8, .7, .95], [.2, .3, .05]])]
        _, confidence, mask = decisions(f, e, 1, 'agreement', .8, True)
        np.testing.assert_array_equal(mask, [True, False, False])
        np.testing.assert_allclose(confidence, [.8, .7, .95])
        _, _, conditional = decisions(f, e, 1, 'agreement', .8, False)
        np.testing.assert_array_equal(conditional, [True, False, True])
        e[1][:, 0] = [.1, .9]
        self.assertFalse(decisions(f, e, 1, 'agreement', .8, True)[2][0])


if __name__ == '__main__':
    unittest.main()
