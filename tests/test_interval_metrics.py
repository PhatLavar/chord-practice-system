import sys
from pathlib import Path
import unittest


try:
    from acr.evaluation.intervals import score
except ModuleNotFoundError as error:
    if error.name != 'mir_eval':raise
    score=None


@unittest.skipIf(score is None,'Install requirements-eval.txt for interval metric checks')
class IntervalMetrics(unittest.TestCase):
    def test_duration_not_event_weighting(self):
        result=score([[0,9],[9,10]],['C:maj','D:maj'],[[0,10]],['C:maj'])
        self.assertAlmostEqual(result['metrics']['root']['score'],.9)

    def test_unknown_and_no_chord(self):
        result=score([[0,2],[2,3]],['X','N'],[[0,3]],['N'])['metrics']['root']
        self.assertEqual(result['eligible_seconds'],1)
        self.assertEqual(result['excluded_seconds'],2)
        self.assertEqual(result['score'],1)

    def test_reduction_and_wrong_root(self):
        result=score([[0,1]],['C:maj7'],[[0,1]],['C:maj'])['metrics']
        self.assertEqual(result['triads']['score'],1)
        self.assertEqual(result['tetrads']['score'],0)
        self.assertEqual(score([[0,1]],['C:maj'],[[0,1]],['D:maj'])['metrics']['root']['score'],0)

    def test_reject_missing_time(self):
        with self.assertRaises(ValueError):score([[0,1],[2,3]],['C:maj','N'],[[0,3]],['N'])
        with self.assertRaises(ValueError):score([[0,1]],['N'],[[0,2]],['N'])
