"""Duration-weighted chord metrics using mir_eval on explicit full timelines."""
import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np

METRICS = ('root', 'majmin', 'triads', 'tetrads')


def validate(intervals, labels):
    intervals = np.asarray(intervals, dtype=float)
    if intervals.shape != (len(labels), 2) or not len(labels):
        raise ValueError('Expected nonempty intervals and labels')
    if not np.isfinite(intervals).all() or (intervals[:,1]<=intervals[:,0]).any() or intervals[0,0]<0:
        raise ValueError('Invalid interval bounds')
    if not np.allclose(intervals[1:,0], intervals[:-1,1], atol=1e-8, rtol=0):
        raise ValueError('Timeline gaps/overlaps: explicitly annotate missing reference regions as X')
    return intervals


def score(reference, ref_labels, estimated, est_labels):
    reference = validate(reference, ref_labels); estimated = validate(estimated, est_labels)
    if not np.allclose(reference[[0,-1],[0,1]], estimated[[0,-1],[0,1]], atol=1e-8, rtol=0):
        raise ValueError('Reference/estimate time domains differ')
    intervals, ref, est = mir_eval.util.merge_labeled_intervals(reference,ref_labels,estimated,est_labels)
    duration = np.diff(intervals,axis=1).ravel()
    results = {}
    for name in METRICS:
        values = getattr(mir_eval.chord,name)(ref,est)
        eligible = values>=0
        seconds = float(duration[eligible].sum())
        correct = float((values[eligible]*duration[eligible]).sum())
        results[name] = dict(score=correct/seconds if seconds else None,
                             correct_seconds=correct,eligible_seconds=seconds,
                             excluded_seconds=float(duration[~eligible].sum()))
    return dict(mir_eval_version=mir_eval.__version__,duration_seconds=float(duration.sum()),metrics=results)


def read_lab(path):
    rows=[line.split(maxsplit=2) for line in path.read_text().splitlines() if line.strip()]
    return [[float(a),float(b)] for a,b,_ in rows],[label for _,_,label in rows]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--estimate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh report path')
    result=score(*read_lab(a.reference),*read_lab(a.estimate))
    result['protocol']='Full explicit timelines; mir_eval metric-specific exclusions. Corpus scores sum correct/eligible seconds per metric, not unweighted song scores.'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
