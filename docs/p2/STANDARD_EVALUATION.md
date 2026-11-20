# Standard interval evaluation

Use the existing mir_eval implementation, pinned to 0.8.2, rather than reimplementing chord reductions. Install the optional evaluation dependency:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-eval.txt
.venv/Scripts/python.exe src/acr/evaluate_intervals.py --reference reference.lab --estimate estimate.lab --output runs/evaluation/metrics.json
```

Input LAB files contain start, end and Harte chord label. Both timelines must cover exactly the same time domain, with finite positive contiguous intervals. Represent unknown reference regions explicitly as X; N denotes actual no-chord. Missing intervals or mismatched domains fail rather than silently receiving N labels. Predictions must cover the reference domain before evaluation.

Root, MajMin, Triads and Tetrads use the library's vocabulary reductions and metric-specific exclusions. Each result contains correct seconds, eligible seconds and excluded seconds; unsupported reference comparisons returned as -1 are excluded. Zero eligible duration yields null. Corpus duration-weighted scores are sum(correct_seconds)/sum(eligible_seconds), computed separately per metric. An unweighted average of song scores is a different metric.

Reference implementation: https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/chord.py . Version is stored in output. Tests cover unequal interval lengths, X versus N, quality reduction, incorrect roots and malformed time domains. Optional metric tests skip when mir_eval is absent; install requirements-eval.txt for the complete suite.

## Current artifact rescoring

reports/p2/jaah-fe-pilot/duration_metrics.json rescores the four existing teachers on their exported gold intervals. Gaps are explicitly X. This is an oracle, supported-event subset diagnostic: exported data cannot recover excluded raw labels or omitted trailing regions. These scores are not full-track benchmark WCSR and do not establish comparability with a paper. F here means mean probabilities pooled over gold events, not its unconstrained frame timeline.

The results demonstrate why selection criteria matter: E+JAAH improves unweighted event accuracy but reduces duration-weighted Tetrads on this common scope. Retain both diagnostics. End-to-end benchmark evaluation requires full reference intervals, predicted boundaries, the same vocabulary/splits as the baseline, and explicit normalization provenance. E's gold-boundary condition must remain separately labeled.
