# Matched-seed GuitarSet transposition repeats

## Question and budget

Does the single-seed Tetrads gain repeat under different initializations and shuffle orders? The first seed 20261119 improved F by 1.06 and E by 2.05 percentage points, with quality regressions. Preserve that result and run seeds 20261120 and 20261121 for natural and transposition arms, for both teachers: eight fresh training jobs. No first-seed job is retrained or replaced.

Reuse the original player CQT cache, normalization and split. Both arms retain architecture, learning rate, 30 epochs, batch size, label mapping, sampling and checkpoint selection. Only the training seed differs from each arm's original config. Within each seed compare natural F to augmented F, and natural E to augmented E. There are still 47/119 updates per epoch for F/E, so no cross-teacher causal architecture ranking follows.

The cache contains a seed field inherited from its preparation config although extraction/splits/normalization do not depend on training seed. An explicit cache_config pointer permits changing only seed and augmentation relative to that cached control; other config changes are rejected. This preserves strict input checks without recomputing CQT for each model seed. Train normalization is never refitted to validation.

## Running

Import notebooks/02k_guitarset_transposition_repeats.ipynb into Kaggle, enable Internet and GPU, and run cells in order. It installs dependencies/tests, verifies official archives/cache, executes an eight-job short CPU smoke sequence, prepares/verifies the 300-recording cache, and then runs all full jobs sequentially in one GPU session. This is four times the number of full trainings in 02j, with full clean train/validation scoring each epoch; allow more runtime.

Download p2-guitarset-transpose-repeat-results.zip into runs/imports/ and save the executed notebook. The ZIP contains runs/p2-guitarset-transpose-repeat/ with generated configs, plan, eight checkpoints/histories/predictions and comparison.json. Raw audio/cache and smoke results are excluded.

Completed stages can be reused only with matching config, smoke flag, revision, source hashes and cache metadata. Incomplete stages are rejected and preserved, not silently restarted or overwritten. Keep the source revision fixed while running the experiment. If a session ends during a job, inspect the partial output before choosing a recovery path; there is no optimizer resume support.

Local orchestration check:

```powershell
.venv/Scripts/python.exe src/acr/repeat_guitarset.py --cache data/processed/guitarset-cqt-player-smoke --output runs/p2-guitarset-repeat-local-smoke --device cpu --smoke
```

## Aggregate and interpretation

The full aggregate has three seeds including the existing original metric snapshots from reports/p2/guitarset-cqt/player-results/ and transposition/results/. Their SHA-256 hashes are recorded in plan.json. Historical source revisions differ and remain documented; the training algorithm and per-arm settings are unchanged. Original checkpoints are not copied into the new ZIP. Smoke aggregation uses only smoke jobs and never combines them with historical full metrics.

For each teacher and metric, report mean and sample standard deviation (n-1 denominator), plus the paired augmented-minus-natural delta for each seed and its mean/std. Primary outcome is validation duration-weighted Tetrads; Root/MajMin/Triads and supported-event accuracy remain visible. Values in JSON are fractions, with delta times 100 giving percentage points. Selection remains each job's best validation Tetrads, earliest epoch on ties.

Inspect per-quality precision/recall and clean train/validation fit for all seeds before deciding whether the common-chord gains justify minor/extended-chord regressions. Do not select the most favorable seed or treat repeated correctness on the same recordings as new independent song/player evidence. Three seeds measure optimization variability on one split, not generalization across players or datasets. Reused-validation and gold-boundary E limitations remain. No promotion follows automatically from a positive mean.

## Local verification

All 61 tests pass. Eight CPU smoke jobs (two epochs each) complete; natural and transposition records have the intended training seeds while retaining the original cache seed/provenance. The smoke aggregate contains only the two smoke seeds, not the full historical seed. Reinvoking the completed sequence reuses stages and produces an identical comparison file. Notebook code cells compile. Evidence is reports/p2/guitarset-cqt/transposition/repeat-setup/verification.json. Full GPU repeat results remain pending.
