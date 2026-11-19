# P2 instructions · Oracle event teacher E

P2 tests whether an event-supervised teacher makes errors different from frame teacher F. The experiment supplies reference interval times to both teachers. Recognition quality and complementarity are measured on those same intervals. Predicted chord boundaries belong to P3.

## Architecture and files

E uses the same 48-channel, five-dilation TCN as the original F (37,225 parameters), initialized and fitted independently. It averages hidden features over each reference event and applies the existing N/root/triad/extension heads once per event. Mean pooling excludes context and padded frames. TCN convolutions still receive up to 31 frames of neighboring evidence on each side.

| File | Purpose |
|---|---|
| src/acr/teacher_e.py | Event pooling and component classifier |
| src/acr/data_e.py | Timestamp-to-frame event mapping, variable-length batching and sampling |
| src/acr/train_e.py | Train E, score reference events and compare F/E outcomes |
| src/acr/prepare_p2_reference.py | Recreate frozen F interval predictions from its saved logits |
| configs/p2-overfit.json | Two-song pipeline check |
| configs/p2-oracle.json | Full oracle-event experiment |
| reports/p2/f-reference/events.csv | Frozen F predictions and standalone component decisions |
| notebooks/02_teacher_e_oracle.ipynb | Kaggle execution |

The F reference is the existing balanced TCN at seed 20261119 (revision d250916), chosen before observing E results. Its reference-event accuracy is 57.92% and rare macro event recall is 4.19%. The small repeat-seed gains and their limitations remain documented in [the P1 report](../p1/P1_REPORT.md).

## Dataset and masks

Reuse the verified P1 cache and the official full.tar.xz. IDs and splits are checked against the frozen manifest; annotation SHA-256 is verified. Merge touching equal supported root/quality labels as in P1. Unsupported intervals are breaks. Select frames using [start,end) timestamps and skip reference events with no sampled frames. Each event retains its full duration, without clipping or splitting long events.

N trains only the no-chord head. Unknown/OOV events are excluded. Neither test nor hidden-training annotations are read. Validation reference times define the evaluation intervals; labels are used for scoring and checkpoint selection. No reference labels are input features to E.

## Training choices

The overfit check uses 0003 and 0006, natural event sampling, 200 epochs and the same 95% training accuracy threshold. It verifies learnability.

The full pilot uses all 367 labeled songs and evaluates on 110 validation songs, seed 20261119, 30 epochs, batch size 32, Adam at 0.001 and gradient clipping at 5. Losses use unweighted component cross-entropy, as in F balanced.

For half the draws retain the shuffled natural event. For the other half, choose uniformly among observed chord qualities, uniformly among songs with that quality, then uniformly among their matching events. Sampling uses replacement and the number of draws per epoch equals the number of supported training events. Log actual quality-event exposures in history.json. This is the event counterpart of F's quality/song-balanced sampling.

One event contributes one target, irrespective of duration. F trained frame targets in 512-frame windows; E trains whole events with variable lengths. Their optimizer-step counts and target weighting therefore differ. Report those differences when interpreting complementarity. This first pilot does not isolate granularity from all optimization-budget effects.

## Run on Kaggle

1. Stop an older GPU session if necessary. Import notebooks/02_teacher_e_oracle.ipynb.
2. Enable Internet and GPU, then run setup and data preparation. It reuses the existing official data/P1 cache when available.
3. Run the overfit cell. Inspect event_accuracy and overfit_pass. Each target is an event, so the scored count differs from P1's frame count.
4. If the check passes, set RUN_FULL = True and execute the full cell. It trains E and writes comparison tables using the frozen F reference already stored in the repo. No upload of a P1 checkpoint is needed.
5. Run the export cell, save the executed notebook version and download p2-oracle-results.zip to runs/imports/.

Use a new run name for a retry; output directories are protected against overwriting. Variable-length batches consume memory according to the longest event in a batch. If CUDA runs out of memory, preserve the failure and record any batch-size change as a revised experiment.

## Outputs and interpretation

run.json records song IDs, event counts, source hashes, environment, model size and revision. best.pt is selected by complete-chord validation event accuracy. history.json logs losses, duration, actual event exposures and validation metrics per epoch. metrics.json includes rare macro event recall, natural-training event accuracy and peak CUDA tensor memory.

events.csv identifies every scored interval and prediction. event_logits.npz contains the component logits in the same row order. per_quality.csv reports precision, recall, song/event support and standalone component accuracy. complementarity.csv counts both correct, F only correct, E only correct and both wrong, grouped by quality and by complete chord or component. Root, triad and extension comparisons exclude N targets.

F averages probabilities over the supplied interval, while E pools hidden features before classification. They use independent learned weights. These are oracle-boundary diagnostics; they do not establish realistic predicted-boundary performance. Preserve the same F reference when assessing E.

## Local commands

Run from the project root:

```powershell
python -m unittest discover -s tests
python src/acr/train_e.py --config configs/p2-overfit.json --output runs/p2-overfit-local --device cpu
python src/acr/train_e.py --config configs/p2-oracle.json --output runs/p2-oracle/full --device cuda
```

The frozen F reference is already versioned. To recreate it locally from the verified original F run, choose a fresh output directory:

```powershell
python src/acr/prepare_p2_reference.py --run runs/p1-balanced-kaggle --output runs/p2-f-reference-check
```

Compare that events.csv with reports/p2/f-reference/events.csv. The preparation script recomputes F probabilities on the gold intervals, matching the P1 diagnostic.


The first Kaggle run has completed and been independently verified. Read [P2_REPORT.md](P2_REPORT.md) for standalone results, F/E complementarity, agreement precision and limitations before rerunning. Oracle-event results use supplied reference times. Full artifacts are in runs/p2-oracle/, with selected tables in reports/p2/kaggle-oracle/.


## Continue P2: natural-event E baseline

Import [02b_teacher_e_natural.ipynb](../../notebooks/02b_teacher_e_natural.ipynb), stop an older GPU session if needed, and enable Internet/GPU. Run the setup/data cells, then the natural-baseline training cell. The already verified overfit stage need not be repeated for this sampler-only change.

The cell trains 30 epochs from a new seeded initialization, using configs/p2-natural.json. The next cell compares its metrics with the completed balanced E run recorded in the repository. The final cell exports /kaggle/working/p2-natural-results.zip. Save the executed notebook version and download the ZIP into local runs/imports/.

The output is runs/p2-natural-kaggle/. Its event logits, per-quality results and F/E complementarity use the same protocol as the original E run. Use a fresh output name if retrying an existing result. The first natural-E run has completed and been verified; see the measured results in P2_REPORT.md. Further E development and the deferred P3 transition are recorded in [P2_REPORT.md](P2_REPORT.md).


The natural-E results are recorded under reports/p2/kaggle-natural/. They show improved overall accuracy with zero rare-event recall for this seed. Read the report's natural-versus-balanced comparison before choosing the next E training policy. Both configurations remain available.


## Repeat natural and balanced E with matched seeds

Import [02c_teacher_e_seed_repeat.ipynb](../../notebooks/02c_teacher_e_seed_repeat.ipynb) into Kaggle. Stop an older GPU session if needed, enable Internet/GPU and run setup/data preparation. Then run the training cell once. It automatically runs natural and balanced E for seeds 20261120 and 20261121 in sequence, each for 30 epochs, including F/E comparison. There is no need to rerun the already verified two-song check.

When all four jobs complete, inspect per_seed.csv and summary.json. Run the final cell, save the executed notebook version and download /kaggle/working/p2-seed-repeat-results.zip into runs/imports/.

Outputs are in runs/p2-seed-repeat/. The original seed pair is read from committed verified reports and not retrained. Each run uses the same fixed F reference. If interrupted, rerun at the same code revision/config to reuse completed jobs. An incomplete job is identified by its path; preserve it for diagnosis and either use a fresh output root (which reruns all four jobs) or remove only that failed subdirectory before retrying.

Local command, from the project root:

```powershell
python src/acr/repeat_e.py --output runs/p2-seed-repeat --device cuda
```

The plan is configs/p2-seeds.json. The first E repeat sweep has completed and been verified; see the measured summary in P2_REPORT.md. See [P2_REPORT.md](P2_REPORT.md) for fixed controls, interpretation and the distinction between E-only seed variation and F/E paired-seed variation.


The E seed-repeat archive has been independently checked. Selected results and per-seed quality tables are under reports/p2/seed-repeat/. Natural and balanced show a repeatable overall-versus-rare tradeoff over these three seeds; inspect the report's precision and distinct-interval analysis before choosing another sampling policy.


## Run the 25% balanced-mixture screening

Import [02d_teacher_e_mixture25.ipynb](../../notebooks/02d_teacher_e_mixture25.ipynb), stop an older active GPU session if needed, enable Internet/GPU and run its cells in order. The notebook trains one new E model for 30 epochs at seed 20261119. It compares the result with the committed natural and 50% balanced E results at the same seed.

The new output is runs/p2-mixture25-kaggle/. Its run.json records balanced_probability=0.25 and natural_probability=0.75. The final cell exports /kaggle/working/p2-mixture25-results.zip. Save the executed version and download that ZIP to local runs/imports/.

No repeat of the unchanged architecture's overfit check is required for this sampler-only trial. Existing output directories are protected; choose a fresh name for retries. The first validation run has completed and been verified. See [P2_REPORT.md](P2_REPORT.md) for the screening results, tradeoffs and proposed seed repeats.


The 25% screening archive has been audited. Selected metrics, quality scores, complementarity counts and exact correct rare intervals are versioned under reports/p2/kaggle-mixture25/. The report distinguishes this one-seed screen from the earlier three-seed comparisons.


## Repeat only the two missing 25% seeds

Import [02e_teacher_e_mixture25_repeat.ipynb](../../notebooks/02e_teacher_e_mixture25_repeat.ipynb), stop an older GPU session if necessary, enable Internet/GPU and run cells in order. The training cell runs only seeds 20261120 and 20261121 with the 25% mixture. Each is 30 epochs and writes a separate run under runs/p2-mixture25-repeat/.

The comparison cell combines those two runs with the seven verified results already in the repository. It writes analysis/per_seed.csv and analysis/summary.json inside the repeat directory, with all nine results and paired deltas for 25% versus each endpoint.

The final cell creates /kaggle/working/p2-mixture25-repeat-results.zip. Save the executed notebook version and download the ZIP into local runs/imports/. If interrupted, completed jobs are reused only at the same revision/config; an incomplete job is reported explicitly. A fresh output root reruns both new jobs.

The aggregate can also be regenerated locally after import:

```powershell
python src/acr/compare_e_mixtures.py --runs runs/p2-mixture25-repeat --output runs/p2-mixture25-repeat/analysis
```

These two full GPU runs are pending. The earlier single-seed screen and the two endpoint policies remain unchanged.
