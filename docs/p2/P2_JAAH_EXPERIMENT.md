# P2 JAAH pilot for both F and E

Status (20 November 2026): data inventory and implementation prepared locally. Full Kaggle training and any improvement claim are pending.

Local verification: 40 unit tests pass, including unchanged 24-channel outputs/RNG, 12-channel gradients into the shared trunk, mixed-channel batch rejection, external-label masking and the fixed replacement budget. A tiny training-only memorization check reached 8/8 correct Billboard events and 8/8 correct JAAH events after 84 alternating epochs (168 optimizer steps). See [overfit_check.json](../../reports/p2/jaah-inventory/overfit_check.json). This is an execution/learnability check, not validation performance or evidence that adding JAAH improves ACR.

## Selected data and research question

Use JAAH as the first external training source. Its pinned public release supplies frame-level chroma and aligned LAB annotations, allowing an experiment without acquiring additional audio. Keep JSIR as a promising rare-harmony candidate requiring matched audio, and GuitarSet as a later guitar-domain experiment requiring feature extraction. Do not combine all three in this first pilot.

Question: does auxiliary jazz supervision improve F, E, or their complementarity on the unchanged Billboard validation set? Run four separately trained models, then evaluate all four pairings. The earlier E-only notebook 02f is superseded by 02g for this experiment.

This tests a combined auxiliary-domain/representation intervention. It does not isolate the contribution of genre, amount of data or input information individually. It does not test the 12-channel branch on unseen JAAH recordings or establish cross-dataset generalization.

## Audited JAAH inventory

Source: [MTG/JAAH](https://github.com/MTG/JAAH/tree/6fec3f6dd1d60c21cf9af855ea173be38e6d7243), revision 6fec3f6dd1d60c21cf9af855ea173be38e6d7243. Source-file hashes, exclusions and cache hashes are recorded in [summary.json](../../reports/p2/jaah-inventory/summary.json).

- Downloaded metadata, LAB archive and 12-dimensional frame chroma for 113 recordings.
- Kept 103 recordings with 15,491 supported, nonempty merged events, including N.
- Excluded ten recordings because annotation/metadata duration and feature duration differ beyond the conservative tolerance. Exclusion does not prove which source is wrong. No timestamp shifts were inferred or applied.
- Retained 1,611 unsupported annotation intervals as masked regions; they are not N targets. Supported consecutive identical chord targets are merged; unsupported intervals remain breaks.
- Bare Harte roots such as `F` are canonicalized to `F:maj` for JAAH only. Altered, extended and explicit-interval labels outside the existing exact vocabulary remain masked. Billboard mapping remains unchanged.
- Verified 13 CSV columns (timestamp + 12 bins), finite nonnegative features, strictly regular timestamps, ordered nonoverlapping LAB intervals and duration consistency. Both feature clocks have a step of 2048/44100 seconds, approximately 46.44 ms.
- Applied per-frame L1 normalization to the JAAH 12-bin vector. Billboard keeps its existing separate normalization of its two 12-bin blocks.

| Quality | Usable merged events | Recordings |
|---|---:|---:|
| dim | 76 | 11 |
| aug | 17 | 8 |
| min6 | 237 | 28 |
| minmaj7 | 42 | 8 |
| hdim7 | 429 | 38 |
| dim7 | 617 | 46 |
| sus2 | 0 | 0 |
| sus4 | 0 | 0 |

See [qualities.csv](../../reports/p2/jaah-inventory/qualities.csv) for all classes and normalized-title composition counts, [inventory.csv](../../reports/p2/jaah-inventory/inventory.csv) for inclusion decisions and [symbols.csv](../../reports/p2/jaah-inventory/symbols.csv) for source-symbol counts. Raw symbol counts and merged event counts are different units.

Normalized titles were compared conservatively against every title in the Billboard index, including protected partitions. No exact normalized-title overlap was found. This is a metadata screen, not audio fingerprint verification or proof that all alternate-title/cover overlaps are absent. Only train_labeled and validation Billboard features/labels are loaded; hidden/test labels remain unused. JAAH is auxiliary training only, with no held-out JAAH performance claim. Duration agreement also does not establish perfect local alignment; further listening/alignment review would require matching audio.

## Model and controlled comparison

Each teacher has its own 24-to-48 Billboard projection and 12-to-48 JAAH projection feeding its own TCN and four heads. F emits frame predictions; E pools hidden states over gold intervals before classification. F and E share no parameters or checkpoints. Both control models contain an unused 12-channel projection, preserving matched architecture within each teacher's comparison.

The input branch does not reconstruct missing bass/treble information. Within each model the trunk/heads learn from both representations. Every batch contains only one representation. The 24-channel computation and initialization RNG are preserved, and auxiliary sampling does not perturb the Billboard CPU sampling RNG stream.

| Job | Within-source sampling | Total updates/epoch | JAAH updates/epoch | Checkpoint selection |
|---|---|---:|---:|---|
| F-control | 50% balanced frame windows | 217 | 0 | Billboard frame accuracy |
| F-mixed | 50% balanced frame windows | 217 | 54 | Billboard frame accuracy |
| E-control | 25% balanced events | 990 | 0 | Billboard complete-event accuracy |
| E-mixed | 25% balanced events | 990 | 248 | Billboard complete-event accuracy |

All four jobs start from scratch at seed 20261119, width 48, learning rate 0.001 and 30 epochs, with Adam, gradient clipping 5 and unweighted head losses. F uses window 512, context 31 on each side and batch size 16. E uses full gold events plus context and batch size 32. JAAH batches replace Billboard batches, never add updates. Rounded source replacement fractions are 54/217 (24.88%) for F and 248/990 (25.05%) for E.

F draws 3,471 windows per epoch in both arms; valid frame counts can differ because of padding and masked regions, and are recorded alongside per-source class exposure. E draws 31,679 events per epoch in both arms. JAAH's natural index draws use replacement; balanced draws use the teacher's existing quality/song sampler. Budgets are matched **within each teacher**, not between F and E. Differences between F and E cannot be attributed solely to frame versus event modeling.

The same 110 Billboard validation songs and 9,673 scored gold intervals are used throughout. Both F checkpoints are pooled using **mean frame probabilities**, never softmax of averaged logits. E uses saved event softmax probabilities. The four comparisons are F-control/E-control, F-mixed/E-control, F-control/E-mixed and F-mixed/E-mixed. Historical teachers remain context; the four-pair experiment uses the freshly trained teachers, not the old fixed F.

## Pair diagnostics

The comparison script checks matching seeds, data provenance, evaluation partitions, event order/targets and within-teacher configs/budgets. It reproduces saved component and complete predictions before scoring.

- individuals.csv and per-run quality tables: common event metrics and complete-chord precision/recall for all four teachers. F's frame metrics remain in its run directory.
- Each pair's complementarity.csv: both correct, F-only correct, E-only correct and both wrong, per quality and component.
- summary.json: complete-label agreement precision/coverage and complementarity for all events and the true-rare subset.
- curves.csv: individual F/E and agreed component precision, coverage, accepted counts and mean confidence at 0, 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99.
- predicted_classes.csv: per-predicted-component precision and accepted recall, including false positives from other qualities/N.
- true_qualities.csv: selected accuracy and coverage within each true quality, including ALL_RARE. This is not precision for predicting that quality.

These use the operational scope from the confidence audit: coverage denominator is all events; component selection requires the relevant confident chord gates, and accepted component predictions on true N count as wrong. Agreement confidence is the minimum of both head scores, not a joint calibrated probability. Thresholds are exploratory; no pseudo-label acceptance policy is fitted. Gold boundaries and reused validation remain limitations.

## Run on Kaggle

1. Import [02g_teachers_fe_jaah_pilot.ipynb](../../notebooks/02g_teachers_fe_jaah_pilot.ipynb), replacing the earlier E-only 02f notebook for this experiment.
2. Enable Internet and GPU. Stop an older GPU session if the concurrent-session limit is reached.
3. Run cells in order: source/data preparation, four short smoke jobs and pair analysis, two full F jobs, two full E jobs, then the four-pair analysis.
4. Save the executed notebook version and download `/kaggle/working/p2-jaah-fe-results.zip` into local `runs/imports/`.

The F smoke run uses a 50% replacement fraction to guarantee that its tiny two-batch epoch exercises the auxiliary path; full F training uses the configured 25%. Smoke scores are execution checks, not results. A completed F or E stage can be reused only at the same config, revision, source hashes and mode. Interrupted stages are protected and require inspection/fresh output paths; this is not mid-job checkpoint resumption.

The archive contains all four checkpoints, F frame logits and pooled event probabilities, E event logits, histories/exposure and pair diagnostics. It excludes raw datasets. F's saved frame outputs make this ZIP larger than the earlier E-only ZIP.

Local commands from the repository root (use fresh output paths):

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests
# First preparation only; skip when the verified cache exists.
.venv/Scripts/python.exe src/acr/prepare_jaah.py --download --report runs/p2-jaah-preparation
# Add --smoke and use a separate root for local CPU execution checks.
.venv/Scripts/python.exe src/acr/experiment_jaah_f.py --output runs/p2-jaah-fe-pilot/F --device cuda
.venv/Scripts/python.exe src/acr/experiment_jaah.py --output runs/p2-jaah-fe-pilot/E --device cuda
.venv/Scripts/python.exe src/acr/compare_jaah_pairs.py --runs runs/p2-jaah-fe-pilot --output runs/p2-jaah-fe-pilot/analysis
```

Raw inputs remain in data/raw/jaah/ and normalized cache in data/processed/p2-jaah/. Both are ignored by Git. Smoke artifacts are in runs/p2-jaah-fe-smoke/. Human-readable docs stay in docs/; selected machine evidence stays in reports/.

## Decision after training

Compare complete-event accuracy, macro quality recall, rare macro recall, per-quality precision and correct rare-event/song counts. Inspect whether rare successes add information beyond F, and whether maj/min or sus classes regress. Re-run confidence/coverage analysis on saved outputs before accepting any pseudo-labeling policy.

An increase in rare recall alone is insufficient if it comes from many false positives or a large common-chord regression. A promising single-seed result must be repeated with matched seeds before changing either preferred teacher or the preferred pair. A negative result should be retained and interpreted as evidence about this specific mixture, mapping and input-adapter setup, not proof that external data cannot help. No threshold, dataset or model is selected based on hidden/test outcomes.


## Result status — 20 November 2026

The primary-seed full GPU pilot is complete and verified. See [P2_REPORT.md](P2_REPORT.md) for results: E accuracy improves but rare recall decreases; F accuracy decreases with only one extra correct rare event. No mixed model or pseudo-labeling rule is promoted. Earlier pending-GPU statements describe the setup stage. Further seeds and source-transfer diagnostics remain unrun.
