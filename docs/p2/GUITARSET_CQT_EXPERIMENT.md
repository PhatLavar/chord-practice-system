# GuitarSet CQT–TCN audio pilot

Status: full GPU pilot imported and verified; training-only memorization diagnostic completed. Neither teacher is promoted. This is a GuitarSet-only source-learning pilot for fresh F and E. It does not yet measure improvement on Billboard or reproduce BTC's architecture/benchmark.

## Literature and scope

The [BTC author configuration](https://github.com/jayg996/BTC-ISMIR19/blob/master/run_config.yaml) and [preprocessor](https://github.com/jayg996/BTC-ISMIR19/blob/master/utils/preprocess.py) motivate 22,050 Hz, 144 CQT bins, 24 bins/octave and hop 2,048. C1 is explicit at 32.70319566257483 Hz. We reuse our tested residual TCN and factor heads as the first audio control, not the BTC Transformer. A previous small Transformer experiment is not a substitute for an author-baseline reproduction.

Differences from BTC: modern pinned librosa 0.11.0/soundfile 0.13.1; full-recording CQT rather than independently transformed overlapping ten-second clips; frame timestamps assigned by interval membership rather than maximum overlap over a frame bin; no pitch augmentation; natural sampling; scalar normalization weighted by every training feature value; three progression groups rather than the published pop-song folds. Log magnitude is ln(abs(CQT)+1e-6). Resampling uses soxr_hq, tuning=0, constant centered padding. Parameters follow BTC, but results must be described as CQT–TCN.

## Data and folds

Read the pinned official microphone/annotation ZIPs directly. The fold-0 cache contains 119 train and 120 validation recordings; the 119 outer-test recordings and two known timing errata are not loaded or extracted. Verify archive hashes and the committed manifest. Cache full log-CQT matrices independently per recording. Scalar mean/std is fitted only on train audio, including its unlabeled regions, then applied to validation. Padding supplied to the model is zero in normalized feature space. Cache hashes, original WAV hashes, library versions and preparation/label code hashes are recorded.

Source targets remain the performed labels. Only complete-supported labels train either model; OOV and unlabeled frames are -100. Partial root/gate proposals are not enabled. Evaluation retains original valid OOV Harte labels, so an unsupported chord is not silently converted to N. No-chord support is absent in this source's performed chord labels: this pilot cannot establish a learned silence/no-chord detector. Gaps outside labels are unknown X during scoring.

The original WAV duration can be up to about 10 microseconds shorter than the decimal annotation endpoint. Scoring clips an endpoint only within one original 44,100-Hz sample; larger mismatches still fail. Original reference intervals remain saved. No onset shifting or annotation repair is applied.

## Models and optimization

Both teachers are initialized fresh at seed 20261119 with 144 input channels, width 48, five residual dilations 1/2/4/8/16 and gate/root/triad/extension heads. Existing 24-channel defaults and checkpoints remain compatible. The hop changes the temporal context: 31 frames is about 2.88 seconds on each side. No old checkpoint is loaded into the new front end.

Use Adam at 0.001, batch 16, gradient clipping 5 and 30 epochs. F uses natural non-overlapping 108-frame target windows plus context; there are 405 training windows (26 updates/epoch). E uses naturally sampled complete-supported gold events; there are 1,326 events (83 updates/epoch). Equal epochs do not mean equal update/data budgets, so this is not a clean causal architecture comparison. The best checkpoint for each model is selected by its own validation duration-weighted Tetrads, with earliest epoch winning ties. Validation is the only scored partition; outer-test evaluation remains a separate future gate.

## Evaluation and exported evidence

F produces an unconstrained frame timeline across the full recording. E is given gold event boundaries, including valid OOV events at inference. E therefore measures oracle event classification, not end-to-end segmentation. For both, Root/MajMin/Triads/Tetrads use mir_eval 0.8.2 and aggregate correct/eligible duration, with explicit excluded duration. Whole-reference coverage is preserved; missing label regions are X. This is a stronger scope than the earlier exported-event subset rescoring, but is still not the published BTC dataset/protocol.

A separate supported-only oracle event accuracy is also recorded: F pools mean probabilities on gold intervals, E uses pooled-feature logits. Per-quality counts in quality_supported_scope use only complete-supported references and must not be interpreted as full-vocabulary precision. Full predicted LAB files, frame/event logits, validation references and per-song metric denominators support later analysis. Do not compare E's gold-boundary score directly with F's frame score as equivalent tasks.

## Run on Kaggle

Import [02h_guitarset_cqt_teachers.ipynb](../../notebooks/02h_guitarset_cqt_teachers.ipynb). Enable Internet and GPU, then run cells in order:

1. Clone/update source, install requirements-audio.txt, run tests.
2. Download/verify official ZIPs (about 696 MB total).
3. Prepare four-recording smoke cache and run two short CPU jobs.
4. Prepare the full train/validation fold-0 CQT cache.
5. Train fresh F and E for 30 epochs each on GPU.
6. Inspect metrics and download /kaggle/working/p2-guitarset-cqt-results.zip to local runs/imports/.

Complete stages may be reused only at matching config, revision and source hashes. Incomplete stages are preserved and rejected; there is no mid-epoch resume. The ZIP contains runs/p2-guitarset-cqt-fold0/ with both teachers and validation_references.json; raw audio/CQT cache stay out of the archive and Git.

Local reproduction:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-audio.txt
.venv/Scripts/python.exe src/acr/prepare_guitarset_cqt.py --output data/processed/guitarset-cqt-fold0-smoke --smoke
.venv/Scripts/python.exe src/acr/train_guitarset_cqt.py --cache data/processed/guitarset-cqt-fold0-smoke --output runs/p2-guitarset-cqt-smoke --smoke
# Fresh full preparation and GPU output paths:
.venv/Scripts/python.exe src/acr/prepare_guitarset_cqt.py --output data/processed/guitarset-cqt-fold0
.venv/Scripts/python.exe src/acr/train_guitarset_cqt.py --cache data/processed/guitarset-cqt-fold0 --output runs/p2-guitarset-cqt-fold0 --device cuda
```

## Verification and decision after the run

All 55 tests pass with audio dependencies installed, including a known-tone CQT pitch/clock check, train-only normalization, OOV/boundary handling, 24-channel initialization compatibility and gradients through both 144-channel teachers. The full 239-recording cache has been prepared locally; both models can run forward and score all 120 validation recordings. Four-recording smoke training completes with finite losses. Smoke metrics are not performance results. Evidence is in reports/p2/guitarset-cqt/verification.json.

Inspect the full run's optimization, per-quality support, duration-weighted scores and source-fit failure modes before changing mixture ratios or launching all folds. Only three underlying progressions makes generalization difficult and estimates variable. A successful GuitarSet source pilot still requires a controlled added-source experiment to establish benefit for Billboard and an author-model baseline before a literature-parity claim.


## Primary pilot result status

The 30-epoch fold-0 GPU results have been imported and verified. F validation Tetrads is 7.68%; E oracle Tetrads is 8.71%. Neither is promoted. See P2_REPORT.md for support-shift, training-fit and component diagnostics. The earlier pending-GPU status describes setup. A LAB timestamp serialization defect was fixed without changing measured NPZ metrics.

## Training-only memorization diagnostic — 21 November 2026

Run locally with the verified full cache:

```powershell
.venv/Scripts/python.exe src/acr/check_guitarset_memorization.py
```

The command requires a fresh output directory (default runs/p2-guitarset-cqt-memorization). It selects two events per available quality from fold-0 training recordings only, restricted to 4–108 frames, preferring different roots and recordings for the second event. Selection is deterministic and does not depend on prediction scores. It retains the original train-fitted normalization and actual neighboring audio as context. Validation/test examples are never selected or scored; the existing cache loader verifies the full train/validation cache.

Both fresh width-48 models use seed 20261119 and Adam 0.001, at most 500 full-batch updates on the same 18 crops. F supervises only frames inside selected gold events; E pools features over those intervals. This deliberately concentrates supervision and is not the natural full-corpus training distribution. Check every ten updates: pass requires every pooled event correct, and F additionally requires at least 95% frame accuracy. Stop each teacher at the first checked pass. Save histories, event selections, predictions, code/cache hashes and local checkpoints. This is a learning sanity check, not model selection for deployment.

Both pass at update 40. F reaches 98.81% selected-frame accuracy and 18/18 pooled events correct; E reaches 18/18 events correct. Nine represented qualities are maj, min, sus2, sus4, maj6, min6, 7, min7 and maj7. N, dim, aug, hdim7, dim7 and minmaj7 are not covered. Evidence is reports/p2/guitarset-cqt/results/memorization.json. No new Kaggle run is needed for this diagnostic.
