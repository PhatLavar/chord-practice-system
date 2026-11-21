# GuitarSet player-held-out development experiment

## Question and fixed protocol

Can fresh CQT–TCN F/E learn from multiple progression groups and generalize to a held-out player? This follows the failed progression-held-out pilot and successful 18-event memorization check. It is a separate development experiment, not a replacement score or a paper benchmark.

Frozen manifest: splits/guitarset-player-v1.csv. Player IDs are assigned deterministically before this run: 00–03 train (240 recordings), 05 validation (60), 04 reserved (58 usable recordings). Two known timing errata remain excluded. All three progression groups are represented in each usable partition. Comp/solo recordings from each player remain together. No reserved-player audio is loaded by this experiment. The manifest is derived from fold-0 inventory rows, preserving original progression signatures and metadata, then assigning roles by player ID.

Important scope: progressions, repertoire and keys overlap across players. This tests unseen-player performance on known repertoire, not unseen-song/progression generalization. Some original progression-test recordings are now training data; the original test cannot remain a pristine future adaptive benchmark after this run. The reserved player also has historical exposure through earlier validation. Do not claim a new independent final test. The original manifest, checkpoints and reported results remain unchanged.

## Training and evaluation

Use configs/p2-guitarset-cqt-player.json and the existing preparation/training scripts. Only the manifest and additional training diagnostics change. CQT, fresh seed 20261119, width 48, natural sampling, complete-supported targets, Adam 0.001, batch 16, 30 epochs and validation-Tetrads checkpoint selection remain as in 02h. Fit scalar normalization on this experiment's training players only. New cache/run paths prevent accidental reuse of progression-fold statistics or checkpoints.

Every epoch now records both training and validation duration metrics, supported-event accuracy and per-quality counts, in addition to the four training head losses. Training diagnostics do not select checkpoints. F has predicted frame intervals; E uses gold boundaries. Equal epochs still imply different update budgets. More training recordings and a different validation distribution mean an improved score cannot be attributed solely to the split or architecture.

Support audit: reports/p2/guitarset-cqt/player-split.json. Training includes hdim7 and minmaj7, unlike the previous fold, but dim7 remains absent while validation contains two events. Some other labels remain extremely sparse (min6 has two training events). All progression groups in training does not imply complete label coverage. No oversampling, augmentation or partial-target training is introduced in this control.

## Kaggle execution

Import notebooks/02i_guitarset_player_teachers.ipynb. Enable Internet and GPU, then run cells in order. It installs dependencies and runs tests, verifies official archives, executes a four-recording smoke run, prepares 300 train/validation recordings, then trains F and E. Evaluating the training set every epoch adds runtime.

Download p2-guitarset-cqt-player-results.zip into runs/imports/. Save the executed notebook version. The archive includes both checkpoints, provenance, normalization, histories with training/validation curves, predictions and metrics. Incomplete output directories are preserved and rejected; do not overwrite previous results.

Do not promote either model before the full run is imported and verified. Inspect training fit, generalization gap, per-quality support and F/E differences first. This experiment does not measure added-source benefit on Billboard.

## Local verification

All 57 tests pass. The new split test checks unique recording IDs, disjoint player roles, all three progression groups, errata exclusion and manifest-specific cache selection. Four-recording preparation and two-epoch CPU smoke runs complete for both F and E; histories contain training and validation diagnostics. Notebook code cells compile. Evidence: reports/p2/guitarset-cqt/player-verification.json. Smoke scores are not performance evidence. Full 300-recording preparation and 30-epoch GPU jobs remain for Kaggle.
