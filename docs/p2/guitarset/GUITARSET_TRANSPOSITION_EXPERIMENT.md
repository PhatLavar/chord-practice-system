# GuitarSet CQT transposition ablation for F and E

## Motivation and primary sources

The natural player-held-out control shows E fitting supported training events well (87.43% complete-event accuracy) with a large validation gap (38.48%). F has both incomplete training fit and a gap. The first augmentation control tests transposition diversity rather than changing architecture, loss weights and data simultaneously.

Sources inspected on 21 November 2026:

- [BTC author preprocessing](https://github.com/jayg996/BTC-ISMIR19/blob/master/utils/preprocess.py) applies audio pitch shifts -5 through +6 semitones and transposes chord IDs. Its waveform/Rubber Band implementation is not used here.
- [ChordFormer author training code](https://github.com/mwaseemrandhawa/ChordFormer/blob/main/confor_head16.py) uses a CQTPitchShifter and structural target shifting. Its resolution, shift schedule and model differ from this experiment.
- [Doai, MIREX 2025 system description, section 3.2](https://futuremirex.com/portal/wp-content/uploads/2025/audio-chord-estimation/MD1.pdf) explicitly describes shifting CQT spectrograms by -5 to +6 semitones with corresponding labels. It also uses other augmentations; they are excluded from this single-change control.
- [Improving Balance in Automatic Chord Recognition with Random Forests, EUSIPCO 2022](https://eurasip.org/Proceedings/Eusipco/Eusipco2022/pdfs/0000244.pdf) distinguishes pitch-class balancing by pitch augmentation from balancing chord qualities. Transposition does not generate new rare qualities.

This is an implementation of a literature-motivated feature-space ablation, not a reproduction of those architectures or published scores. CQT shifting approximates pitch transposition; it does not create a real new performance or reproduce waveform pitch-shifting artifacts/timbre.

## Frozen comparison

Control: the imported 02i natural jobs at b64c8dd, F selected epoch 20 and E epoch 30. Use configs/p2/p2-guitarset-cqt-transpose.json for the new arm. It must equal the original player config except for augmentation and an explicit pointer to the original cache config; training rejects other config changes through this path.

Both new teachers start fresh at seed 20261119, with width 48, unchanged CQT, natural sampling, unweighted losses, Adam 0.001, batch 16, gradient clipping 5 and 30 epochs. The same train players 00–03 and validation player 05, original train-only normalization and update counts (F 47/epoch, E 119/epoch) are retained. There is no dataset expansion. Existing test-exposure/repertoire-overlap limitations remain.

For each training example, independently draw one integer semitone shift uniformly from -5 through +6, including zero. At 24 CQT bins/octave, one semitone is two frequency bins. Positive shifts move energy toward higher-frequency bins. Shift the complete-label root modulo 12, preserving quality, gate, N=0 and ignored labels=-100. Shift the whole sample/context by the same amount. Labels and timing are never shifted independently.

Do not roll/wrap the frequency axis. Discard energy moved outside the finite CQT range and fill exposed bins with the normalized log-amplitude floor: (ln(1e-6) - training mean) / training std. Temporal padding remains zero in normalized coordinates. Original cached real frames are checked not to be all zero so padding detection is unambiguous. The existing 144-bin range imposes edge information loss; this is a limitation, not additional observed low/high-frequency audio.

A dedicated RNG seeded seed+1 chooses augmentation shifts without consuming the global RNG used by model initialization and shuffled data loading. No augmentation changes dataset length, batches or optimizer updates. Histories record actual per-epoch shift counts. Validation and per-epoch training-fit evaluation always use clean original features and labels. There is no test-time augmentation or confidence acceptance policy.

## Running on Kaggle

Import notebooks/p2/guitarset/02j_guitarset_transposition_teachers.ipynb. Enable Internet and GPU and run cells in order. It verifies/reuses the original 02i CQT cache using the original config; the training config adds only the augmentation. It executes two short CPU smoke jobs, then fresh full F/E jobs. Completed jobs are reused only at matching source/config/revision; partial outputs are preserved and rejected.

Download p2-guitarset-cqt-transpose-results.zip into runs/imports/ and save the executed notebook version. The run root is runs/p2-guitarset-cqt-transpose; original 02i jobs are preserved. No extra audio feature generation is needed if the original player cache remains available in the session.

Local smoke command:

```powershell
.venv/Scripts/python.exe -m acr.training.guitarset --config configs/p2/p2-guitarset-cqt-transpose.json --cache data/processed/guitarset-cqt-player-smoke --output runs/p2-guitarset-transpose-local-smoke --smoke
```

## Evaluation and decision rules

Primary outcome is validation duration-weighted Tetrads, compared within each teacher against 02i; also report Root/MajMin/Triads, supported-event accuracy, per-quality errors and clean training/validation curves. F has frame predictions and E gold boundaries, so compare augmented F to control F and augmented E to control E. Keep root/extension precision and recall visible; no rare-only promotion criterion is introduced.

This is a single-seed screening, not a repeatability claim. A gain must survive matched-seed repeats before promotion. If it hurts F while helping E, report the divergent outcomes instead of choosing one shared policy prematurely. No benefit is assumed: transposition may reduce root/key shortcuts but does not supply new players, timbres, chord qualities or seventh evidence that is absent from the source.

## Local verification

All 59 tests pass, including a known-tone pitch-direction check, every supported root/quality transposition, positive/negative/no shift, finite frequency edge fill, temporal padding and N/OOV preservation. Both two-epoch CPU smoke jobs finish with finite losses and populated augmentation shift counts. Histories include clean train/validation metrics; notebook code cells compile. Evidence is reports/p2/guitarset-cqt/transposition/verification.json. Full 30-epoch GPU training remains pending.

## Full GPU results imported — 21 November 2026

Both 30-epoch jobs at d7c834bb9d6ec2325ffccc7f496f40dce61b8389 have completed and been verified. Full-timeline rescoring from NPZ reproduces the saved results; original annotation/reference lists, data/normalization identity with control, code hashes, checkpoint selection, positive LAB durations and per-epoch augmentation counts were checked. Replaying the dedicated RNG reproduces every recorded shift histogram. F/E retain 47/119 updates and 740/1,894 samples per epoch. No validation augmentation or reserved-player inference is present.

| Teacher | Control Tetrads | Augmented Tetrads | Delta (percentage points) | Selected epoch |
|---|---:|---:|---:|---:|
| F | 24.45% | 25.51% | +1.06 | 30 |
| E, gold boundaries | 29.41% | 31.46% | +2.05 | 28 |

Supported event accuracy is F 38.94% (169/434) and E 40.55% (176/434), versus 34.56%/38.48% for control. Gains are not uniform: E duration Triads falls from 37.06% to 36.68%; both models predict maj more frequently. Both lose their remaining complete maj7 successes. Full counts are in transposition/per_quality_comparison.csv.

Clean training fit is much lower with augmentation: selected-checkpoint training Tetrads is F 28.71%, E 34.31%, versus 36.58%/59.95% for control. Local frozen-checkpoint CPU evaluation exactly reproduces those Kaggle training scores. The smaller train/validation gap is partly due to reduced fitting; it is not evidence that chord-quality reliability has broadly improved. Neither augmented model correctly predicts raw 6, 7 or bb7 extension on supported validation events. Higher overall extension accuracy is dominated by none.

Decision: retain the original natural control and this single-seed candidate. Do not promote or claim rare-chord improvement. Next, use matched new seeds for both control and augmentation before changing shift range, learning rate, epochs or architecture. Compare within teacher, report mean/variation and lost/gained events, and preserve all seeds rather than choosing the best one. Full repeats have not run. Evidence is reports/p2/guitarset-cqt/transposition/{results,diagnostics,training-fit}/ and comparison.json.
