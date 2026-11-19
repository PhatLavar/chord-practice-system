# P1 report · Kaggle frame teacher F baseline

The first full frame-teacher baseline completed 30 epochs on Kaggle CUDA. The selected checkpoint is epoch 27 with 57.11% complete-chord frame accuracy on 110 validation songs. The two-song training check passed at 95.49%. These are initial pilot results.

## Provenance and verification

Both runs used commit 8ec616c78c1466da62ad958326a175c5d3836c10 with a clean worktree, PyTorch 2.11.0+cu130, NumPy 2.1.3 and a 37,225-parameter model. The baseline trained on exactly the 367 labeled representatives and evaluated on the 110 validation representatives from the P0 manifest. The two-song check trained and evaluated on 0003 and 0006.

The returned archive contained both checkpoints, run records, histories, predictions and component logits. All five source-code hashes match the committed source. Recomputing the confusion matrix from the 110 prediction files reproduced the saved matrix, frame accuracy and macro chord recall. Of 514,764 validation frames, 58,718 were masked as OOV, unknown or uncovered, leaving 456,046 scored frames. 260,458 of those were classified correctly.

The archived run's source files use LF line endings; a Windows working copy may use CRLF. Verification compared against the Git objects for the recorded commit.

[Recorded checks and archive checksum](../../reports/p1/kaggle-baseline/verification.json) · [Run configuration and source hashes](../../reports/p1/kaggle-baseline/run.json)

## Baseline results

| Measure | Result |
|---|---:|
| Whole-chord frame accuracy | 57.11% |
| Macro recall over observed whole-chord classes | 19.38% |
| Best epoch / total epochs | 27 / 30 |
| Scored validation frames | 456,046 |
| Peak PyTorch CUDA allocated memory | 26,010,112 bytes |

Validation accuracy increased from 42.97% at epoch 1 to 54.40% at epoch 10 and 56.36% at epoch 20. Epoch 30 reached 57.03%. Summed per-head training loss fell from 3.457 to 1.578. The recorded epoch loops totaled about 53.6 seconds; this excludes notebook setup and data preparation. Peak allocated CUDA memory measures PyTorch tensors, rather than all GPU memory held by the runtime.

[Metrics](../../reports/p1/kaggle-baseline/metrics.json) · [Epoch history](../../reports/p1/kaggle-baseline/history.json)

## Quality-level diagnosis

Each recall below requires the complete chord, including its root, to be correct. Scores are pooled by quality across validation frames.

| Quality | Scored frames | Complete-chord recall |
|---|---:|---:|
| maj | 256,881 | 79.42% |
| min | 68,723 | 47.91% |
| 7 | 44,897 | 13.10% |
| min7 | 37,117 | 15.43% |
| maj7 | 9,154 | 5.62% |
| sus4 | 7,322 | 0% |
| maj6 | 3,185 | 0% |

All seven P0 rare qualities have validation support and zero correct complete-chord frames in this run. These are aug (214 frames), dim (678), dim7 (90), hdim7 (292), min6 (306), minmaj7 (98) and sus2 (1,371). These counts are frame support; rare-quality event recall and independent-song support require event-level evaluation.

The model frequently simplifies seventh chords. G:7 was predicted as G:maj on 5,152 scored frames, A:7 as A:maj on 5,135, E:min7 as E:min on 5,044 and A:min7 as A:min on 4,872. The high overall score is influenced by the many major and minor frames.

[All quality results, including precision](../../reports/p1/kaggle-baseline/per_quality.csv) · [Per-song counts](../../reports/p1/kaggle-baseline/per_song.csv)

## Timeline interpretation

The exported example for recording 1125 contains C:7, D:7 and G:7 reference chords. It predominantly predicts C:maj, D:maj and G:maj. Thus, broadly matching transition positions can coexist with incorrect chord qualities. Vertical distance between numeric class IDs has no musical meaning. The numeric plot should be interpreted with the class vocabulary in run.json.

## Status and next experiment

P1's initial implementation, two-song overfit check, full baseline and validation audit are complete. The rare-quality objective remains unmet. The next controlled P1 experiment should evaluate training-only pitch transposition and class balancing, while preserving this baseline, manifest, vocabulary and validation protocol. Make each change a named experiment with its own output directory.

Before P2 teacher comparisons, define and implement reference-event evaluation and export per-quality song support. Those diagnostics will distinguish errors near transitions from errors within stable chord intervals. The test partition remains reserved for final evaluation. This run provides no evidence yet about teacher complementarity or gains from pseudo-labeling.

For local inspection, the imported runs are runs/p1-overfit-kaggle/ and runs/p1-baseline-kaggle/. The original archive is retained under runs/imports/p1-results.zip. These large outputs are ignored by Git; the selected tables and run metadata above are versioned.
