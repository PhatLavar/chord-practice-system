# B0 BTC source compatibility and I/O contract

Status: author source pinned and adapter implemented. This document covers local compatibility and a training-only fitting diagnostic, not B1 benchmark training. TE-DS compatibility and admission of additional raw-audio corpora remain separate B0 work.

## Source and architecture

Use [BTC author commit 2682317](https://github.com/jayg996/BTC-ISMIR19/commit/2682317be668032e6e4b269ded36adaa2ad57df0). Minimal MIT-licensed source is in src/acr/vendor/btc/ with LICENSE, UPSTREAM.md and UPSTREAM.json. Module class ASTs were compared to source at the pin; all model class computations match. Compatibility changes are limited to package imports, removal of a standalone demonstration/unused YAML import, and np.float → np.float64.

configs/baselines/btc-b0.json selects the author eight-layer, width-128, four-head configuration with 144 input bins, 108 frames, original dropout values 0.2 and a flat 170-label readout. probs_out=True requests raw logits; it is not a probability calibration choice. Unused author output-LSTM parameters are retained for state-dict compatibility. This is the author BTC architecture, not the earlier custom two-layer project Transformer. No pretrained checkpoint is loaded.

## Label contract

The adapter's project-to-BTC mapping is checked against all 168 supported chord outputs of the pinned author's isolated convert_to_id_voca method. Integer IDs are not shared implicitly.

| Meaning | Project cache | BTC adapter |
|---|---:|---:|
| Supported complete chord | 1 + root×14 + project quality index | root×14 + author quality index |
| N, annotated no chord | 0 | 169 |
| Ignored target / unsupported target | −100 | −100, ignored by diagnostic loss |
| Predicted unknown chord | No learned unknown class in old TCN | 168, exported semantically as X; maps to project −100 for diagnostic interchange |

Author quality order is min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4. Tests round-trip all 168 supported labels plus N, and distinguish unknown prediction from N. Input labels with invalid IDs/dtypes are rejected.

The current GuitarSet complete-label policy masks OOV rather than treating it as N. Therefore the BTC unknown output has no explicit positive OOV training targets in this compatibility diagnostic. Its presence preserves output geometry but does not establish a learned unknown detector. A future native-unknown author-recipe condition needs a separate supervision specification. Project −100 also covers unlabeled/padded frames, so a converted predicted unknown is not itself a scoring-eligibility mask: downstream evaluation must keep the original reference mask and count/score the prediction as X, not silently drop the event.

## CQT and time contract

Load the existing verified player cache under configs/p2/p2-guitarset-cqt-player.json. Normalization uses training players 00–03 only; raw archive hashes, recording IDs and cache contents are checked by the existing loader. Actual frame timestamps are n×2048/22050 seconds. Frame targets use start-inclusive/end-exclusive membership in performed-label intervals.

BTCFrame accepts B×144×108 normalized input and returns B×170×108 raw logits. It does not add the TCN's 31-frame context. For full-track I/O checks, predict_frames tiles consecutive non-overlapping 108-frame windows, pads only the last window with zero in normalized coordinates, trims padded outputs and preserves the original frame count/order. There is no key-padding mask in the author architecture, so padded tail frames can influence real predictions in that last window. This policy must be reported in a benchmark recipe; it is not asserted to match the original author's inference preprocessing.

Feature dimensions alone do not establish original-recipe equivalence. Our cache uses full-recording librosa CQT and pinned modern libraries with its own train-only scalar normalization. The author's preprocessing/audio-window and timestamp recipe must be audited separately before a published-protocol reproduction claim. B0 uses the project cache specifically to test whether the pinned architecture can consume the admitted source.

## Local command

```powershell
.venv/Scripts/python.exe -m acr.experiments.baselines.btc_check
```

Requires the existing full data/processed/guitarset-cqt-player cache. Output defaults to a fresh runs/b0-btc-compatibility directory; an existing output is rejected. A different path can be selected with --output. The diagnostic verifies vendor hashes, chooses four train-only natural windows with distinct dominant supported qualities, requires at least 16 scored chord frames per window, and uses a fixed full batch. No validation score or test recording selects the windows or stop point.

Train from seed 20261119 with the full BTC, original dropout settings and Adam 0.0001, at most 300 updates. Score clean selected-frame complete-chord accuracy every ten updates (and after the first); stop at the first checked value ≥95%. Ignore cache −100 targets in cross entropy; an all-ignored loss is zero rather than NaN. Check finite gradients and full-track tiling on one selected training recording. Save the diagnostic checkpoint, history, source/cache hashes and selection metadata. This is deliberately concentrated memorization, not natural full-corpus training or a validation checkpoint-selection rule.

No B1 learning rate, stopping schedule, corpus mixture or GPU-memory claim follows from this diagnostic. No Kaggle training notebook is needed until the remaining B0 model/input gates are specified.

## Verified local result — 21 November 2026

The full 3,036,842-parameter BTC passes forward/backward and fixed-window fitting on CPU with torch 2.15.0+cpu and NumPy 2.5.5. Four 108-frame windows from three player-00 training recordings contain 240 scored frames across six qualities (maj7, maj, min7, min, min6 and 7); other frames are ignored. The first checked passing result is 228/240 = 95.00% at update 160. Accuracy and stopping are computed from integer correct/total counts to avoid float32 rounding at the threshold. Full-track tiling returns finite logits with shape 170×241, exactly matching that recording's real frame count.

Reloading the saved checkpoint reproduces 228 correct frames. Per-quality correct/total is maj 19/20, min 39/39, min6 16/16, 7 5/16, min7 59/59 and maj7 90/90. The global 95% gate permits concentrated dominant-seventh errors; it is not a requirement of 95% for every quality. Preserve this limitation rather than interpreting the fit check as reliable chord recognition.

All 66 tests pass. New checks cover label round trips/special labels, ignored-target loss/gradients, vendor identity, full model forward/backward, frame tiling/order/tail trimming, directional attention masks and the author's normalization convention. The BTC-specific five-test subset also passes after removing an unnecessary scalar-conversion warning in a test.

Evidence: reports/baselines/b0-btc/verification.json includes selection, quality counts, training history, source/cache/checkpoint hashes and environment. The local diagnostic checkpoint is in ignored runs/b0-btc-verified/. These 240 training frames are not a validation benchmark, and this source has no N supervision in the selected windows. No GPU runtime or recognition on new recordings has been measured. BTC compatibility is one completed part of B0, not completion of B0/B1 as a whole.
