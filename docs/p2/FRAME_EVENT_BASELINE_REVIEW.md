# Selecting stronger frame and event audio baselines

Review date: 21 November 2026. Status: paper and source-code review completed; upstream models have not been installed, executed or trained locally. Findings support a reproduction plan, not a claim that either model improves GuitarSet.

## Recommendation

Select **BTC large-vocabulary as the first frame-based reproduction anchor** and **TE-DS as the first genuine event-sequence baseline, followed by pTE-DS**. Keep **ChordFormer as the next modern structural frame baseline**. This ordering prioritizes an auditable implementation path and distinguishes a trusted comparison baseline from the newest architecture. It does not rank their expected GuitarSet accuracy without experiments.

| Direction | Selected baseline | Reason for selection | Main adaptation cost |
|---|---|---|---|
| Frame | BTC, 170-label author variant | Established audio Transformer, author implementation, close to current CQT geometry | Label IDs, legacy dependencies and exact preprocessing/window inference |
| Event | TE-DS, then pTE-DS | Explicit audio-to-time/root/quality token sequence, author code linked from the paper | Tokenizer, time grid, free-running decoder and source/template discrepancies |
| Modern frame follow-up | ChordFormer | Conformer plus structural chord outputs | New higher-resolution CQT cache, structural vocabulary and legacy training wrapper |

The small custom Transformer previously tested in this project is not a reproduction of any of these architectures. Its result cannot settle whether a published Transformer/Conformer is suitable.

## Frame anchor: BTC

The [BTC author repository](https://github.com/jayg996/BTC-ISMIR19) contains model, preprocessing and training code, including a large-vocabulary mode. Its published-data audio must be obtained separately; available annotations do not supply those recordings.

The [author configuration](https://github.com/jayg996/BTC-ISMIR19/blob/master/run_config.yaml) specifies 144-bin CQT, 24 bins/octave, 22,050 Hz, hop 2,048, 108-frame inputs, eight layers, width 128, four heads and dropout settings. The active template defaults to 25 labels; the 170-label variant is commented and must be explicitly selected. [btc_model.py](https://github.com/jayg996/BTC-ISMIR19/blob/master/btc_model.py) has separate forward/backward attention branches, convolutional positionwise blocks and positional timing signals. A generic TransformerEncoder with the same layer count is not an exact substitute.

Our current cache matches several dimensional/time parameters, making BTC the lowest front-end migration cost. However, full-recording extraction, normalization, crop generation and inference differ from the original recipe. Reusing the current cache produces a clearly named GuitarSet adaptation, not a published benchmark reproduction.

The [author label conversion](https://github.com/jayg996/BTC-ISMIR19/blob/master/utils/chords.py) orders qualities differently from this project and uses a different placement for special labels. Reuse symbols as the interchange representation, not integer IDs. The project currently masks unsupported complete targets; adding an explicit unknown output is a separate supervision choice. Resolve it before training. A first faithful flat-label baseline should retain the author's readout; a later four-head BTC adaptation is a separate ablation, not the same model.

## Event target: TE-DS and pTE-DS

[Kim and Park, ICASSP 2026](https://arxiv.org/html/2604.24386v1) predicts time-aligned chord tokens autoregressively. TE-DS splits chord tokens into root and quality; pTE-DS adds chord-similarity encoder pretraining. The paper reports five-fold Tetrads of 72.0% for TE-DS and 73.2% for pTE-DS, versus 65.5% for BTC in its 471-song setup. These are literature context, not predicted GuitarSet outcomes or directly comparable scores to our validation. The paper also reports segmentation quality, which must accompany free-running event evaluation.

The [author repository](https://github.com/KimLeekyung/ACR_seq2seq) is public and linked by the paper, with full_train, pre_train and base code. Its README is minimal; it is not a ready-made Kaggle training package. Start with TE-DS without encoder pretraining to isolate sequence modeling, then measure the pretraining increment separately.

### Source findings that must be resolved

- [full_train/hparams.yaml](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/full_train/hparams.yaml) configures four encoder and four decoder layers, width 512, eight attention heads, 144 feature dimensions, 44,100-Hz audio and hop 4,410. Its crop length is 25.6 seconds. Existing 144-bin arrays are not automatically equivalent: the current hop duration is about 92.88 ms, while this model's feature/time recipe is 100 ms.
- [dataset.py](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/full_train/dataset.py) defines 257 time, 13 root, 15 quality and three special tokens, totaling 288. It has one N quality token and no separate unknown-quality token in that list. The paper describes 289 SPLIT tokens including unknown handling. Define and test a single vocabulary contract before running; do not relabel GuitarSet OOV as N or assume the paper's token count matches this template.
- The full-training template has apply_log=False, whereas [pre_train/hparams.yaml](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/pre_train/hparams.yaml) has apply_log=True and drop_last_frame=True. [SignalProcessor](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/base/signal_processor.py) implements a convolutional waveform CQT. These are observed source/template differences to investigate, not runtime-verified defects.
- [model.py](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/full_train/model.py) provides masked autoregressive decoding. Some token/time constraints are conditional on batch size one. Verify valid grammar, monotonic times, EOS termination, repeated-time behavior and final-clip coverage under the exact inference path used for scoring.
- [modules.py](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/full_train/modules.py) trains with shifted teacher-forced token cross entropy. That training accuracy is not free-running chord recognition. Inspect [run.py](https://github.com/KimLeekyung/ACR_seq2seq/blob/main/full_train/run.py) for local username/hostname paths, Lightning integration and pretraining-checkpoint loading assumptions.

## Modern structural frame alternative: ChordFormer

The [author repository](https://github.com/mwaseemrandhawa/ChordFormer) documents four Conformer layers with 16 heads and structural outputs for root/triad, bass, seventh, ninth, eleventh and thirteenth. Its full system includes five-fold ensemble and HMM decoding. A single acoustic model must be reported separately from that ensemble/decoder system. These outputs relate to the user's proposed richer decomposition, but adopting them would expand our current target contract.

Source inspection resolves an important front-end detail: [CQTV2](https://github.com/mwaseemrandhawa/ChordFormer/blob/main/extractors/cqt.py) caches 288 bins at 36 bins/octave from F#0 with hop 512. [CQTPitchShifter](https://github.com/mwaseemrandhawa/ChordFormer/blob/main/mir/nn/data_decorator.py) crops the expanded array to the 252-bin network input; zero shift starts at bin 18. [confor_head16.py](https://github.com/mwaseemrandhawa/ChordFormer/blob/main/confor_head16.py) pads four dimensions to width 256 and selects GPU index 5 at import. The 288/252 dimensions serve different stages; they are not a contradiction or a reason to feed our 144-bin cache unchanged.

Before execution, adapt device selection and paths, pin dependencies and audit label limits, weighting and inference. The reweighting implementation's max_clip=1.0 deserves checking against the intended recipe rather than assuming the README's loss description is reproduced. No ChordFormer runtime failure or accuracy has been observed locally.

## Three distinct evaluation tracks

1. **F frame recognition:** predict a full timeline without reference boundaries; score raw acoustic outputs before optional smoothing/CRF/HMM. Export original logits, timestamps and denominators.
2. **E oracle classification:** retain the existing gold-event TCN/transposition control. A stronger encoder plus event pooling could be a useful controlled adaptation, but it must be labeled an adaptation rather than the published TE-DS baseline.
3. **E_seq2seq event recognition:** predict time/root/quality tokens freely from audio, reconstruct a complete timeline and evaluate chord plus segmentation metrics. No gold boundaries or gold previous tokens at test time. Gold-prefix likelihood, if measured, is an explicitly separate oracle diagnostic.

TE-DS is not a drop-in replacement for the existing four independent factor heads. Token conditional probabilities cannot be treated as the same root/triad/extension confidence values used in the F/E25 audit. First compare common complete-label timelines; define a new calibrated confidence interface before pseudo-labeling. The older oracle E score and free-running event score answer different questions.

## Proposed implementation sequence

1. Pin exact upstream commits and record file hashes/attribution. Review root-license/dependency/checkpoint terms before copying code; this review does not claim redistribution or pretrained-data provenance has been cleared. No third-party source is vendored by this document.
2. Create small I/O adapters and known-answer tests: supported quality symbols, N/OOV, frame times, event-token round trip, masks, EOS and full timeline coverage. Check model forward/backward and memory locally. Do not pretrain on validation players.
3. Bring up the author BTC large-vocabulary model as the frame anchor and TE-DS without pretraining as the event anchor. Keep current TCN results as measured controls. Use the frozen player development manifest and train-only preprocessing statistics.
4. Separate **architecture comparisons on a declared common front end** from **author-recipe adaptations with different front ends/training schedules**. Equal epochs across different architectures do not imply equal compute. Record parameters, updates, elapsed time and peak memory. Do not inherit TCN learning rate/30 epochs merely for convenience and call it faithful author training.
5. After smoke and small-set fitting succeed, prespecify a single-seed screening and repeat promising configurations. Compare overall duration Tetrads, common-label metrics, per-quality tradeoffs and segmentation. For event models, distinguish sequence errors from boundary errors with separate diagnostics.
6. Add pTE-DS pretraining and ChordFormer as later, independently measured increments. GuitarSet remains a small known-repertoire development source; reproducing paper headline results still requires matching benchmark audio, labels, folds and evaluation.

No new Kaggle notebook or training result is supplied by this review. The next concrete work is the pinned-code compatibility and I/O-adapter stage, not another run of 02k. Source review evidence is reports/p2/baseline-review/source_review.json.
