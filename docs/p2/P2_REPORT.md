# P2 report · Oracle event teacher E

## Purpose and protocol

P2 tests teacher complementarity at different temporal prediction units. F classifies frames and averages probabilities inside each reference interval. E averages TCN hidden features inside that interval and classifies the event. E fits its own weights; no F checkpoint is loaded. Both teachers receive the same reference times for this diagnostic.

The backbone, 37,225-parameter capacity, chroma normalization, target inventory and frozen song split are shared. E uses reference interval classification only. The chord-change boundary head is deferred to P3. Larger musical-section boundaries remain a separate MSA task.

The fixed F comparator is the balanced TCN seed 20261119 from revision d250916. Its complete oracle-event accuracy is 57.92% and rare macro oracle-event recall 4.19%, based on two correct rare intervals. Its variability is documented in [P1](../p1/P1_REPORT.md). F was fixed before E validation results were observed. P2 does not establish superiority to every F seed or configuration.

[Execution instructions](P2_INSTRUCTION.md) · [Frozen F provenance](../../reports/p2/f-reference/provenance.json)

## Implemented experiment choices

Event membership uses original timestamps in [start,end), merging touching identical supported root/quality labels after inversion collapse. Pooling masks exclude context and batch padding; context remains available to the convolutional encoder. Unsupported intervals are excluded. Event durations are preserved, including short and long intervals. A reference event without any sampled frame is explicitly omitted.

The full training set contains 31,679 supported events from 367 songs. Validation contains 9,673 scored events from 110 songs; one further supported interval has no frame. Teacher E receives one target per event. Batch size is 32 with 990 optimizer steps per epoch, for 30 epochs in the initial pilot. Compare this with F's 217 window updates per epoch. Differences in weighting and optimization budget limit attribution solely to prediction granularity.

Sampling retains natural events on half of draws. Other draws select quality uniformly, then song uniformly, then an event within that song uniformly. Losses remain unweighted for all component heads. N only supervises the no-chord gate. Event exposures are logged by quality each epoch.

The initial configuration uses width 48, seed 20261119, Adam at 0.001 and gradient clipping at 5. Checkpoints are selected by complete-chord validation event accuracy, including N. Record rare quality precision, recall, component correctness and song/event support for the selected checkpoint. This differs from F's checkpoint criterion (frame accuracy) and must be reported when interpreting the pair.

## Local checks and limits

E's two-song CPU overfit check completed 200 epochs and reached 100% complete-event training accuracy (80/80 events), with best checkpoint at epoch 108. This establishes that the pipeline can fit those examples. Its rare recall is measured only on the rare qualities present in those two training songs; it is not a validation result.

E's full data path was also exercised with a one-epoch CPU smoke run. That run is a technical check of batching, optimization, serialization and comparison, rather than the planned 30-epoch experiment. It reached 52.69% validation event accuracy after one epoch. The exported event decisions and F/E counts were independently checked. Do not use it as the final E baseline. [Local check records](../../reports/p2/local-checks.json)

Twenty tests passed, including preserved F state keys and numerical forward behavior, masked hidden-feature averaging, context/padding exclusion, variable-length batch invariance at pooled frames, finite gradients, F/E four-way comparison with N masks, timestamp membership, zero-frame omission, and reproducible event sampling. Existing P1 and Transformer checks remain passing.

The frozen reference contains all 9,673 validation events and F decisions, including raw component argmax values after probability pooling. The E comparison aligns exact recording IDs and interval endpoints and checks reference labels. Correctness is scored after prediction. The component comparisons use standalone head decisions; complete-chord decoding is also reported separately.

## Planned Kaggle outputs and analysis

Status: implementation and local checks complete; full GPU overfit/30-epoch validation run pending.

The Kaggle notebook will export checkpoints, run metadata, losses, event logits and per-quality results. For each quality, complementarity.csv reports both teachers correct, only F correct, only E correct and both wrong for complete chords and supported components. N masks root/triad/extension comparisons.

The central questions are whether E recognizes complete rare events missed by F and whether their component errors are complementary. Report small support and false positives. Oracle boundaries remove timing estimation from this comparison. Any observed complementarity must later be retested with predicted intervals in P3.

## Experiment log

| Date | Status | Evidence or decision |
|---|---|---|
| 19 November 2026 | Implemented | E shares F's TCN implementation and component heads, with independent optimization and masked mean pooling over reference events. |
| 19 November 2026 | Fixed reference | Preserve the original balanced F seed 20261119 on the same 9,673 validation intervals. |
| 19 November 2026 | Local check | Two-song event overfit reached 80/80 correct. |
| 19 November 2026 | Pending | Run the Kaggle overfit check and full 30-epoch oracle-event trial, then audit the returned logits and complementarity. |

## Weekly presentation notes

- P1 explored natural sampling, component weighting, balanced sampling and one compact Transformer.
- Balanced TCN remains the frame reference, with limited rare-event gains across seeds.
- P2 changes the supervision unit to reference chord events and pools hidden features before classification.
- The local event pipeline can overfit two songs.
- Full E validation accuracy and complementarity are still unmeasured at the planned training budget.
