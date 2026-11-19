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

Status: the first full GPU oracle-event trial has completed and been verified; see the measured results below.

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
- The first 30-epoch E validation trial and its F/E complementarity have now been measured; see the results below.


## First Kaggle oracle-event results · 19 November 2026

The returned ZIP contains the two-song check and the full event-teacher run from revision dbd870a2693fea2e2ffb70bd47a065b8828a76c7. The exact revision is recorded in the linked run metadata. Both runs use the 37,225-parameter TCN, PyTorch 2.11.0+cu130 and NumPy 2.1.3 on CUDA. E's weights were fitted independently without loading a teacher F checkpoint.

The two-song GPU check reached 80/80 correct training events, with best epoch 112. The full run completed 30 epochs and selected epoch 30. It obtained 75.54% event accuracy on the original natural training events and 58.45% on validation. Since the selected epoch is the last epoch of the initial budget, this does not establish that training has converged or that 30 epochs is optimal.

### Verification

The ZIP passed its integrity check. All source hashes match the recorded commit. Configs, train/validation song IDs, cache metadata, detailed-label hash and model capacity match the frozen protocol. Independent interval reconstruction reproduced all saved event IDs, times and labels, and decoded component logits reproduced both runs' predictions and metrics.

Every per-quality count, complete-chord recall, precision and standalone component score was recomputed. The four-way F/E table was independently rebuilt from reference labels and both teachers' decisions. Counts sum to 9,673 validation events. The frozen F CSV hash matches the full run record.

[Verification](../../reports/p2/kaggle-oracle/verification.json) · [Full run metadata](../../reports/p2/kaggle-oracle/full/run.json) · [Metrics](../../reports/p2/kaggle-oracle/full/metrics.json)

### Standalone results on identical reference intervals

| Metric | Fixed F balanced | Oracle event E |
|---|---:|---:|
| Complete-chord event accuracy including N | 57.92% | 58.45% |
| Correct complete events | 5,603 / 9,673 | 5,654 / 9,673 |
| Rare macro event recall | 4.19% | 2.09% |
| Correct rare complete events | 2 / 111 | 4 / 111 |

E has 51 more correct events overall, but the difference depends on N. Of 299 N reference events, F classifies 96 correctly and E 159. On the 9,374 supported chord events excluding N, F correctly classifies 5,507 (58.75%) and E 5,495 (58.62%). Thus E's overall gain does not demonstrate better chord recognition across the inventory.

The natural-sampling F reference from P1 reached 58.57% oracle-event accuracy, so E is also not the best standalone model among every earlier configuration. The comparator fixed for P2 remains balanced F. Different checkpoint criteria and optimization budgets are retained as limitations.

### Rare qualities and precision

| Quality | Reference events | F correct | E correct | E recall | E precision |
|---|---:|---:|---:|---:|---:|
| dim | 23 | 1 | 3 | 13.04% | 7.69% |
| aug | 5 | 0 | 0 | 0% | 0% |
| sus2 | 62 | 0 | 1 | 1.61% | 2.00% |
| min6 | 6 | 0 | 0 | 0% | 0% |
| minmaj7 | 4 | 1 | 0 | 0% | 0% |
| hdim7 | 9 | 0 | 0 | 0% | 0% |
| dim7 | 2 | 0 | 0 | 0% | 0% |

E correctly identifies more rare events by pooled count but has lower macro rare recall because each quality receives equal weight: F's single minmaj7 success is 25% recall for that quality, while E's successes are concentrated in dim and sus2 with larger denominators. E's low dim/sus2 precision also signals false positives. These figures do not establish reliable rare-chord recognition.

For other qualities, E's complete-event recalls are 7: 29.60%, min7: 29.55%, maj7: 29.84%, sus4: 12.11% and maj6: 8.33%. The corresponding F recalls are 24.60%, 25.94%, 12.57%, 0% and 0.93%. These local improvements coexist with lower minor-event recall (F 50.60%, E 40.32%).

[Full per-quality results](../../reports/p2/kaggle-oracle/full/per_quality.csv) · [E event decisions](../../reports/p2/kaggle-oracle/full/events.csv)

### Teacher complementarity

| Outcome | All scored events | Rare events only |
|---|---:|---:|
| Both correct | 4,650 | 1 |
| F correct, E wrong | 953 | 1 |
| E correct, F wrong | 1,004 | 3 |
| Both wrong | 3,066 | 106 |
| Total | 9,673 | 111 |

The initial pair shows complementary errors under reference boundaries: neither teacher's correct set contains the other's. For rare events, E supplies two additional dim successes and one sus2 success; F retains a minmaj7 success missed by E. However, 106/111 rare events remain wrong for both.

An oracle that uses reference correctness to pick the right teacher whenever either is correct would reach 68.30% overall event accuracy and 5.67% rare macro recall. This is a diagnostic upper bound requiring the answers, not an ensemble result achievable at inference.

The teachers agree on the complete predicted label for 6,294 reference events. Of those, 4,650 are correct, giving 73.88% agreement precision before any confidence filtering. Agreement alone is therefore insufficient as a reliability rule. Component agreement and confidence calibration must be evaluated before accepting pseudo-labels. The large overlap in wrong rare predictions constrains what pseudo-labeling can recover without other supervision.

[Four-way counts by quality/component](../../reports/p2/kaggle-oracle/complementarity.csv) · [Aggregate complementarity and agreement](../../reports/p2/kaggle-oracle/complementarity_summary.json)

### Resources, limitations and next step

E's full run recorded 341,702,656 bytes of peak PyTorch CUDA allocated tensor memory. The epoch loops totaled about 321.03 seconds, excluding preparation. Event lengths and optimizer steps differ from F, and sessions are separate; these are observed resource figures rather than a controlled efficiency claim.

This is one E seed paired with one prespecified F checkpoint, using supplied gold interval times. It establishes initial oracle-boundary complementarity, not robust gains across seeds, predicted-boundary performance or improvement from student pseudo-labeling. Repeating E and calibrating agreement remain useful follow-up checks.

The first P2 pilot is complete, while teacher E development continues in P2 as specified below. The later P3 will add a learned chord-change head and compare the teachers on predicted intervals under an explicitly fixed event-matching rule. Preserve these oracle results to measure the loss caused by segmentation. Do not describe the oracle metric as end-to-end recognition performance.

## Weekly update after Kaggle verification

- E successfully trained and reached 58.45% complete-event validation accuracy with reference times.
- F and E make complementary errors, with 953 F-only and 1,004 E-only correct events.
- The rare-event objective remains weak: 106 of 111 rare reference events are wrong for both.
- Whole-label agreement has 73.88% precision before filtering, so confidence auditing is necessary.
- P2 oracle pilot is complete; P3 predicted boundaries and student experiments remain unmeasured.

| Date | Status | Evidence and decision |
|---|---|---|
| 19 November 2026 | Verified | GPU overfit: 80/80. Full E: 5,654/9,673 correct at epoch 30. |
| 19 November 2026 | Measured | Both teachers show oracle-boundary complementarity, with only limited rare coverage and low rare precision. |
| 19 November 2026 | Planned | Preserve this reference-time comparison before P3 learned-boundary evaluation and any confidence-filtered pseudo-label experiment. |

Full checkpoints and logits remain under runs/p2-oracle/. The imported ZIP is in runs/imports/p2-oracle-results.zip; selected auditable results are versioned under reports/p2/kaggle-oracle/.


## Continue teacher E development before P3 · 19 November 2026

The user requested that E receive further development and validation comparable to F before moving to P3. The initial oracle run completes one experiment, not the entire E development phase. The next active work remains within P2.

E currently has one full configuration and one seed: balanced event sampling. F has natural, weighted and balanced trials plus repeated seeds and a Transformer trial. The balanced result for F does not establish that the analogous sampler is best for E. Their target frequencies differ because E weights events instead of frames.

### Next controlled comparison: natural versus balanced E

The new natural-event baseline uses [p2-natural.json](../../configs/p2-natural.json). Its only change from the completed E configuration is sampling=natural. Preserve seed 20261119, training/validation IDs, 37,225-parameter TCN, interval pooling, unweighted losses, batch size 32, learning rate 0.001, 30 epochs, clipping, fixed F reference and checkpoint criterion. Both schemes retain 31,679 event draws and 990 updates per epoch. Natural sampling visits each source event once in shuffled order.

Compare complete-event accuracy, rare macro recall, per-quality precision/recall and component accuracy, plus the four-way F/E counts. A standalone E gain and complementary correct rare events are different outcomes and should both be reported. The original balanced run remains preserved.

Status: the natural-E configuration has now completed its first Kaggle run and been verified; see the results below. The existing E two-song pipeline check already passed, so this unchanged architecture does not require retraining that check before this sampling comparison.

### Subsequent E experiments to decide from evidence

1. Compare natural and balanced sampling before attributing failures to the encoder or pooling.
2. Inspect the defining component confusions and training event frequencies. If component weighting is tested, derive weights from E's labeled training event counts, rather than reusing F's frame-frequency weights.
3. Repeat promising E configurations with matched additional seeds, preserving the F reference and identifying its fixed-seed limitation. Repeat E/F pairs if making a stability claim about teacher complementarity.
4. Consider one pooling/encoder or optimization-budget change at a time, motivated by the diagnostic. Any extended training budget should be compared under a declared rule because the first E checkpoint occurred at its final epoch.
5. Keep richer decomposition and Transformer alternatives documented as separate proposals.

These are candidates and evaluation priorities, not a requirement to run every technique previously tried for F. Select the next change from E's results and log any omitted or revised experiment. The objective is reliable rare-event supervision and useful teacher complementarity. No numeric improvement or guaranteed E performance is assumed.

### Weekly decision log

| Date | Status | Decision |
|---|---|---|
| 19 November 2026 | Revised next step | Continue E development in P2 instead of moving directly from one pilot run to P3. |
| 19 November 2026 | Prepared | Add natural-event E baseline with only sampling changed. |
| 19 November 2026 | Pending | Run the natural configuration on Kaggle and compare with balanced E before selecting the next intervention. |


## Natural-event E results · 20 November 2026

The natural-event trial completed 30 epochs at revision cac35e5, with the exact Git revision recorded in run.json. Its best checkpoint is epoch 13. Source hashes match the recorded commit. Training and validation IDs, feature/label versions, architecture, parameter count, batch size, learning rate, seed and total epochs match balanced E. Sampling is the intended configuration difference.

The imported archive passed integrity checks. Reference intervals and labels were rebuilt from the verified detailed annotation archive. Independent decoding of the saved logits reproduced all event predictions, component decisions and metrics. Per-quality recall/precision, the four-way F/E table and the fixed F reference hash were checked. All natural-sampling epochs visit the same 31,679 event targets once, in shuffled order.

[Verification](../../reports/p2/kaggle-natural/verification.json) · [Run metadata](../../reports/p2/kaggle-natural/run.json) · [Training history](../../reports/p2/kaggle-natural/history.json)

### Natural versus balanced E, seed 20261119

| Metric | Balanced E | Natural E |
|---|---:|---:|
| Complete-event accuracy including N | 58.45% | 63.38% |
| Correct events | 5,654 / 9,673 | 6,131 / 9,673 |
| Macro quality event recall | 17.09% | 14.18% |
| Rare macro event recall | 2.09% | 0% |
| Correct rare events | 4 / 111 | 0 / 111 |
| Selected epoch | 30 | 13 |

Natural sampling improves overall event accuracy by 4.93 percentage points, but macro quality recall falls by 2.92 points and rare macro recall falls by 2.09 points. Overall accuracy therefore gives a different answer from the rare-quality objective.

The gain is not explained by N alone: natural E correctly classifies 164/299 N events versus 159 for balanced E, a gain of five. On the 9,374 supported chord events excluding N, natural E correctly classifies 5,967 (63.66%) versus 5,495 (58.62%), a gain of 472 events.

| Quality | Balanced E event recall | Natural E event recall |
|---|---:|---:|
| maj | 74.90% | 83.21% |
| min | 40.32% | 49.24% |
| 7 | 29.60% | 17.54% |
| min7 | 29.55% | 33.79% |
| maj7 | 29.84% | 12.04% |
| sus4 | 12.11% | 2.63% |
| maj6 | 8.33% | 0% |
| dim | 13.04% | 0% |
| sus2 | 1.61% | 0% |

Every rare quality has zero complete-event recall in natural E. Some individual components remain recognizable: for min6 the root is correct on all six events, but extension recall is zero. For hdim7 the root is correct on 77.78% of events, but triad recall is zero. These observations support further component-level investigation; they do not by themselves identify a successful intervention.

[Numeric comparison](../../reports/p2/kaggle-natural/comparison.csv) · [Per-quality results including precision and components](../../reports/p2/kaggle-natural/per_quality.csv)

### Complementarity with the same fixed F

| Outcome | F + balanced E | F + natural E |
|---|---:|---:|
| Both correct | 4,650 | 5,022 |
| F only correct | 953 | 581 |
| E only correct | 1,004 | 1,109 |
| Both wrong | 3,066 | 2,961 |

Natural E provides more correct events missed by F overall. For rare events, however, all E-only gains disappear: F alone gets two correct rare events and both teachers miss the remaining 109. Balanced E contributes three rare events beyond F, though with low precision and limited support.

Full-label agreement for F + natural E occurs on 6,896 intervals, with 5,022 correct (72.82% precision before confidence filtering). F + balanced E has 73.88% agreement precision. Higher standalone accuracy thus does not automatically improve agreement reliability.

An oracle that selects the correct teacher using known reference correctness would reach 69.39% overall accuracy for F + natural E. This is only a diagnostic ceiling, not an achieved inference system or a pseudo-label reliability estimate.

[Four-way counts](../../reports/p2/kaggle-natural/complementarity.csv) · [Complementarity/agreeing-label summary](../../reports/p2/kaggle-natural/complementarity_summary.json)

### Checkpoint and stability limitations

The highest validation event accuracy occurs at epoch 13, while epochs 26–30 range from 61.05% to 62.04%. The natural-training accuracy at the selected checkpoint is 71.50%. The final epoch is not the reported checkpoint. Changing to rare-aware checkpoint selection after inspecting the results would be a separate experiment and must apply consistently to comparators.

This remains one matched seed for E. We have not established that the large overall gain, or balanced E's four rare-event successes, are stable across initialization and sampling variation.

### Decision and next planned E work

Retain natural E as the strongest current E reference for overall event accuracy. Retain balanced E as the candidate with rare-event complementarity. Do not choose the pseudo-label teacher configuration solely from overall accuracy.

The next proposed check is to repeat both E configurations with matched additional seeds before choosing which training policy to improve. Subsequent component weighting, if tested, must derive its weights from training events rather than F's frame counts. A lower balanced mixture or another optimization change is also a separate hypothesis; no such training change has been made.

P2 development continues. P3 is not the immediate next action simply because one more E run has completed. Both configurations and their tradeoff remain documented so that weekly reporting can distinguish general recognition performance from the rare-quality research objective.

| Date | Status | Evidence and decision |
|---|---|---|
| 20 November 2026 | Verified | Natural E reaches 63.38% overall event accuracy at epoch 13; all rare qualities retain zero event recall. |
| 20 November 2026 | Recorded | Natural E improves common-chord performance, while balanced E retains the only E-specific rare successes. Preserve both. |
| 20 November 2026 | Proposed | Repeat both E configurations with matched seeds, then select a justified component/sampling intervention. |

### Weekly presentation notes

- Established a controlled natural-versus-balanced event-teacher comparison.
- Natural sampling improves overall event accuracy by 4.93 points.
- Rare and macro quality recall decline, illustrating the effect of the training distribution.
- F + natural E has more complementary correct events overall, but no new correct rare events.
- Seed stability and component-level improvement remain open.

Raw outputs are in runs/p2-natural-kaggle/. The original archive remains in runs/imports/p2-natural-results.zip. Selected records and tables are versioned under reports/p2/kaggle-natural/.
