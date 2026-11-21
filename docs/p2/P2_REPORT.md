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


## Matched E seed-repeat protocol · 20 November 2026

Status: all four new GPU runs have completed and been independently verified; see the measured summary below. The user agreed to repeat natural and balanced E before selecting a policy.

Use the fixed plan [p2-seeds.json](../../configs/p2-seeds.json): additional seeds 20261120 and 20261121, each with natural and balanced event sampling. The original verified seed 20261119 pair is reused. Run the four new jobs sequentially in one GPU notebook session, with fresh independently fitted E weights per job.

Preserve the TCN, reference intervals, data split, normalization, component targets, unweighted losses, optimizer, batch size 32, learning rate 0.001, 30 epochs and checkpoint selection by validation complete-event accuracy. Each policy draws 31,679 training events per epoch. Natural visits each event once; balanced uses the declared mixture with replacement.

F is held fixed at the original balanced TCN seed 20261119. Its reference CSV hash is checked for every imported or completed E result. This tests E's variation and its complementarity against one fixed F. It does not measure variability of independently reseeded F/E pairs.

### Outputs and analysis plan

Retain each new checkpoint, history, logits, per-quality scores and complementarity table under runs/p2-seed-repeat/<method>-<seed>/. The repeat controller exports per_seed.csv and summary.json. Report mean and sample standard deviation over all three seeds, plus paired balanced-minus-natural differences for overall event accuracy, macro quality event recall and rare macro event recall.

Also record rare correct-event counts, total rare-event support and E-only rare successes relative to fixed F for every seed. Inspect per-quality precision and the actual event identities before interpreting nonzero macro recall. Repeated correctness on the same interval across seeds does not increase independent-song support.

Reusing an existing completed new job requires the same config and code revision. An incomplete run is reported explicitly. Original seed provenance is preserved in the existing reports; new jobs share one recorded revision. No checkpoint criterion, seed or training budget should be chosen after observing the repeat results without declaring a new experiment.

These three seeds provide descriptive stability evidence on the same validation songs, not statistical significance or certainty about new songs. The controller and aggregation were checked locally; the subsequent GPU execution and verified comparison are reported below.

### Weekly decision log

| Date | Status | Decision |
|---|---|---|
| 20 November 2026 | Prepared | Two additional matched seeds for natural/balanced E, four new training jobs. |
| 20 November 2026 | Fixed control | Keep F's original reference predictions unchanged to isolate E seed variation. |
| 20 November 2026 | Pending | Audit returned logits and rare-event identities before choosing the next component/sampling intervention. |


Local checks for the repeat workflow passed all 22 tests. The E controller test verifies four new training launches, reuse of completed jobs and inclusion of the original pair without retraining. A separate check verifies the balanced-minus-natural delta direction. Original F repeat aggregation remains numerically identical after sharing the aggregation helper. The existing E reports and frozen F hash were read successfully, and notebook syntax/document links were checked. These checks preceded the returned GPU runs assessed below.


## E repeated-seed results · 20 November 2026

The four new E jobs completed from revision f6a43eb73eb6ccac660f5bff5790fa523aa8884d. All source hashes match that commit. Configs, frozen train/validation IDs, normalized data, detailed annotations and the 37,225-parameter TCN were checked. Each job completed 30 epochs. F's original reference hash is unchanged.

Independent recomputation from event logits reproduced predictions, per-quality precision/recall and component correctness. Original reference intervals were reconstructed for all four jobs. Every four-way comparison count was rebuilt against fixed F. Finally, all three paired-seed aggregate means, sample standard deviations and deltas matched the notebook output.

[Verification](../../reports/p2/seed-repeat/verification.json) · [Plan and code provenance](../../reports/p2/seed-repeat/plan.json) · [Per-seed metrics](../../reports/p2/seed-repeat/per_seed.csv)

### Three-seed summary

The original seed 20261119 pair is reused with the additional seeds 20261120 and 20261121. Values are mean ± sample standard deviation; standard deviations are in percentage points.

| Metric | Natural E | Balanced E |
|---|---:|---:|
| Complete-event accuracy including N | 63.33% ± 0.18 | 58.44% ± 0.62 |
| Macro quality event recall | 14.65% ± 1.04 | 17.29% ± 1.59 |
| Rare macro event recall | 0.21% ± 0.36 | 3.95% ± 1.67 |

Balanced-minus-natural changes average -4.89 percentage points for overall event accuracy, +2.64 for macro quality recall, and +3.75 for rare macro recall. Balanced rare recall is higher for all three matched seeds, and overall accuracy is lower for all three. Macro quality recall improves on two seeds and declines slightly on one.

This is a consistent direction within these three seeds, not a statistical-significance or new-song generalization claim. Both methods use the same fixed validation songs. F stays at one reference checkpoint; this is E-seed variation rather than independent F/E seed-pair variation.

### Rare events and additional information beyond F

| Seed | Natural rare correct | Balanced rare correct | Natural rare macro recall | Balanced rare macro recall | Natural E-only rare correct | Balanced E-only rare correct |
|---|---:|---:|---:|---:|---:|---:|
| 20261119 | 0 / 111 | 4 / 111 | 0% | 2.09% | 0 | 3 |
| 20261120 | 0 / 111 | 5 / 111 | 0% | 5.34% | 0 | 5 |
| 20261121 | 1 / 111 | 3 / 111 | 0.62% | 4.42% | 1 | 2 |

Across balanced runs there are 12 correct-event occurrences covering 11 distinct intervals in six validation songs. Natural E recognizes one distinct rare interval. Repeated seeds still reuse the same validation set, so these interval/song counts describe the observed successes rather than additional independent evaluation data.

Balanced E recognizes dim in all three seeds (3/23, 4/23 and 1/23 events). It recognizes sus2 in two seeds (1/62 each), aug in one (1/5), and minmaj7 in one (1/4). The larger macro values for rare classes with very low support are sensitive to a single correct prediction. Min6, hdim7 and dim7 remain at zero complete-event recall in every balanced and natural run.

Rare precision remains low: balanced dim precision ranges from 3.33% to 7.69%, sus2 successes have about 2% precision, the one aug success has 5%, and the minmaj7 success has 10%. Therefore stronger rare recall has not yet yielded reliable rare pseudo-labels.

[Quality counts, independent-song support and precision by seed](../../reports/p2/seed-repeat/rare_quality_by_seed.csv) · [Correct rare interval identities](../../reports/p2/seed-repeat/rare_successes.json) · [Full aggregate statistics](../../reports/p2/seed-repeat/summary.json)

### Decision and next E hypothesis

Keep natural E as the current overall-accuracy reference. Use balanced E as the more promising candidate for rare-event supervision and E-only rare information, while preserving its accuracy and precision limitations. Do not describe either as a solved rare-chord teacher or begin accepting rare pseudo-labels solely from its predicted class.

The repeat test supports the sampling tradeoff more consistently than the first single-seed comparison. The next proposed controlled experiment is an intermediate mixture: 25% quality/song-balanced draws and 75% natural draws, compared with the existing 0% and 50% policies. The hypothesis is that a lower balanced fraction may recover some common-chord accuracy while retaining useful rare exposure. This is untested, not an assumed improvement.

Before running it, expose and record the mixture probability as an explicit configuration setting, keep the model/loss/budget and evaluation protocol fixed, and preserve the existing endpoints. Begin with the original seed as a screening experiment and repeat additional matched seeds if its tradeoff warrants it. No intermediate-mixture run has started.

Component weighting derived from E event counts remains another separate option if diagnostics justify it. P2 development continues; P3 remains a later stage. Confidence calibration and false-positive analysis are necessary before claiming pseudo-label reliability.

### Weekly presentation update

- Repeated natural and balanced E over three matched seeds.
- Natural E consistently has higher overall event accuracy.
- Balanced E consistently has higher rare-event recall, contributing extra correct rare events beyond fixed F in each seed.
- The rare successes span more than one interval, but precision remains poor and three rare qualities are never correctly classified.
- Proposed next test: a 25% balanced mixture, with the existing 0%/50% policies retained as references.

| Date | Status | Decision |
|---|---|---|
| 20 November 2026 | Verified | Four new E runs, per-event logits, F/E counts and three-seed statistics checked independently. |
| 20 November 2026 | Measured | Balanced improves rare macro recall in 3/3 seeds but loses about 4.89 overall-accuracy points on average. |
| 20 November 2026 | Proposed | Test 25% balanced exposure as a separate controlled sampling experiment; implementation/training pending. |

Full runs remain in runs/p2-seed-repeat/. The downloaded archive is in runs/imports/p2-seed-repeat-results.zip; selected auditable outputs are versioned under reports/p2/seed-repeat/.


## Intermediate mixture specification · 20 November 2026

Status: the first 25% mixture Kaggle run has completed and been independently verified; see the results below. Use [p2-mixture25.json](../../configs/p2-mixture25.json) with [02d_teacher_e_mixture25.ipynb](../../notebooks/02d_teacher_e_mixture25.ipynb).

The sampler now takes balanced_probability. Its default is 0.5, preserving the previous balanced policy. The new configuration explicitly selects 0.25. Each shuffled natural-event index has a 75% chance of being retained. Otherwise choose uniformly among observed qualities, uniformly among their training songs, and uniformly among that song's matching events. These probabilities describe the two draw branches, not a guarantee that exactly 25% of each class or batch is rare.

Preserve TCN width 48, the 367/110 train/validation split, normalization, event membership/pooling, targets, unweighted component losses, batch size 32, 30 epochs, learning rate 0.001, seed 20261119, clipping, fixed F reference and best-checkpoint criterion. Per-epoch budget stays 31,679 draws and 990 optimizer updates. Initialization and architecture are unchanged; realized training targets and their RNG sequence differ by sampling policy.

Record actual natural/balanced probabilities in run.json along with event/step budget. Record sampled event counts by quality in history.json, as before. Natural and 50% configurations remain available.

### Local checks

All 25 unit checks passed. New checks confirm invalid probability rejection, the exact 25% branch threshold, zero-balanced behavior and preservation of the previous default 50% sampled sequence. Existing F, E, repeat-controller and Transformer checks remain passing.

A train-only simulated epoch and one CPU forward/backward/optimizer update on actual event data were checked. [Exposure table](../../reports/p2/mixture25-setup/event_exposure.csv) · [Verification](../../reports/p2/mixture25-setup/verification.json)

The simulated draws are repeated observations from the same training songs. Their precise sequence differs from training because initialization and shuffle also consume random numbers. They establish sampler behavior, not improved recognition.

### Evaluation plan

Start with the prespecified original seed 20261119 and compare against the existing natural and 50% E runs at that seed. Use overall event accuracy, macro quality recall, rare macro event recall, quality-level precision and E-only correct rare events. Check distinct event identities and song support, rather than treating an isolated macro gain as robust improvement.

The hypothesis is that reducing balanced exposure could retain some rare-class learning while recovering common-quality accuracy. The measured endpoint tradeoff motivates this test, but the outcome is unknown. Do not assume linear interpolation between 0% and 50%.

A promising screening result can be repeated on matched additional seeds before changing the preferred teacher policy. No model accuracy or pseudo-label precision is claimed from the local implementation checks. P2 development continues; P3 has not started.


## 25% mixture results · 20 November 2026

The returned run completed 30 epochs at revision df662dd, with the exact revision in run.json. Checkpoint selection chose epoch 6. Source hashes match the recorded commit, and the explicit sampling policy is 25% balanced / 75% natural, with 31,679 event draws and 990 optimizer updates per epoch.

The ZIP integrity, configuration, original split and source versions were verified. Reference event IDs, times and labels were reconstructed from the annotation archive. Independent decoding reproduced all event predictions and component decisions. Per-quality precision/recall and F/E counts were recomputed. The fixed F reference hash is unchanged.

[Verification](../../reports/p2/kaggle-mixture25/verification.json) · [Run metadata](../../reports/p2/kaggle-mixture25/run.json) · [History](../../reports/p2/kaggle-mixture25/history.json)

### Same-seed screening comparison

All values below use seed 20261119 and the same 9,673 reference intervals.

| Metric | Natural 0% | Mixture 25% | Balanced 50% |
|---|---:|---:|---:|
| Complete-event accuracy | 63.38% | 59.85% | 58.45% |
| Macro quality event recall | 14.18% | 14.13% | 17.09% |
| Rare macro event recall | 0% | 3.94% | 2.09% |
| Correct rare events | 0 / 111 | 4 / 111 | 4 / 111 |
| Best epoch | 13 | 6 | 30 |

The 25% run correctly classifies 5,789 events, an increase of 135 over 50% balanced E (+1.40 percentage points overall). Its rare macro recall also increases by 1.85 points, but macro quality recall decreases by 2.97 points. Relative to natural E, overall accuracy remains 3.54 points lower.

This is not a uniform improvement. Minor-event recall falls to 28.92%, compared with 49.24% for natural and 40.32% for 50% balanced. Major recall is 82.38%, close to natural's 83.21% and above balanced's 74.90%. Common-class composition heavily influences overall accuracy.

[Three-policy comparison](../../reports/p2/kaggle-mixture25/comparison.csv) · [Per-quality results](../../reports/p2/kaggle-mixture25/per_quality.csv)

### Rare support and complementarity

| Rare quality | Correct / reference events | Recall | Precision |
|---|---:|---:|---:|
| dim | 1 / 23 | 4.35% | 3.57% |
| aug | 1 / 5 | 20.00% | 4.17% |
| sus2 | 2 / 62 | 3.23% | 9.52% |
| min6 | 0 / 6 | 0% | 0% |
| minmaj7 | 0 / 4 | 0% | 0% |
| hdim7 | 0 / 9 | 0% | 0% |
| dim7 | 0 / 2 | 0% | Undefined |

The macro score is (1/23 + 1/5 + 2/62) / 7 = 3.94%. One aug success contributes 2.86 percentage points to that aggregate, so the score is sensitive to this small class. Both 25% and 50% runs get four rare events right; their macro difference is caused by which qualities those events belong to.

All four successes of the 25% run are missed by fixed F: one aug event in 0303, two sus2 events in 0859 and one dim event in 0964. This covers four intervals in three validation songs. It increases E-only rare correct events from three in the 50% run to four in this screen. F retains its two rare successes missed by E. The remaining 105/111 rare events are wrong for both.

| F/E outcome | All events | Rare events |
|---|---:|---:|
| Both correct | 4,752 | 0 |
| F only correct | 851 | 2 |
| E only correct | 1,037 | 4 |
| Both wrong | 3,033 | 105 |

The teachers agree on complete labels for 6,601 events, with 4,752 correct (71.99% precision before filtering). The four E-only rare successes indicate complementary information, not reliable pseudo-labels in general. Precision for the successfully recognized rare qualities remains below 10%.

[Exact rare intervals](../../reports/p2/kaggle-mixture25/rare_successes.json) · [Four-way counts](../../reports/p2/kaggle-mixture25/complementarity.csv) · [Agreement summary](../../reports/p2/kaggle-mixture25/complementarity_summary.json)

### Decision after the screening run

The intermediate mixture is worth a matched-seed follow-up because it improves same-seed overall accuracy and rare macro recall over 50% sampling and provides four rare events beyond F. It is not selected as the best general configuration: macro quality recall and minor recognition are worse, and only one seed has been tested.

The next proposed check is to run this same 25% configuration at the already designated additional seeds 20261120 and 20261121, comparing against the existing natural/50% runs at those seeds. Preserve all controls, checkpoint criteria and F reference. Inspect per-quality precision and event identities before deciding whether the tradeoff persists. No new mixture repeat runs have started.

The selected checkpoint is early (epoch 6), whereas epoch 30 reaches 57.94% overall event accuracy. The selected checkpoint's natural-training event accuracy is 63.68%. These facts do not establish a unique cause for instability, nor justify changing the epoch criterion after viewing results.

The 25% configuration remains a candidate. Neither this screening result nor its oracle boundaries establish end-to-end or student improvements. P2 development continues.

### Weekly update

- Measured the intermediate 25% mixture between natural and 50% sampling.
- Same-seed overall accuracy and rare macro recall improve over 50%, while macro quality and minor recall worsen.
- Four correct rare intervals all add information missed by F, but precision is still poor.
- The next proposed step is to repeat the 25% policy on the two existing additional seeds.

| Date | Status | Decision |
|---|---|---|
| 20 November 2026 | Verified | Mixture 25%: 59.85% event accuracy, 3.94% rare macro recall, four E-only rare successes, selected epoch 6. |
| 20 November 2026 | Proposed | Repeat seeds 20261120/20261121 before adopting the intermediate mixture. |

Raw outputs are preserved in runs/p2-mixture25-kaggle/. The imported ZIP remains in runs/imports/p2-mixture25-results.zip, with selected records under reports/p2/kaggle-mixture25/.


## Repeat 25% mixture before selecting the policy · 20 November 2026

Status: the two new runs have completed and been independently verified; see the measured three-policy comparison below. The user agreed to follow up the single-seed screen.

Use [p2-mixture25-seeds.json](../../configs/p2-mixture25-seeds.json) and [02e_teacher_e_mixture25_repeat.ipynb](../../notebooks/02e_teacher_e_mixture25_repeat.ipynb) to train only the 25% mixture at seeds 20261120 and 20261121. Keep all settings in p2-mixture25.json except the seed. Each job retains 30 epochs, 31,679 event draws, 990 updates per epoch, unweighted losses and the frozen F reference.

Reuse seven existing results: the six verified natural/50% runs across three seeds and the original 25% seed 20261119. Two new jobs complete a nine-result, three-policy matched-seed table. No endpoint or original-seed retraining is required.

The summary reports mean and sample standard deviation of event accuracy, macro quality recall and rare macro event recall. Report paired 25%-minus-natural and 25%-minus-50% differences, including positive/negative/tied seed counts. Export rare correct counts and E-only rare correct counts per seed. Keep each quality's precision and the exact rare interval identities for the subsequent verification.

This is E variation against fixed F, on the same validation songs. It does not establish new-song confidence intervals, significance or independently reseeded F/E complementarity. The single aug event's influence on the original 25% macro score is a reason to inspect the repeated results carefully.

The notebook protects existing job directories and reuses completed jobs only at the same revision/configuration. Its comparison reads versioned endpoint results and records source revisions for all nine rows. The statistic helper rejects missing or duplicate policy/seed results.

Decision after the repeat should consider accuracy, quality balance, rare-event precision and useful additional events beyond F together. The 25% setting remains a candidate until these results are measured. P2 development continues; no P3 training starts as part of this notebook.


Local verification for the 25% repeat workflow passed all 28 tests. Hand-computed checks verified three-policy means, sample standard deviations and both paired-delta directions, with missing/duplicate policies rejected. A mocked execution of the actual notebook training cell launched exactly two 25% jobs at the designated seeds and reused them on rerun. Notebook syntax, document links and loading the original 25% result were checked. The two real GPU jobs have since completed; their verified results are recorded below.


## Verified three-policy comparison after 25% repeats · 20 November 2026

The two additional 25% E jobs completed at revision 8eb7e2dbf3ef82a9c4e33e2f9fcda7ca016535b9. Source hashes, configs, explicit 25% sampling probabilities, model capacity, frozen song split, data and F-reference hash were verified. Each run completed 30 epochs, and its checkpoint was selected by the declared event-accuracy criterion.

For both jobs, the detailed-label archive reproduced every reference interval and target. Decoding saved logits reproduced complete predictions and component decisions. All quality-level counts, recall, precision, component scores and four-way F/E tables were independently recomputed. Aggregating the two new jobs with the seven prior results reproduced all nine-row summary statistics and paired deltas.

[Verification](../../reports/p2/mixture25-repeat/verification.json) · [All nine seed/policy rows](../../reports/p2/mixture25-repeat/per_seed.csv) · [Full descriptive statistics and provenance](../../reports/p2/mixture25-repeat/summary.json)

### Three-seed descriptive results

Values are mean ± sample standard deviation across seeds 20261119, 20261120 and 20261121. Standard deviations are expressed in percentage points.

| Metric | Natural 0% | Mixture 25% | Balanced 50% |
|---|---:|---:|---:|
| Complete-event accuracy | 63.33% ± 0.18 | 60.58% ± 0.67 | 58.44% ± 0.62 |
| Macro quality event recall | 14.65% ± 1.04 | 17.68% ± 3.09 | 17.29% ± 1.59 |
| Rare macro event recall | 0.21% ± 0.36 | 7.88% ± 4.06 | 3.95% ± 1.67 |

Compared with 50%, the 25% mixture improves overall accuracy by 2.15 percentage points and rare macro recall by 3.93 points on average. Both changes are positive in all three matched seeds. Macro quality recall improves by 0.39 points on average, but is positive in only two seeds. Relative to natural sampling, the mixture loses 2.75 overall-accuracy points on average and gains 7.67 rare-recall points.

These are descriptive comparisons on one frozen validation set. They do not establish statistical significance or performance on unseen test songs. The three-seed average should not be replaced with the best seed's score.

### Rare-event results per seed

| Seed | 25% event accuracy | 25% rare macro recall | Correct rare events | E-only correct rare events versus fixed F |
|---|---:|---:|---:|---:|
| 20261119 | 59.85% | 3.94% | 4 / 111 | 4 |
| 20261120 | 61.14% | 12.05% | 10 / 111 | 10 |
| 20261121 | 60.77% | 7.64% | 6 / 111 | 4 |

The corresponding 50% E-only rare successes are 3, 5 and 2. Thus the intermediate mixture adds more correct rare information beyond the same F on every seed. F is fixed at its original balanced checkpoint; variation across independently reseeded F/E pairs is not measured here.

Across 25% runs there are 20 correct rare-event occurrences, covering 16 distinct intervals in eight validation songs. Correct occurrences by quality are dim 11, aug 3, sus2 3, min6 1, minmaj7 1, hdim7 1 and dim7 0. Some intervals recur across seeds; the counts must not be interpreted as 20 independent examples.

The 25% policy recognizes at least one complete event from six of the seven rare qualities somewhere across the runs. Min6 and hdim7 each succeed in only one event and one seed. Dim7 remains unrecognized in every tested policy. This is improved coverage, not reliable recognition of all rare qualities.

[Per-quality support/precision by seed](../../reports/p2/mixture25-repeat/rare_quality_by_seed.csv) · [Exact rare successes and F correctness](../../reports/p2/mixture25-repeat/rare_successes.json)

### Precision and seed sensitivity

Rare recall remains variable (3.94% to 12.05%). Small denominators matter: a single min6 event represents 1/6 of that quality's support, and one aug event represents 1/5. Independent-song support is unchanged by repeating seeds.

Precision also remains insufficient for unconditional pseudo-labeling. For seed 20261120, dim precision reaches 22.22%, but aug is 8.33%, sus2 0.95% and min6 4.17%. For seed 20261121, minmaj7 precision is 16.67% and hdim7 3.13%. Aggregate improvement does not make every class or predicted component trustworthy.

### Policy decision for further P2 work

Use the 25% mixture as the preferred current E candidate for the rare-oriented research branch. Keep natural E as the strongest overall-accuracy reference and preserve 50% E as a sampling ablation. This is a development choice supported by the matched validation comparisons, not a final test-set claim. Do not select a new seed solely because it produced the highest rare score.

For a primary same-seed F/E diagnostic, retain the original E25 seed 20261119 checkpoint alongside fixed F seed 20261119, with the two repeats documenting variation. This avoids silently replacing the primary pair with the most favorable E seed. Any later checkpoint/seed choice must be recorded explicitly.

The next proposed analysis is a confidence/partial-target reliability audit using saved F and E logits. Measure agreement precision and coverage per component and rare quality before accepting pseudo-labels. This can be performed locally without another full training run. It is development-set evidence because the same validation songs have already informed model choices; any fitted thresholds need a clearly specified calibration/evaluation protocol before final claims.

P3 remains later: this entire comparison supplies gold intervals. No predicted-boundary or student improvement has been measured. The current result justifies preserving E25 for further analysis, rather than claiming the two-teacher method is already successful.

### Weekly presentation update

- Completed a matched three-seed comparison of 0%, 25% and 50% event sampling.
- The 25% mixture improves both overall accuracy and rare recall over 50% in all three seeds.
- It reaches 60.58% average event accuracy and 7.88% average rare macro recall.
- Correct rare events span 16 intervals and eight validation songs, but precision and low support remain major limitations.
- The next proposed step is to audit component confidence/partial-label reliability using the saved teacher outputs.

| Date | Status | Decision |
|---|---|---|
| 20 November 2026 | Verified | Two new E25 jobs and all nine aggregate rows checked independently. |
| 20 November 2026 | Selected candidate | Prefer E25 for further rare-oriented P2 analysis; retain natural/50% controls and the original seed as the primary pair. |
| 20 November 2026 | Proposed | Audit component agreement precision, coverage and confidence before pseudo-label generation or P3 claims. |

Raw runs remain in runs/p2-mixture25-repeat/. The ZIP is preserved in runs/imports/p2-mixture25-repeat-results.zip. Selected auditable outputs are versioned under reports/p2/mixture25-repeat/.

## Component confidence, precision and coverage audit — 20 November 2026

### Scope and reproducibility

Completed locally using the original balanced F and E25 checkpoints at seed 20261119. F revision is d250916f82a1a5e2f0310920209450465636b23e; E revision is df662dd3ffac14c5419f77137e94b01ba9ec6aa1. No new training, augmentation, extra dataset, pseudo-label generation or student run was performed.

The audit covers the same 9,673 scored gold intervals in 110 validation songs: 9,374 chord events and 299 N events. One reference interval without frames remains excluded. F probabilities are averaged over each gold interval; E probabilities are the softmax of its saved event logits. This is an event-level diagnostic, not F's frame-level score or predicted-boundary performance. Train-hidden/test recordings are not evaluated.

The loader checks split/vocabulary metadata, annotation checksum, original F provenance, E's F-reference checksum, event order and boundaries, frame timestamps/labels, finite logit dimensions, saved component decisions and complete predictions. Input logit hashes are recorded in [summary.json](../../reports/p2/confidence-audit/summary.json). An independent count check uses the frozen CSV predictions rather than the new loader; all 168 curve rows satisfy count identities, and class/quality partitions and confidence bins conserve totals. [Verification](../../reports/p2/confidence-audit/verification.json). The full test suite passes 31 tests, including three new selection tests.

### Definitions and exploratory protocol

- **Confidence:** maximum per-head probability. F uses mean frame probabilities, E event softmax. Agreement requires the same raw head argmax; its score is the minimum of the two confidences, not a calibrated joint probability.
- **Precision:** correct accepted component predictions divided by all accepted predictions. With no accepted predictions this is undefined, exported as a blank CSV field.
- **Coverage:** accepted events divided by the stated eligible event population. Correct coverage additionally requires the accepted component to be correct. Counts and distinct accepted songs accompany rates.
- **Conditional scope:** gate evaluated on all events; root/triad/extension on true chord events, independently of the predicted gate. This diagnoses each head but uses ground truth to define eligibility.
- **Operational scope:** denominator is all 9,673 events. Root/triad/extension also require that the relevant teacher predicts chord with gate confidence at least the same threshold. Agreement requires both teachers' chord gates. A component accepted on a true N event counts as wrong. These filters can be computed without reference labels, although evaluation still uses gold intervals.
- **Class precision versus true-quality slices:** predicted_classes.csv groups accepted predictions by their predicted component value and includes false positives from every reference quality. true_qualities.csv instead conditions on a true chord quality; its precision column is selected accuracy within that reference slice, not precision for predicting that quality. accepted_recall in the class table is correct accepted predictions divided by true class support.

The fixed exploratory grid is 0, 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99. Threshold zero means no confidence rejection; gate/argmax requirements still apply. The 0.9 rows below illustrate the tradeoff, not a selected deployment threshold. No thresholds or probability calibrators are fitted. Ten fixed-width bins in reliability_bins.csv summarize conditional confidence versus observed accuracy; they do not certify calibration. Validation has already informed model choices, so all findings remain development evidence.

### Individual heads before filtering

Conditional scores below retain every eligible event. Root/triad/extension each use 9,374 true chord events; gate uses 9,673 events. Confidence is a mean score, not an empirical success probability.

| Head | F accuracy | F mean confidence | E accuracy | E mean confidence |
|---|---:|---:|---:|---:|
| Gate | 97.69% | 98.11% | 98.37% | 98.96% |
| Root | 81.14% | 75.25% | 85.57% | 84.99% |
| Triad | 86.37% | 83.19% | 84.03% | 88.77% |
| Extension | 79.08% | 79.22% | 77.69% | 75.57% |

E is stronger on root for this pair, whereas F is stronger on triad and extension. E's average triad confidence exceeds its observed accuracy. These averages alone are not calibration tests; the selection curves show why confidence cannot be interpreted directly as reliability.

### Precision/coverage tradeoff with operational filtering

Every cell below is **precision / coverage**, with coverage over all 9,673 events. Threshold 0.9 is applied to the component and required chord gate(s).

| Head | F at 0.9 | E at 0.9 | Agreement, no confidence rejection | Agreement at 0.9 |
|---|---:|---:|---:|---:|
| Gate | 99.04% / 96.29% | 98.92% / 98.30% | 98.56% / 98.91% | 99.47% / 95.33% |
| Root | 92.04% / 28.72% | 93.89% / 56.19% | 89.14% / 84.57% | 95.03% / 24.13% |
| Triad | 95.30% / 44.43% | 93.26% / 63.64% | 88.92% / 86.60% | 96.54% / 39.41% |
| Extension | 92.46% / 31.67% | 89.75% / 22.38% | 82.25% / 84.64% | 93.49% / 15.39% |

Agreement at 0.9 retains 9,221 gate, 2,334 root, 3,812 triad and 1,489 extension targets. Precision gains have a substantial coverage cost. Raising thresholds does not guarantee improved precision: agreement-root precision falls from 95.03% at 0.9 to 94.31% at 0.95, while coverage falls from 24.13% to 11.44%.

### Rare-quality failure hidden by aggregate scores

There are 111 events in the seven training-defined rare qualities. These rows show correct/accepted component decisions **within those true rare events**, not precision of predicting a rare quality.

| Head | Agreement, threshold 0 | Coverage of 111 events | Agreement, threshold 0.9 | Coverage of 111 events |
|---|---:|---:|---:|---:|
| Gate | 111 / 111 | 100.00% | 105 / 105 | 94.59% |
| Root | 74 / 89 | 80.18% | 10 / 11 | 9.91% |
| Triad | 3 / 90 | 81.08% | 0 / 24 | 21.62% |
| Extension | 70 / 81 | 72.97% | 11 / 12 | 10.81% |

The central failure is shared triad error. Before confidence rejection, F gets 7/111 and E 10/111 rare-event triads correct; requiring agreement retains 90 predictions but only three are correct. At threshold 0.9 all 24 retained rare-event triads are wrong. High agreement can preserve a common mistake.

Class-level accepted predictions explain the apparently strong aggregate triad score:

- At agreement threshold 0.9, every accepted triad is major (3,494 predictions) or minor (318). No dim, aug, sus2 or sus4 triad is retained. Even threshold zero retains only four predicted dim triads, two correct, and no aug/sus2/sus4 predictions.
- Of 1,489 accepted extension predictions at 0.9, 1,488 are `none`; the remaining one is `b7`. None are `6`, `7` or `bb7`. The single correct b7 prediction is not evidence of reliable b7 recognition.
- Extension correctness on rare events often represents correctly predicting `none` on qualities such as dim/aug/sus2. It does not establish that the rare chord itself was recognized.
- Whole-label agreement without a confidence threshold accepts 6,601 events, with 4,752 correct (71.99% precision; 68.24% coverage). Within the 111 rare events, it accepts 58 and gets **zero** complete chords correct. The primary pair has no shared complete rare success despite E's four correct rare events individually.

### Decision and next research step

Do not adopt one universal confidence threshold or generate unrestricted pseudo-labels from this pair. Aggregate root/triad precision can support further investigation of partial targets, but the current acceptance policy supplies essentially no confident rare-defining triads or nontrivial extensions. It could reinforce the majority-class bias if used uncritically.

The next discussion should examine training-data diversity: augmentation and/or additional datasets, starting with per-quality and per-component support across independent songs. Preserve the Billboard-only results and frozen validation/test split as controls. Any later training experiment should distinguish additional examples from additional optimizer steps and reassess both precision and coverage, including rare-class false positives. This audit identifies a failure pattern; it does not prove which data intervention will fix it. No augmentation or new dataset has been implemented yet.

Threshold fitting, if pursued later, requires an explicit calibration/evaluation protocol; the reused development set cannot supply an unbiased final reliability claim. This audit concerns one preselected F/E seed pair, not confidence robustness across all seeds.

### Weekly presentation notes

- Audited frozen F/E outputs locally without another Kaggle run.
- Agreement plus confidence improves aggregate precision but reduces accepted target coverage.
- At threshold 0.9, triad precision is 96.54% overall, yet all 24 accepted triad decisions on rare events are wrong.
- Accepted confident triads cover only major/minor; accepted extensions are almost entirely `none`.
- Hold pseudo-label deployment. Investigate data diversity next, keeping the current baseline and evaluation controls.

Machine-readable outputs: [curves](../../reports/p2/confidence-audit/curves.csv), [predicted component classes](../../reports/p2/confidence-audit/predicted_classes.csv), [true-quality slices](../../reports/p2/confidence-audit/true_qualities.csv), [confidence bins](../../reports/p2/confidence-audit/reliability_bins.csv). Reproduction instructions are in P2_INSTRUCTION.md.

## Dataset feasibility review — 20 November 2026

Reviewed the user's ACR literature matrix and primary dataset sources after the confidence audit. See [DATASET_REVIEW.md](DATASET_REVIEW.md) for sources, access/annotation differences, priorities and admission criteria. Prioritize JSIR/JAAH annotation audits for rare harmony and GuitarSet for an accessible guitar-audio pilot; retain Isophonics as a pop/rock expansion conditional on matching audio. JAAH's documented 12-dimensional chroma must not be assumed compatible with our 24-dimensional input. JSIR requires valid-segment filtering and composition-level grouping. No candidate is yet imported, no complete label-support audit has been performed, and no augmentation or new training has started. This is a source-level recommendation, not a measured data improvement.

## JAAH pilot implementation — 20 November 2026

Following the source-level review, JAAH was downloaded at a pinned revision and audited at file level. Keep 103/113 recordings and 15,491 supported merged events; exclude ten duration mismatches. Six of the seven training-defined rare qualities receive additional examples, including 617 dim7 and 429 hdim7 events. There are no supported sus2/sus4 targets under the exact mapping. The original review above describes the earlier stage; the new measured inventory is under reports/p2/jaah-inventory/.

Implement a separate 12-to-48 input projection for JAAH and preserve the 24-to-48 Billboard projection, sharing E's TCN, event pooling and heads. The two fresh pilot arms use the same seed and total update budget: Billboard control versus 248/990 auxiliary batch replacements per epoch. F remains fixed. See [P2_JAAH_EXPERIMENT.md](P2_JAAH_EXPERIMENT.md) for the protocol and notebook.

This is an implementation and data-feasibility milestone. Full GPU experiments, repeatability across seeds and any improvement in rare-chord precision/coverage remain unmeasured. Metadata matching does not replace audio fingerprinting, and duration checks do not prove local label alignment. The added projection handles differing input dimensions; it does not recover missing bass/treble information.

Local verification passes 36 tests. A training-only check memorized eight Billboard and eight JAAH events, reaching 100% on each branch after 168 optimizer steps. This confirms that both input paths can learn through the shared model, not that validation scores improve. Raw smoke artifacts remain in runs/p2-jaah-smoke/ and selected evidence in reports/p2/jaah-inventory/overfit_check.json.


## Expanded F/E JAAH pilot — 20 November 2026

The preceding E-only pilot is superseded for the next experiment by four fresh training jobs: F control/mixed and E control/mixed. Dataset expansion is now tested independently for both teachers. Each teacher has separate Billboard/JAAH input projections and its own shared trunk/heads; F and E do not share weights. F replaces 54/217 batches and E replaces 248/990 batches per full epoch. Both arms within each teacher keep the same seed and training budget.

Evaluate all four F/E combinations on identical Billboard gold intervals. F outputs are pooled as mean frame probabilities; E uses event probabilities. The new analysis reports individual scores, per-quality precision/recall, F-only/E-only successes and confidence/precision/coverage by head, predicted component class and true quality. The E run directories retain their historical fixed-F comparator for compatibility; the four-pair analysis directory is authoritative for this experiment.

Use notebook 02g_teachers_fe_jaah_pilot.ipynb instead of the E-only 02f notebook. Four CPU smoke jobs and all four pair analyses passed on 163 validation-subset events; both auxiliary training paths were exercised. The F smoke uses 50% replacement solely to exercise JAAH in its two-batch epoch, while full training uses 25%. These execution checks establish no generalization improvement. Selected evidence: [verification.json](../../reports/p2/jaah-fe-pilot/verification.json). Full GPU training and multi-seed confirmation remain pending. See [P2_JAAH_EXPERIMENT.md](P2_JAAH_EXPERIMENT.md) for the complete protocol.


## JAAH F/E pilot results — 20 November 2026

Imported runs/imports/p2-jaah-fe-results.zip and verified four full 30-epoch runs at clean revision 79ec0355111ad03dc93f8a03111d0e630ed80096, seed 20261119. Source hashes match that Git revision. F uses 217 updates/epoch (54 auxiliary replacements in mixed); E uses 990 (248 auxiliary). Best epochs are F control 20, F mixed 28, E control 11, E mixed 12. Recomputed all 13 analysis files from saved probabilities/logits on the same 9,673 validation events. Counts, decisions and scores match; mean-confidence roundoff is at most 2.23e-16. Evidence: [import verification](../../reports/p2/jaah-fe-pilot/import_verification.json), [individual scores](../../reports/p2/jaah-fe-pilot/results/individuals.csv), [pair summaries](../../reports/p2/jaah-fe-pilot/results/summary.json).

| Model | Event accuracy | Rare macro recall | Macro quality recall |
|---|---:|---:|---:|
| F control | 58.348% | 0.621% | 12.585% |
| F + JAAH | 57.118% | 1.242% | 13.180% |
| E control | 59.981% | 0.852% | 14.779% |
| E + JAAH | 61.170% | 0.230% | 13.560% |

F loses 1.230 percentage points of event accuracy; its rare improvement is only one additional correctly recognized dim event (1 to 2 out of 23). E gains 1.189 points of event accuracy but loses rare and macro quality recall: its correct rare events fall from two (one dim, one sus2) to one sus2. Neither mixed model correctly recognizes any aug, min6, minmaj7, hdim7 or dim7 validation event. This pilot has not achieved the intended rare-harmony transfer despite the added source examples.

### Pair complementarity and confidence

Here C means Billboard control and J means the JAAH mixed arm. Complete-label agreement has no additional confidence threshold in this table.

| F/E pair | Both correct | F-only | E-only | Both wrong | Agreement precision | Agreement coverage | Correct rare agreements / accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| C/C | 4820 | 824 | 982 | 3047 | 72.51% | 68.72% | 1/77 |
| C/J | 4929 | 715 | 988 | 3041 | 71.72% | 71.05% | 0/72 |
| J/C | 4785 | 740 | 1017 | 3131 | 72.22% | 68.50% | 0/74 |
| J/J | 4845 | 680 | 1072 | 3076 | 71.99% | 69.58% | 0/72 |

C/J has only six more events correct in at least one teacher than C/C (6632 versus 6626). This is an oracle union, not an achievable fusion score. Increased agreement coverage is accompanied by lower agreement precision. No pair is promoted for pseudo-labeling.

At exploratory head-confidence threshold 0.9, operational agreement precision / coverage is:

| F/E pair | Root | Triad | Extension |
|---|---|---|---|
| C/C | 94.84% / 22.62% | 96.56% / 38.14% | 94.73% / 16.68% |
| C/J | 94.57% / 23.62% | 96.45% / 37.60% | 93.64% / 17.23% |
| J/C | 94.96% / 20.91% | 97.11% / 36.09% | 94.37% / 18.56% |
| J/J | 94.56% / 21.86% | 97.09% / 36.27% | 94.07% / 20.76% |

On true rare-quality events, accepted triads are correct 0/21, 0/21, 0/18 and 0/18 respectively. High aggregate triad precision therefore still hides shared confident rare errors. Extension coverage increases mainly for the `none` class: J/J accepts 1,998 none and only 10 b7 predictions, versus C/C 1,592 none and 21 b7; no other extension classes are accepted at this threshold. This is not broad coverage of complex extensions. Full per-class and per-quality evidence is in [results](../../reports/p2/jaah-fe-pilot/results/curves.csv).

### Decision and next investigation

Retain the existing selected teachers; do not replace both with the mixed models or start rare pseudo-labeling. Keep E+JAAH as an accuracy candidate, but one seed on reused gold-boundary validation cannot establish repeatability. A new control run is the appropriate comparator; differences from historical runs are not dataset effects.

Before spending more GPU time on mixtures, diagnose source-specific learning and transfer: evaluate training-only JAAH subsets by component (explicitly a fit diagnostic, not generalization), inspect rare Billboard errors and source normalization/clock semantics, and distinguish insufficient auxiliary fitting from cross-domain transfer failure. This experiment replaces Billboard updates, so reduced Billboard exposure is another plausible explanation; it does not isolate a pure added-data effect. Test any augmentation or changed sampling as a separate matched-budget experiment with the original control. If E's accuracy tradeoff is pursued, repeat control/mixed with matched additional seeds before selection. No new training experiment has been run as part of this import review.


### Auxiliary-source fit diagnostic

Evaluated the unchanged mixed best checkpoints on all 103 audited JAAH auxiliary-training recordings and 15,491 supported events, without optimizer updates. F uses saved full-song frame logits pooled as mean probabilities; E uses gold-event pooling with batch size 32. Cache hashes are checked before evaluation. This is a training-source fit check, with checkpoints selected on Billboard validation, and provides no held-out JAAH generalization estimate.

F event accuracy is 24.091%; E is 26.596%. F gets none of the six present rare qualities correct; E has some source rare successes (dim 2/76, aug 4/17, min6 17/237, minmaj7 23/42, hdim7 12/429, dim7 4/617). Source rare macro recall is 0% for F and 15.257% for E; sus2 is absent, so this macro has a different support set from Billboard's seven-quality metric.

Both models overpredict dominant seventh quality: F emits 13,341 and E 12,331 such labels among 15,491 events, while the reference has 6,519. F's min7 triad accuracy is only 7.34% despite 96.33% extension accuracy on that quality; E has 16.57% and 94.99%. This identifies component errors and a source prediction imbalance, but does not establish their cause. Low source fit means a pure transfer-only explanation is insufficient. Check feature-bin conventions, normalization, temporal semantics and source-specific optimization before another large training sweep. Existing tiny-event memorization showed that the paths can learn, not that they learn the full source adequately.

Machine evidence: [source-fit summary and method](../../reports/p2/jaah-fe-pilot/source-fit/summary.json), [F quality details](../../reports/p2/jaah-fe-pilot/source-fit/F_qualities.csv), [E quality details](../../reports/p2/jaah-fe-pilot/source-fit/E_qualities.csv). Temporary predictions remain under ignored runs/p2-jaah-source-fit/.


### Pipeline and realized exposure audit — 20 November 2026

Compared all 103 cached feature matrices directly with the pinned raw CSV files: timestamps match exactly, and cached features exactly equal per-frame L1 normalization of all 12 source columns in their original order. All 15,491 event slices match timestamp search bounds and frame targets. Median frame step is 0.046439909297 seconds for every recording. No dropped/reordered feature column or cache/interval indexing discrepancy was found. This checks implementation consistency only; it does not establish the physical pitch-class convention or audio/annotation alignment of the upstream features.

Accumulated actual auxiliary exposure through the selected checkpoints (F epoch 28, E epoch 12). Dominant seventh accounts for 37.38% of F's scored JAAH training frames and 33.76% of E's sampled JAAH events. Yet the fitted models predict this quality for 86.12% and 79.60% of source events respectively. The denominators differ for F, but the disparity supports investigating prediction collapse beyond mere source frequency. Rare targets did reach training: F receives 265,606 dim7 and 296,608 hdim7 frame exposures; E receives 4,693 dim7 and 3,858 hdim7 event exposures. These are repeated sampling counts, not unique examples. Missing rare-target delivery is therefore not the explanation for these two classes. sus2/sus4 remain absent by source mapping.

Evidence: [pipeline_audit.json](../../reports/p2/jaah-fe-pilot/pipeline_audit.json). Next unresolved checks are the upstream pitch-bin/feature semantics and source-specific optimization under shared heads. The present checks justify neither rotating features nor changing labels. Keep the next experiment focused on source fitting before increasing the dataset mixture or launching seed repetitions.


### Upstream semantics and revised research priority — 20 November 2026

Reviewed JAAH's docs_utils/site_utils.py and the upstream pychord_tools third_party.py / low_level_features.py. The NNLS extractor sums a weighted semitone spectrum into chroma and explicitly rolls A-based bins to C-based order. The CSV writer emits i*2048/44100 timestamps; annotation-root rotation is in a separate beat-statistics stage. This supports keeping the distributed frame features in their existing column order. Retrieved code hashes are recorded in [upstream evidence](../../reports/p2/jaah-fe-pilot/upstream_feature_evidence.json). The dependency revision used to produce the historical CSVs is not pinned by this inspection, so intended semantics are stronger evidence than shape alone but do not prove physical audio alignment.

The user clarified that overall accuracy and comparison with published ACR should take priority over rare-chord optimization. Adopt [MULTICORPUS_ACCURACY_PLAN.md](MULTICORPUS_ACCURACY_PLAN.md): expand pop/rock sources with matching audio, use GuitarSet as an accessible audio pilot, reuse BTC author preprocessing/baselines, and introduce a controlled grouped evaluation protocol. Keep rare metrics as secondary diagnostics. Gold-boundary event accuracy is not directly comparable to published end-to-end duration-weighted WCSR. Five-fold evaluation and corpus expansion address different problems. New source-fit training, added-corpus ingestion and benchmark runs remain pending.


### Quality-diverse memorization check

Selected the first two eligible JAAH events per present quality (8–100 frames), giving 24 events across 12 qualities; sus2/sus4 have no source support. Both models start fresh at seed 20261119, width 48, Adam learning rate 0.001, clipping 5, one fixed full batch and at most 300 updates. F trains on event-interior frame targets with context masked; its event decisions average frame probabilities. E trains on pooled event targets. Accuracy is checked every ten updates. F reaches 24/24 after 190 updates and E after 160. The exact selected intervals are saved in [diverse_memorization.json](../../reports/p2/jaah-fe-pilot/diverse_memorization.json).

This rules out an absolute inability of either current input/label path to learn these varied examples. It does not validate all upstream audio alignment, explain full-corpus failure, or measure generalization. Full-source fit and cross-source optimization remain distinct questions; no broad architecture advantage follows from memorization.


## Audio admission and standard metrics — 20 November 2026

The user confirms no matching Isophonics/USPOP/Robbie audio is currently available. Documented access dependencies in [POP_AUDIO_ACCESS.md](POP_AUDIO_ACCESS.md); these annotation collections are not counted as new training data. GuitarSet 1.1.0 is the accessible audio pilot.

Added the optional pinned mir_eval 0.8.2 evaluator for full explicit LAB timelines, with correct/eligible/excluded seconds per Root, MajMin, Triads and Tetrads metric. Tests verify duration rather than event weighting, unknown/no-chord handling, reductions and invalid timeline rejection. All 44 tests pass with the optional package installed. See [STANDARD_EVALUATION.md](STANDARD_EVALUATION.md).

Rescored the four existing teachers on exported gold intervals as a scope-limited oracle diagnostic. F control/mixed Tetrads are 62.72%/62.31%; E control/mixed are 61.87%/60.57%. Thus E's event-accuracy improvement does not extend to duration-weighted Tetrads. These results exclude unavailable raw/OOV/trailing regions and must not be compared directly with published full-track WCSR. Machine evidence: reports/p2/jaah-fe-pilot/duration_metrics.json. No model promotion or new training follows from this diagnostic.


### GuitarSet file-level inventory completed

Verified official annotation and microphone archives by published MD5, paired 360 recordings and checked WAV headers/label bounds. Excluding the two documented timing errata leaves 358 candidates. The existing label mapping supports 66.34% of performed labeled duration, with 3,678.62 seconds OOV; instructed labels have full mapping coverage but different supervision semantics. Explicit power chords are a major unsupported category and cannot be silently relabeled as major/minor. Candidate groups are not yet final leakage groups, especially across styles sharing the three underlying progressions. See [GUITARSET_INSTRUCTION.md](GUITARSET_INSTRUCTION.md) and reports/p2/guitarset-inventory/. No new training data or folds were admitted.


### GuitarSet exact label mapping and progression folds

Implemented pitch-set equivalence with mir_eval's Harte primitives, preserving additions/omissions and annotated roots while collapsing bass as in the current task. Complete-label coverage stays 66.34%; the remaining valid labels are proposed for root/gate-only supervision, with other heads masked. No partial-target training has been enabled. This confirms the OOV gap is not recoverable merely by spelling normalization.

Audited all instructed sequences and found exactly three transposition-invariant progression signatures across styles/tempi/players. Created three outer progression folds with separate train/validation/test progression groups and the two known errata excluded. Full checks verify disjoint groups, comp/solo pairing and unchanged previously supported class IDs; 48 tests pass. This is a small unseen-progression protocol, not unseen-player evaluation or the published BTC benchmark. See [GUITARSET_INSTRUCTION.md](GUITARSET_INSTRUCTION.md) for the contract and reproduction commands. CQT extraction and model training remain pending.


## GuitarSet audio baseline prepared

Implemented BTC-inspired log-CQT extraction and fresh 144-channel F/E TCN jobs, with a training-only scalar normalization and fold-0 train/validation loading. Prepared all 239 development recordings locally; outer test remains unloaded. Full raw performed labels are retained for duration metrics, while only complete-supported labels train the models. E remains a gold-boundary oracle. The notebook 02h exports checkpoints, references, predictions and metrics for review. No full GPU training has occurred. See [GUITARSET_CQT_EXPERIMENT.md](GUITARSET_CQT_EXPERIMENT.md) for parameters, deviations from BTC, training budgets and limitations.


## Imported fold-0 CQT results — 20 November 2026

Verified runs/imports/p2-guitarset-cqt-results.zip (SHA-256 in reports/p2/guitarset-cqt/results/import_verification.json). Both jobs use revision 44e84a3cff2ea7b0e88d6de43d5232459d663482 with source hashes matching that commit. Both complete 30 epochs; F has 26 updates/epoch and E 83. Checkpoint selection matches the highest validation Tetrads: F epoch 16, E epoch 12. All validation metrics were independently reconstructed from saved NPZ predictions and original reference JSON and agree within 1e-12. No test inference occurred.

| Model | Root | MajMin | Triads | Tetrads | Supported oracle event accuracy |
|---|---:|---:|---:|---:|---:|
| F | 44.04% | 23.91% | 13.84% | 7.68% | 15.31% |
| E | 49.93% | 26.33% | 15.24% | 8.71% | 15.52% |

The duration metrics use the full raw performed reference vocabulary, whereas the event metric covers 986/1,680 complete-supported events only. F predicts frame boundaries; E has gold intervals. The differing tasks and update budgets prevent a clean architecture ranking. These are weak pilot results, not evidence of improvement over Billboard models or parity with a paper.

### Error and support analysis

On supported gold events, F root/triad/extension component accuracies are 47.46%/55.38%/54.67%; E reaches 49.70%/52.13%/57.00%. Gate accuracy is 100% because the scoped chord labels contain no N, so it does not establish silence detection. F predicts maj for 894/986 supported events and E for 714/986. F correctly recognizes only 2/210 min and 0/98 min7; E recognizes 2/210 and 1/98. Overall failure is broader than missing rare labels.

The fixed fold also creates a strong support shift. Train has no hdim7, dim7 or minmaj7 complete events, while validation has 90, 2 and 2. Dominant 7 has only 40 training events versus 164 validation events. This was visible in the committed pre-training support audit and limits conclusions from this one-progression train split. Do not redefine the existing split after seeing its results or present a new easier split as the same experiment.

Evaluated the imported best checkpoints on the local training cache without updates: F training Tetrads is 35.78% and E 46.26%, versus validation 7.68% and 8.71%. The local cache has identical pinned sources/config with small platform CQT numerical differences from Kaggle. This training-fit check suggests both incomplete fitting and a generalization gap; it does not isolate their cause. E's training loss at epoch 30 is lower than at the validation-selected epoch 12, so selected-checkpoint fit alone cannot diagnose optimizer capacity.

### Export defect and disposition

Found 12 zero-duration rows in E's exported LAB files: decimal rounding to nine digits collapsed tiny positive trailing gaps. Saved NPZ/reference metrics are valid; no training rerun is needed for this export issue. Future exports use 17 significant digits to preserve float64 endpoints, with a regression check for a tiny positive gap. The original imported archive remains unchanged. Reconstruct E intervals from event_logits.npz plus validation_references.json rather than using those rounded rows. F has no zero-length LAB rows.

### Next controlled step

Do not mix these checkpoints into the Billboard/JAAH system or promote either teacher. Keep this progression-held-out experiment as a documented negative result. Next first verify learning on a small quality-diverse audio subset, then compare training-fit and validation curves with explicit per-head support. A separate leave-player-out development protocol retaining all three progressions in training is a reasonable source-learning diagnostic; it answers unseen-player generalization rather than unseen progression and must be named separately. No such new split or training job has been run here. Increasing corpus diversity remains the main plan, conditional on matching audio and a validated baseline.

Evidence: reports/p2/guitarset-cqt/results/ contains original selected metrics/history, independently rescored duration metrics, component diagnostics, import verification and training_fit.json. Tests pass after the export fix; full GPU retraining was not performed.

## Quality-diverse CQT memorization — 21 November 2026

Completed the proposed training-only learning diagnostic locally on CPU. The reproducible command and selection/stop rules are in GUITARSET_CQT_EXPERIMENT.md; source is src/acr/check_guitarset_memorization.py. Eighteen fixed events cover nine supported training qualities, with two events per quality. Both teachers start fresh with the original seed, width, learning rate and train-fitted CQT normalization. F supervises individual frames inside the selected intervals; E supervises pooled gold events. Context remains actual neighboring audio. No validation/test examples are used for training, evaluation or stopping.

| Teacher | First checked pass (updates) | Selected-frame accuracy | Selected event accuracy |
|---|---:|---:|---:|
| F | 40 | 98.81% | 18/18 |
| E | 40 | Not applicable | 18/18 |

This rules out an inability to memorize these particular CQT examples with the existing architecture and loss/decoder path. It does not rule out annotation issues, limited representation quality, optimization problems at full scale, or generalization failure. Gold-guided crops and equal quality representation make this much easier than natural full-corpus training. It says nothing about no-chord detection or qualities absent from this subset. Do not replace the negative progression-held-out results with these training accuracies.

Decision: retain both original checkpoints and the original folds without promotion. Before changing architecture or adding another source, the next informative experiment is a separately named source-learning protocol with all progression groups represented in training and a held-out player, alongside training/validation curves and per-quality support. This would test a different generalization question; it has not been prepared or run by this diagnostic. Additional dataset work remains warranted, but this result alone does not establish cross-corpus benefit.

Evidence: reports/p2/guitarset-cqt/results/memorization.json includes selections, predictions, histories and provenance. Local model weights remain under ignored runs/. All 56 tests pass, including deterministic selection, root diversity preference, and exclusion of OOV/N/overlong events.

## Player-held-out control prepared — 21 November 2026

Prepared a separate deterministic player split, config and notebook 02i for both fresh teachers. Train players 00–03 provide 240 recordings, player 05 provides 60 validation recordings, and player 04 is reserved (58 usable); the two timing errata remain excluded. All progression groups are present in training. Selection is based on player IDs, not model scores. Original progression folds and results remain intact. See GUITARSET_PLAYER_EXPERIMENT.md and reports/p2/guitarset-cqt/player-split.json for protocol and label support.

This changes the generalization question and training-set size. It introduces known repertoire overlap and reuses historical data: old progression-test recordings can now enter training, and the reserved player had earlier validation exposure. Consequently neither a score increase nor a later reserved-player score establishes independent benchmark parity. No new test result is claimed.

Preparation/cache validation now accepts an explicit manifest in the config while keeping the previous default and hash checks. The new config enables full training/validation metrics and per-quality diagnostics in each epoch's history; checkpoint selection remains validation Tetrads. Natural sampling, model architecture, seed, optimizer and label mapping are unchanged. Full GPU training remains pending on Kaggle; run 02i and return its separately named ZIP.

## Player-held-out GPU results verified — 21 November 2026

Imported p2-guitarset-cqt-player-results.zip, SHA-256 8d31cd4f9b281c89c8c9b8ae43704cef8d0278bbc2d418dbd14528c94816c4f6. Both fresh jobs complete 30 epochs at b64c8dd5b9e491c5c14b995d8e085c7528c9f8ef, seed 20261119. Independently rescored full timelines from NPZ and original reference JSON; all duration numerators/denominators agree within 1e-8 seconds, and supported event accuracies within 1e-12. Official annotation event lists match all 60 validation recordings. Split IDs, train-only normalization IDs, source Git hashes, checkpoint choice and decoded predictions match. The Linux manifest hash matches the Git blob; Windows CRLF explains working-copy byte differences without changing rows. No reserved player 04 predictions or zero-length LAB rows were found.

| Teacher | Root | MajMin | Triads | Tetrads | Supported oracle event accuracy |
|---|---:|---:|---:|---:|---:|
| F | 48.31% | 47.88% | 30.83% | 24.45% | 34.56% |
| E, gold boundaries | 56.63% | 56.91% | 37.06% | 29.41% | 38.48% |

There are 720 annotated validation events, of which 434 have complete-supported labels and 286 are OOV. Event accuracy is restricted to those 434 events (F 150 correct, E 167). Duration metrics preserve the raw labels and their metric-specific eligibility rules. F/E do different tasks and receive 47/119 updates per epoch respectively, so the table is descriptive rather than an architecture ranking.

### Training fit and error concentration

F selects epoch 20: training/validation Tetrads is 36.58%/24.45%. At epoch 30 training improves to 40.76%, with validation 24.27%. E selects the last epoch 30: training/validation is 59.95%/29.41%. E may remain sensitive to training budget, but an endpoint maximum does not prove more training will help validation. The evidence supports both imperfect source fit and a generalization gap, not a single established cause.

On supported gold events, root/triad/extension accuracy is F 55.99%/72.12%/62.21%, E 59.22%/71.66%/63.36%. Gate is 100% in this chord-only scope and cannot establish no-chord detection. Major bias remains: F emits maj on 354/434 supported events and E on 287/434. F/E correctly classify 121/124 of 194 maj events and 26/26 of 58 min events, respectively. For 7 the counts are 1/8 of 26; hdim7 1/4 of 32; min7 0/3 of 36; maj7 1/2 of 52. Both miss every min6 (12), maj6 (22), and dim7 (2) event. These counts cover complete chords, not just correctly identifying the quality.

### Interpretation and next decision

Having all progression groups in training is associated with substantially higher validation scores than the earlier pilot, but training size, player content and validation targets also changed. This does not isolate the effect of progression coverage or establish cross-corpus improvement. Keep both experiments and do not promote these checkpoints.

Before another large run, prioritize inspecting root and extension confusion and representative audio/annotation intervals for maj versus seventh/sixth errors, then define one controlled training change if supported by that inspection. Additional epochs for E can be a budget ablation; they should not silently replace the 30-epoch control. A literature-backed stronger audio baseline and additional matching audio remain relevant; this one-seed development result does not resolve dataset diversity or architecture choice. Evidence is reports/p2/guitarset-cqt/player-results/ (original histories/metrics, independent rescoring and import verification). No new training was performed during import.

## Component confusion and audio review preparation — 21 November 2026

Ran src/acr/diagnose_guitarset_player.py on the saved validation logits. Scope is 434 complete-supported gold chord events, with F mean frame probabilities and E event probabilities. Confusion matrices use raw per-head argmax, before the constrained legal-quality decoder; complete predictions use that decoder. This distinction matters: E has 166 events with all three raw component heads correct, but 167 correct complete predictions after legal decoding. Matrix totals and complete-event accuracy are checked against the imported metrics. No training or reserved-player evaluation occurs.

| True extension | Events | F correct raw extension | E correct raw extension | F predicted none | E predicted none |
|---|---:|---:|---:|---:|---:|
| none | 252 | 250 | 231 | 250 | 231 |
| 6 | 34 | 0 | 0 | 28 | 27 |
| b7 | 94 | 19 | 40 | 75 | 51 |
| 7 | 52 | 1 | 4 | 48 | 40 |
| bb7 | 2 | 0 | 0 | 2 | 2 |

Of 182 events requiring an extension, F emits none for 153 (84.07%) and E for 120 (65.93%). High overall extension accuracy is therefore largely carried by none. This pattern supports investigating extension underprediction; it does not alone distinguish weak acoustic evidence, imbalanced supervision, optimization, or annotation mismatch.

Within those wrong-none extension predictions, F has 59 and E 86 with raw extension confidence at least 0.9. These are single-head confidence counts, without a chord-gate or F/E-agreement filter. High softmax confidence alone would therefore preserve many incorrect extension labels; it must not be used as a substitute for measured precision.

Extension is not the only bottleneck. F/E raw root is wrong on 191/177 events. Only 49/47 events have both raw root and triad correct while extension is wrong; changing extension alone cannot repair all complete-chord errors. Conversely, 120/108 events have correct extension but incorrect complete chord. An extension-only intervention must therefore report root and full-chord metrics as well as extension recall/precision.

Created five deterministic paired-error review clips with one-second context, spanning 7, maj7, min7, maj6 and min6. Paths and exact original labels/timestamps are in reports/p2/guitarset-cqt/player-diagnostics/review_candidates.csv; local WAVs are in ignored runs/p2-guitarset-player-review/. Examples include B:maj7 predicted B:maj by both teachers and G#:min6 predicted G#:min7 by both. These are candidate listening cases, not verified annotation defects. No independent listening verdict or automatic retiming is claimed. Review clips before changing labels, especially because performed GuitarSet labels have sheet-informed semantics.

Decision: preserve this natural-sampling control. A future controlled loss/sampling ablation should keep split, seed, architecture, budgets and checkpoint criterion fixed, derive any weights solely from training support, and inspect per-class precision as well as recall. Existing P1/P2 balanced/weighted results show that simply emphasizing rare targets can lower overall accuracy; those results are motivation for caution, not proof of transfer to CQT. Do not launch a new weighting scheme or replace TCN solely from this confusion table. First distinguish training-fit errors from held-out-player errors with the same component protocol, then select one intervention. Source diversity and an author audio baseline remain separate open work.

Evidence: reports/p2/guitarset-cqt/player-diagnostics/ contains full confusion matrices, aligned supported-event predictions, summary counts and audio review candidates. The diagnostic itself checks event alignment between F/E, matrix totals and reproduction of imported complete-event scores.

## Frozen training-versus-validation component audit — 21 November 2026

Prepared the 300-recording player cache locally from pinned archives, fitted normalization only on players 00–03, and evaluated the original selected checkpoints (F epoch 20, E epoch 30) without updates. Matched all selected recording/audio hashes, source archives, config, checkpoint epochs and manifest identity. Local CPU training Tetrads is exactly 36.58481673558038% for F and 59.9458362532555% for E, matching the original Kaggle history with zero difference. No reserved-player data or new training was used. Reproduction commands are in GUITARSET_PLAYER_EXPERIMENT.md.

### Same event protocol on both partitions

The table uses 1,894 complete-supported training chord events and 434 validation chord events. Component accuracy uses raw head decisions; complete-event accuracy uses the legal decoder. F is pooled over gold intervals for this comparison and E has gold event boundaries. Values are event-weighted, not duration-weighted.

| Teacher / partition | Root | Triad | Extension | Complete event |
|---|---:|---:|---:|---:|
| F train | 70.01% | 82.58% | 84.74% | 53.33% |
| F validation | 55.99% | 72.12% | 62.21% | 34.56% |
| E train | 93.29% | 95.99% | 96.62% | 87.43% |
| E validation | 59.22% | 71.66% | 63.36% | 38.48% |

The earlier training Tetrads values must not be confused with these supported-only event accuracies: duration metrics use full raw reference timelines including valid OOV labels and metric-specific exclusions, while this table excludes OOV events. High E component fit on supported train events is consistent with a lower full-timeline duration score.

Of 502 training events requiring an extension, F predicts none on 253 (50.40%), E on 52 (10.36%). On validation the corresponding rates are 84.07% and 65.93%. E therefore demonstrably learns many extended targets in its training distribution; its held-out failure is not a blanket inability of this architecture to learn extensions. F still has substantial incomplete fitting even on supported training data. Different F/E tasks and update budgets remain confounders, so this is not proof of architectural superiority or of one causal mechanism. Only one held-out player and one seed were examined; label/support distributions also differ across players.

### Confidence sweep, not a fixed 0.9 policy

Measured extension precision/coverage at thresholds 0, 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99 for both partitions, including a diagnostic subset of true extended chords. On that validation subset (182 events), F has 20 correct raw extension predictions with no filter; threshold 0.9 accepts 60 predictions but only one is correct. E at threshold 0.9 accepts 118 with 28 correct; at 0.99 it accepts 73 with 16 correct. Increased softmax thresholds do not guarantee increased precision on these events.

These are single-head descriptive curves on supported gold chord events. The true-extended subset uses ground truth and is not an operational selector. No chord-gate or F/E-agreement filter, fitted calibration or final acceptance threshold is implied. Do not compare these numbers directly with the earlier Billboard operational confidence audit, whose denominator includes N and uses additional gate rules.

### Decision

Do not treat class weighting as a demonstrated solution for both teachers. For E, the dominant observed issue is the large train-to-held-out gap across all three chord components; more training or higher confidence alone is not justified as a fix. For F, train fitting remains an additional issue. A future controlled augmentation/regularization or literature-baseline experiment should measure train fit and held-out performance together, and target overall full-chord accuracy before rare recall. Selecting one intervention requires an explicit protocol and source reference; this audit does not silently launch a new training scheme. Calibration should follow improvements in predictive reliability and requires separate validation design.

All 58 tests pass, including a synthetic known-answer audit with explicit training references, identical expected F/E outputs, and no audio files required when clip extraction is disabled. Evidence: reports/p2/guitarset-cqt/training-fit/ contains training confusion, selected-checkpoint metrics/provenance, the train/validation comparison and extension threshold sweep. Original checkpoints and validation results are preserved.

## CQT transposition ablation prepared — 21 November 2026

Selected one literature-motivated augmentation after inspecting BTC author preprocessing, ChordFormer author training code and the MIREX 2025 degree-based ACR submission (section 3.2). Source links and implementation differences are recorded in GUITARSET_TRANSPOSITION_EXPERIMENT.md. BTC transposes waveforms, while the selected control shifts CQT bins and chord roots. This is an approximate feature-space transformation, not a waveform augmentation or author-baseline reproduction.

The hypothesis is that exposing multiple transpositions of existing training examples may reduce reliance on observed absolute-root/key combinations. It does not add players or chord qualities and is not an established cure for the observed extension errors. All original class imbalance and progression/repertoire limitations remain. No additional source dataset has been added.

New config p2-guitarset-cqt-transpose.json adds only uniform training-example shifts -5 through +6 semitones and a pointer to the original cache config. At 24 bins/octave each semitone moves two bins; complete labels rotate root modulo 12 while preserving quality, N and ignored labels. Finite frequency edges are cropped and filled with normalized log-floor values, without circular wrap. Temporal padding stays zero; real all-zero frames are rejected to avoid confusing data with padding. Labels and timing remain aligned.

Both teachers remain fresh width-48 TCNs with seed 20261119, natural sampling, the original losses, 30 epochs and the same 47/119 updates per epoch. A separate augmentation RNG avoids perturbing shuffle/init randomness. Clean train and validation evaluation remain unchanged. There is no confidence filtering, loss reweighting, dropout or time masking in this ablation.

Notebook 02j prepares/verifies the original CQT cache, runs separate augmented smoke jobs, then full augmented F/E jobs and prints the stored natural-control comparison. The output ZIP is p2-guitarset-cqt-transpose-results.zip. CPU smoke execution completes for both teachers with finite losses; smoke zero validation scores are not a performance finding. Unit checks cover positive/negative shift direction, frequency-edge fill/no wrap, zero shift, temporal padding, N/OOV preservation, every supported root/quality and known-tone direction. Full GPU training is pending; neither augmentation benefit nor promotion is claimed.

Evaluate full-chord duration metrics first, along with clean fitting curves, root/extension errors and per-quality support. A promising single-seed outcome requires matched repeats. Differing F/E responses should be retained rather than forcing a common augmentation policy. This new control does not replace the earlier negative or natural runs.

## Transposition GPU screening verified — 21 November 2026

Imported p2-guitarset-cqt-transpose-results.zip (SHA-256 289db0946186f769bfec1e817e47119f07b2a8b399cdceb2d738d4153f3c0bc7). Both jobs complete 30 epochs at d7c834bb9d6ec2325ffccc7f496f40dce61b8389. Verified exact source Git hashes, original reference events against official annotation, identical train/validation IDs and normalization to control, unchanged update budgets, and earliest validation-Tetrads checkpoint selection. Recomputed all duration numerators/denominators from saved NPZ/reference timelines within 1e-8 seconds and supported event accuracy within 1e-12. All LAB intervals are positive. Replaying seed+1 augmentation sampling exactly matches each epoch's shift histogram; F/E each visit 740/1,894 sampled examples per epoch with 47/119 updates. No new training was done during import.

| Teacher / arm | Root | MajMin | Triads | Tetrads | Supported event accuracy |
|---|---:|---:|---:|---:|---:|
| F control | 48.31% | 47.88% | 30.83% | 24.45% | 34.56% |
| F augmented | 49.02% | 49.90% | 32.00% | 25.51% | 38.94% |
| E control | 56.63% | 56.91% | 37.06% | 29.41% | 38.48% |
| E augmented | 57.42% | 56.94% | 36.68% | 31.46% | 40.55% |

Primary Tetrads changes are +1.0635 percentage points for F and +2.0477 for E. F selects epoch 30; E selects 28 (its epoch-30 score is 29.40%). These are validation-selected single-seed outcomes on the same known-repertoire held-out player, not independent-test or repeatability evidence. F/E have different tasks; compare each against its own control.

### Which events improve or regress?

Aligned the same 434 supported gold events. F has 133 correct under both arms, 17 correct only under control, 36 only under augmentation, and 248 wrong under both (net +19). E has 148 both-correct, 19 control-only, 28 augmentation-only and 239 both-wrong (net +9). Event accuracy is supported-only and is not the full-timeline duration metric.

| Quality | Support | F correct control → augmented | E correct control → augmented |
|---|---:|---:|---:|
| maj | 194 | 121 → 143 | 124 → 137 |
| min | 58 | 26 → 18 | 26 → 24 |
| 7 | 26 | 1 → 7 | 8 → 10 |
| hdim7 | 32 | 1 → 0 | 4 → 1 |
| min7 | 36 | 0 → 1 | 3 → 4 |
| maj7 | 52 | 1 → 0 | 2 → 0 |
| min6 / maj6 / dim7 | 12 / 22 / 2 | All zero → zero | All zero → zero |

Both augmented arms predict maj more often on supported references: F 354 → 378, E 287 → 334. Thus the primary gain is compatible with improved common-chord recognition while some harmonic distinctions regress; it is not broad rare/extended-chord improvement.

### Component fit and confidence implications

Validation raw root/triad/extension accuracy is F 60.60%/72.58%/62.67% and E 60.37%/72.35%/65.21%. Despite a higher aggregate extension score, neither augmented teacher correctly predicts raw 6, 7 or bb7 extension on this subset. Correct raw b7 counts are F 20 (control 19), E 35 (control 40); correct none counts are 252 (control 250) and 248 (control 231). Among 182 events truly requiring an extension, wrong-none predictions increase F 153 → 154 and E 120 → 134. An aggregate score increase therefore does not justify a pseudo-label acceptance threshold or a claim of better extension reliability.

Evaluated the selected augmented checkpoints on the local clean train cache without updates. Training Tetrads exactly matches Kaggle: F 28.71%, E 34.31%, versus control 36.58%/59.95%. On supported training events, raw root/triad/extension accuracy is F 63.57%/76.98%/77.35%, E 64.78%/81.10%/80.10%. Clean fitting is substantially reduced. The smaller train/validation gap alone must not be interpreted as solved generalization; stronger augmentation can increase optimization difficulty under the same budget. F's best epoch at the endpoint also does not establish convergence.

### Decision and next experiment

Keep this as a promising overall-accuracy screening candidate with documented regressions, not a promoted F/E policy. Next compare natural versus transposition at additional matched seeds 20261120 and 20261121 for both teachers, retaining this original seed in the aggregate. Keep source split, CQT cache/normalization, epochs, update counts and selection criterion fixed. Report mean/variation within each teacher and quality tradeoffs. Do not simultaneously tune shift strength, learning rate, training duration or architecture, and do not choose a seed for its best score. Repeats have not been prepared/run by this import audit.

Evidence: reports/p2/guitarset-cqt/transposition/results/ holds imported histories/metrics, independent rescoring and verification; diagnostics/ holds supported-event confusion and decisions; training-fit/ holds frozen clean-training metrics/confusion/provenance; comparison.json and per_quality_comparison.csv record control deltas and event tradeoffs. Raw archive/checkpoints remain in ignored runs/. No code change or full training rerun was required for this verification.

## Matched-seed transposition repeats prepared — 21 November 2026

Prepared repeat_guitarset.py and notebook 02k to run seeds 20261120 and 20261121 for both natural/transposition arms and both teachers. This is eight new training jobs, run sequentially. The existing seed 20261119 stays in the full aggregate using hashed original metrics; it is not retrained or replaced. The plan records source/revision, cache metadata and historical metric-file hashes, while generated configs and run records preserve each stage's provenance.

The original cache seed describes preparation configuration rather than model randomness. Training config resolution now permits exactly a seed change plus the existing augmentation setting through an explicit cache_config pointer; other cache-control config differences still fail. Cache contents, split and normalization remain fully checked. A regression test confirms seed reuse is allowed while changed fold or learning rate through this control path is rejected.

Per-teacher aggregate includes Root/MajMin/Triads/Tetrads and supported-event accuracy, means and sample standard deviations, and matched-seed augmented-minus-natural differences. A known-answer test verifies the paired arithmetic and n-1 standard deviation. Smoke results are explicitly separated and exclude historical full metrics. Saved per-quality metrics and clean train/validation histories support later tradeoff analysis. Completed stages require matching provenance; partial stages are preserved and rejected without automatic resume.

Read GUITARSET_TRANSPOSITION_REPEATS.md for running, recovery limits and interpretation. The new split is still reused-validation development on one player, and E still has gold boundaries. Repeated seeds measure optimization variability, not additional independent players. Full GPU repeats remain pending; no new performance claim follows from the notebook setup.

Local verification completes all eight two-epoch CPU smoke jobs and confirms fresh training seeds with fixed cache provenance. Re-running completed orchestration reuses stages and leaves comparison.json identical. Smoke aggregation excludes historical full metrics. All 61 tests pass; notebook code cells compile. Evidence: reports/p2/guitarset-cqt/transposition/repeat-setup/verification.json. Smoke scores are execution checks only.

## Three-seed augmentation results verified — 21 November 2026

Imported p2-guitarset-transpose-repeat-results.zip, SHA-256 d883a7a09e24e94d6ace385d2479e432c77ef23c0bb04ec6b2de541eca715a67, containing eight complete full jobs at f9a5174. Independently checked the source/plan hashes against Git, historical metric hashes, generated training configs, seed/split/normalization identity and unchanged update budgets. All jobs complete 30 epochs and select the earliest maximum validation-Tetrads checkpoint. Full metrics were reconstructed from saved NPZ/reference intervals (duration counts within 1e-8 seconds; supported-event accuracy within 1e-12); per-quality counts and stored decoded predictions match. All LAB intervals are positive. Replayed each augmented job's RNG and reproduced all per-epoch shift histograms. Recomputed the three-seed aggregate and paired deltas exactly. No training or code modification was needed for import.

### Primary result, within each teacher

| Seed | F natural | F augmented | F delta | E natural | E augmented | E delta |
|---|---:|---:|---:|---:|---:|---:|
| 20261119 | 24.45% | 25.51% | +1.06 pp | 29.41% | 31.46% | +2.05 pp |
| 20261120 | 24.53% | 24.09% | −0.44 pp | 28.08% | 31.35% | +3.27 pp |
| 20261121 | 25.02% | 24.09% | −0.93 pp | 27.70% | 32.47% | +4.77 pp |
| Mean | 24.67% | 24.57% | −0.10 pp | 28.40% | 31.76% | +3.36 pp |
| Sample SD | 0.31 pp | 0.82 pp | 1.04 pp | 0.90 pp | 0.61 pp | 1.36 pp |

The metric is validation duration-weighted Tetrads. Sample SD is optimization variability on this one split, not uncertainty over independent players. E improves in all three matched comparisons; F does not repeat its initial gain. E's mean supported-event accuracy also increases 36.79%→41.63% (+4.84 pp), whereas F remains 35.87% under both arms. No significance or final-test claim is made with three seeds and reused validation.

### Harmonic tradeoffs persist

Per-seed reference support is unchanged. Mean counts below average correct complete-chord events over three seeds; recurring intervals are not independent new examples.

| Quality / support per seed | F natural → augmented | E natural → augmented |
|---|---:|---:|
| maj / 194 | 124.00 → 139.33 | 124.67 → 140.00 |
| min / 58 | 25.00 → 9.00 | 23.00 → 24.33 |
| 7 / 26 | 1.67 → 6.33 | 6.00 → 9.33 |
| hdim7 / 32 | 1.33 → 0.00 | 3.00 → 1.00 |
| min7 / 36 | 2.33 → 1.00 | 2.00 → 4.00 |
| maj7 / 52 | 1.33 → 0.00 | 1.00 → 2.00 |
| min6 / 12, maj6 / 22, dim7 / 2 | All zero → zero | All zero → zero |

Mean maj prediction counts rise F 351.67→393.00 and E 311.33→337.00 among 434 supported reference events. F trades substantial minor recognition for major/dominant gains. E has broader gains but hdim7 regression remains, and sixth/diminished-seventh recognition is unresolved. Complete-label precision here remains scoped to supported gold events; it is not operational pseudo-label precision.

### Complementarity with natural F

Computed all four F/E arm combinations on identical supported gold events for every seed. Holding F natural and switching E natural→augmented increases events correct in at least one teacher: 179→186, 181→195, 183→208 out of 434. E-only successes increase 29→36, 23→37, 24→49. The mixed-policy pair therefore adds complementary correct information in every seed, but these are oracle union counts and do not constitute an achievable fusion metric.

Complete-label agreement precision/coverage for natural F + augmented E is respectively 59.57%/54.15%, 60.25%/55.07%, and 69.04%/45.39%. These are unfiltered, supported-only, gold-interval diagnostics, not the earlier operational Billboard confidence protocol. Teachers still agree incorrectly often; reliable pseudo-label generation is not established.

### Development policy and next scope

For the GuitarSet CQT branch, retain natural F and prefer transposition E for the overall-accuracy development reference. This is a per-teacher policy decision, not a forced shared augmentation strategy. Keep original seed 20261119 as the primary pair to preserve continuity; all repeated checkpoints remain available, and the highest-scoring new seed is not substituted. Existing Billboard/JAAH teachers and their policies are unaffected.

The consistent E gain does not resolve low absolute accuracy, sparse source diversity, full-label OOV coverage, or gold-boundary dependence. Further identical-seed reruns are not indicated. The next experiment should be separately specified around a stronger literature audio baseline or one intervention addressing remaining root/quality errors, using this matched-seed comparison as a control. A new split or more datasets would answer additional questions and must not be presented as the same benchmark. No new training job is launched by this verification.

Evidence: reports/p2/guitarset-cqt/transposition/repeats/ includes all new job histories/metrics, exact rescoring, import verification, independently rebuilt comparison, event decisions, component summaries, per-quality/per-seed and mean counts, and all-pair complementarity. Original archive and model weights remain under ignored runs/.

## Stronger frame/event baseline selection — 21 November 2026

Completed a primary-source paper/code review in response to the request for stronger baselines in both directions. The recommendation is BTC large-vocabulary as the first frame reproduction anchor and TE-DS as the first true audio event-sequence baseline, followed by pTE-DS encoder pretraining. ChordFormer is the next modern structural frame candidate. This is an implementation-priority judgment, not a measured ranking on GuitarSet. See [the relocated baseline review](../baselines/FRAME_EVENT_BASELINE_REVIEW.md) for linked sources, audited code details and the staged execution plan; source inventory is reports/baselines/source_review.json.

The frame anchor offers a relatively close CQT geometry and explicit author code, while requiring an intentional 170-label configuration and symbol-level remapping rather than project integer IDs. ChordFormer requires a distinct expanded CQT/crop front end and a richer structural target contract; model-only and ensemble/HMM results must be separated. Its GPU selection and training wrappers need compatibility work before use.

The event candidate predicts boundaries and chord tokens autoregressively, unlike the existing oracle E. Preserve three evaluation tracks: full-timeline frame recognition, oracle event classification, and free-running event sequence recognition. A pooled Transformer on gold intervals is a useful possible adaptation but must not be named a reproduction of TE-DS. Likewise, token probabilities do not supply the same independent component confidence interface as the current four heads.

Static review identified concrete reproduction questions: the event repo's token list/template totals 288 rather than the paper's 289; full_train apply_log=False differs from pre_train apply_log=True; the event feature clock is 100 ms rather than the current approximately 92.88 ms. These are source observations, not runtime-tested defects. Known-answer I/O, vocabulary, timestamp, mask and EOS checks must precede GPU training. Pin source commits and record any necessary adaptation. A current-cache experiment is a GuitarSet adaptation and cannot claim published-dataset score reproduction.

Next executable stage is pinned-code compatibility and I/O adapters for BTC and TE-DS, then small-set fitting before a prespecified full screening. Keep the verified TCN controls and current F-natural/E-transposition policy as references. Do not add pretraining, a new decomposition, a new corpus and a new architecture in one unlabelled change. Upstream models have not been imported/executed, no new performance is measured, and no new Kaggle run is required by this review.

## Handoff to Stage B — 21 November 2026

At the user's request, subsequent literature-baseline and corpus-generalization work is a separate research stage, rather than further P2 expansion. The plan and ongoing log are in docs/baselines/BASELINE_AND_CORPUS_PLAN.md and BASELINE_RESEARCH_LOG.md. The baseline review and source inventory were relocated to docs/baselines/ and reports/baselines/. Preserve all prior P2 outcomes, including negative results and exposure limitations; this organizational handoff does not claim every original P2 acceptance condition was met. Original P3 and student milestones remain distinct. This is the final Stage B handoff entry in the historical P2 report.
