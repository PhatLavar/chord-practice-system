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

## Component and reference-event evaluation · 19 November 2026

The additional diagnostic uses the saved validation logits and the original detailed annotation archive. Its reader reproduces every saved frame reference and its decoder reproduces every saved prediction. Component scores measure the standalone root, triad and extension argmax on supported chord frames, conditional on a chord being present. They do not require the no-chord gate or other components to be correct.

| Standalone component | Supported-chord frame accuracy |
|---|---:|
| Root | 81.48% |
| Triad | 86.15% |
| Extension | 77.37% |

These pooled component scores are dominated by common qualities. No-chord gate accuracy over all scored frames, including N, is 96.58%. Conditioning on quality reveals failures hidden by the aggregates.

| Reference quality | Root accuracy | Triad accuracy | Extension accuracy |
|---|---:|---:|---:|
| 7 | 90.90% | 89.88% | 18.25% |
| min7 | 82.38% | 78.83% | 17.36% |
| maj7 | 78.16% | 91.06% | 5.80% |
| sus2 | 86.65% | 0% | 100% |
| maj6 | 58.12% | 93.28% | 0% |
| hdim7 | 40.41% | 0% | 20.55% |

A 100% extension score for sus2 means the model predicts the absence of an extension correctly. It cannot compensate for the missed suspended triad. The zero triad accuracy on dim, aug, sus2, sus4 and hdim7 and zero sixth-extension accuracy on maj6/min6 indicate that root-only improvements will be insufficient.

### Reference-event protocol

Merge touching intervals with the same supported root/quality after inversion collapse, matching the P0 event-count convention. Unsupported intervals remain breaks. For each supported interval, average frame probabilities separately for each head using timestamps in [start,end), then apply the existing no-chord gate and valid-chord decoder. Each scored event contributes one count. This uses oracle reference times and evaluates classification on fixed intervals; it does not evaluate predicted segmentation.

Of 9,674 supported reference events including N, 9,673 contain at least one sampled frame. One event without a frame is explicitly omitted from the diagnostic denominator. Accuracy including N is 58.57%. Rare-quality macro event recall is 0%, averaged over all seven train-defined rare qualities with validation event support. This is an oracle-interval diagnostic; a final event-detection metric with predicted boundaries still needs its matching rules.

| Rare quality | Scored events | Independent validation songs | Complete-chord event recall |
|---|---:|---:|---:|
| dim | 23 | 8 | 0% |
| aug | 5 | 2 | 0% |
| sus2 | 62 | 5 | 0% |
| min6 | 6 | 2 | 0% |
| minmaj7 | 4 | 2 | 0% |
| hdim7 | 9 | 4 | 0% |
| dim7 | 2 | 1 | 0% |

Low support, especially two dim7 events in one song, limits conclusions about those qualities. Gaps and masked intervals are excluded from scored targets. The current per-quality song counts refer to songs with scored events.

### Boundary diagnosis

As an exploratory diagnostic, mark scored frames within 0.1 seconds of a reference change between two supported labels, including N transitions. Their complete-chord accuracy is 38.03% across 38,090 frames. The other 417,956 scored frames reach 58.85%. This comparison does not prove a causal effect of boundaries because chord and song composition can differ across the two groups.

Oracle interval pooling still yields zero rare-quality recall. Transition errors therefore coexist with failures to classify rare chord content when interval times are supplied.

[Full frame-component tables](../../reports/p1/kaggle-diagnostics/frame_qualities.csv) · [Reference-event tables and song support](../../reports/p1/kaggle-diagnostics/event_qualities.csv) · [Per-event decisions](../../reports/p1/kaggle-diagnostics/events.csv) · [Diagnostic summary](../../reports/p1/kaggle-diagnostics/summary.json)

## Experiment log and planned changes

Update this report whenever a P1 observation, protocol or planned training change changes. Mark proposals as planned until a run produces measured evidence. Record the date, code revision, changed factor, reason, fixed controls, results and next decision. Keep full numeric outputs under reports/p1/ and run artifacts under runs/.

| Date | Status | Observation or decision | Implication |
|---|---|---|---|
| 19 November 2026 | Measured | CPU two-song run improved from 94.51% to 95.58% after lowering learning rate and increasing epochs. | The pipeline can fit a small training subset. |
| 19 November 2026 | Measured | Kaggle overfit reached 95.49%; full baseline reached 57.11% validation frame accuracy at epoch 27. | Preserve commit 8ec616c as baseline. |
| 19 November 2026 | Measured | Oracle interval accuracy is 58.57%; all seven rare qualities retain 0% recall. | Correct interval times alone do not recover the missing qualities. |
| 19 November 2026 | Planned | First test class-frequency weighting separately for triad and extension losses, derived only from labeled training representatives. | Target underlearned components while keeping sampling, backbone and decoder fixed. Specify weights before training and record per-class counts. |
| 19 November 2026 | Planned | Run pitch transposition as a separate ablation after the weighting comparison. | Test root coverage independently; it preserves quality and cannot itself increase the proportion of rare qualities. |
| 19 November 2026 | Planned | Document TCN and component-head research precedents before architectural changes. | Distinguish literature choices from this project's pilot configuration. |

The weighting proposal is motivated by the diagnostics; its effectiveness remains untested. Compare frame accuracy, per-quality component scores and rare oracle-event recall against this baseline on the same validation songs. Keep the fixed test partition reserved. Full student evaluation will require its own finalized event matching and uncertainty reporting.

## Weekly presentation notes

- Completed the Kaggle training pipeline and verified its returned predictions.
- Established a 57.11% validation frame baseline with a 37,225-parameter TCN.
- Found that roots are often recognized while sevenths, sixths and rare triads are missed.
- Confirmed zero rare recall even when reference interval times are provided.
- Completed the weighted component-loss experiment. Macro chord recall improved, while overall accuracy fell and rare oracle-event recall stayed at zero.
- These results concern teacher F; teacher complementarity and pseudo-label benefits remain open.

For local inspection, imported runs are runs/p1-overfit-kaggle/ and runs/p1-baseline-kaggle/. The archive is retained under runs/imports/p1-results.zip. Large artifacts are ignored by Git; the linked summaries and tables are versioned.


## Weighted-loss experiment specification · 19 November 2026

Status: completed on Kaggle and independently verified. The results are recorded below. Use experiment name p1-weighted-kaggle and [configs/p1-weighted.json](../../configs/p1-weighted.json).

For triad and extension separately, count observed frame targets across exactly the 367 labeled training representatives before augmentation. Mask N, OOV, unknown and uncovered component targets. For class c with positive count n_c, compute a_c = min(10, sqrt(max(n) / n_c)). Divide all positive weights by their frequency-weighted mean, sum(n_c a_c) / sum(n_c). An unobserved class receives zero weight and has no training targets. Weighted cross-entropy divides its loss sum by the sum of the observed target weights in that batch, following the existing mean-reduction convention.

Apply these weights only to triad and extension. Keep root and no-chord losses, random seed 20261119, initialization, 48-channel TCN, data normalization, training windows, shuffle, batch size 16, learning rate 0.001, 30 epochs, decoder and checkpoint selection by validation frame accuracy as in the baseline. Pitch augmentation and sampling are unchanged. The complete per-class frame counts and proposed weights are recorded in [weighted setup](../../reports/p1/weighted-setup/loss_weights.json); each actual run also records its own weights.

Six unit checks cover prior behavior, weight counting, bounded ratios, masked targets, and finite weighted gradients. A CPU batch from the actual labeled training cache passed forward, backward and optimizer update. [Local verification](../../reports/p1/weighted-setup/verification.json)

The local checks verified the implementation. The returned Kaggle results below now provide the weighted validation scores. Compare its saved diagnostics against baseline frame accuracy 57.11%, macro chord recall 19.38%, oracle-event accuracy 58.57%, and rare oracle-event macro recall 0%. Also inspect per-quality precision to detect excessive rare-class predictions. Loss weighting can shift class probabilities and decrease frequent-class accuracy; use the unchanged decoder for this ablation and record any tradeoff.

The component-loss weighting proposal has now been implemented, run and evaluated. The next section records the actual revision, comparison and decision.


## Weighted-loss results and decision · 19 November 2026

The returned p1-weighted-results.zip contains the completed 30-epoch CUDA run at revision d4557df803ddd680a2164c20c6669e56e9d67232, with checkpoint epoch 26 selected by validation frame accuracy. Source hashes match that commit. Training/evaluation IDs, source data, vocabulary, model capacity, software environment, seed and all baseline configuration values match the original experiment; component weighting is the added setting. The actual counts and weights exactly match the locally computed training-only setup.

Independently rerunning the diagnostic reproduced the returned summary and confusion matrix, including predictions decoded from saved logits. The run scored the same 456,046 validation frames and 9,673 oracle-interval events. One short reference event contains no sampled frame and remains excluded from the event denominator.

| Metric | Baseline | Weighted | Change (percentage points) |
|---|---:|---:|---:|
| Complete-chord frame accuracy | 57.11% | 54.12% | -3.00 |
| Macro recall over observed complete-chord classes | 19.38% | 22.73% | +3.35 |
| Oracle-interval event accuracy including N | 58.57% | 55.96% | -2.61 |
| Rare-quality macro oracle-event recall | 0% | 0% | 0.00 |

The weighted model correctly classified 246,791 validation frames versus 260,458 for the baseline. Its macro recall improvement indicates a redistribution of performance across observed root-quality classes; it does not establish improvement on the seven predeclared rare qualities.

| Quality | Baseline frame recall | Weighted frame recall | Baseline event recall | Weighted event recall |
|---|---:|---:|---:|---:|
| maj | 79.42% | 72.16% | 79.43% | 72.78% |
| min | 47.91% | 42.09% | 54.90% | 48.37% |
| 7 | 13.10% | 16.31% | 10.48% | 14.62% |
| min7 | 15.43% | 29.85% | 11.97% | 29.68% |
| maj7 | 5.62% | 34.06% | 2.62% | 28.80% |
| sus4 | 0% | 6.35% | 0% | 3.16% |
| maj6 | 0% | 1.57% | 0% | 0.93% |
| hdim7 | 0% | 1.37% | 0% | 0% |

For hdim7 only four reference frames are exactly correct. The model predicts hdim7 on 2,722 scored frames, yielding complete-chord precision of 0.147%. Thus those four correct frames are accompanied by many false positives and provide little evidence of reliable rare-chord recognition. Every predeclared rare quality still has zero correctly classified oracle events.

The aggregate standalone root/triad/extension accuracies changed from 81.48%/86.15%/77.37% to 80.98%/83.25%/74.97%. Per-quality extension recognition improved for sevenths, but accuracy on common unextended chords declined. Boundary-adjacent frame accuracy decreased from 38.03% to 34.91%, and the other scored frames from 58.85% to 55.87%.

[Weighted metrics and provenance](../../reports/p1/kaggle-weighted/run.json) · [Numeric baseline comparison](../../reports/p1/kaggle-weighted/comparison.csv) · [Weighted quality precision/recall](../../reports/p1/kaggle-weighted/per_quality.csv) · [Oracle-event diagnostic](../../reports/p1/weighted-diagnostics/event_qualities.csv) · [Verification](../../reports/p1/kaggle-weighted/verification.json)

### Interpretation and next decision

Retain the unweighted run as the initial baseline and retain the weighted run as a completed ablation. The inverse-square-root component weighting with a tenfold ratio cap improved some seventh qualities but failed the rare-event objective. A single fixed-seed experiment supports this comparison of returned checkpoints; it does not establish robustness across seeds or broader superiority.

The current checkpoint policy selects the highest overall validation frame accuracy. We have evaluated only the selected checkpoint's detailed logits. We cannot conclude that no earlier weighted epoch ever recognized a rare event. Any change to checkpoint selection must be documented as a separate protocol change and applied consistently to both comparators.

Next proposed diagnostic, before launching another training run: count train windows containing each rare quality and inspect head/confusion behavior for the severely undersupported triads. Quality-balanced sampling is a candidate because changing component weights alone did not produce correct rare events. Define the sampling unit, sampling probabilities, epoch budget and controls before testing it. Pitch transposition remains a separate planned ablation.

| Date | Status | Evidence and decision |
|---|---|---|
| 19 November 2026 | Completed | Weighted experiment: frame accuracy -3.00 pp, macro chord recall +3.35 pp, rare oracle-event recall unchanged at 0%. Keep both checkpoints. |
| 19 November 2026 | Proposed | Audit training-window exposure before implementing a quality-balanced sampler. Full new training has not started. |

For the weekly presentation, report this as a measured tradeoff: increasing rare component loss weights helped some seventh chords, reduced major/minor accuracy, and did not recover rare event recognition. Include both aggregate and per-quality metrics, plus the low support of the rarest validation classes.

The diagnostic summary is printed twice in the Kaggle notebook because the script prints its result and the notebook prints summary.json again. Identical repeated output does not indicate a second training run.
