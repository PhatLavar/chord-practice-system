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


## Training exposure audit and balanced-sampling proposal · 19 November 2026

The train-only audit read exactly 367 labeled representatives. The original 512-frame scheme contains 3,471 eligible windows, yielding 217 optimizer steps per epoch with batch size 16. Windows can contain several qualities. Counts below describe source observations, before repeated sampling.

| Quality | Frames | Independent train songs | Natural windows containing the quality |
|---|---:|---:|---:|
| aug | 772 | 10 | 26 |
| dim | 3,206 | 30 | 92 |
| dim7 | 556 | 3 | 12 |
| hdim7 | 2,904 | 19 | 65 |
| min6 | 4,972 | 19 | 66 |
| minmaj7 | 569 | 5 | 12 |
| sus2 | 3,627 | 18 | 53 |

[All qualities and simulated exposure](../../reports/p1/sampling-setup/quality_exposure.csv) · [Audit metadata](../../reports/p1/sampling-setup/summary.json)

### Sampler specification

Use the unweighted baseline configuration and change only the training-window sampling policy. Keep the 30-epoch budget and 3,471 sampled windows per epoch. For each shuffled natural-window index, draw a 50% chance of retaining the natural window. Otherwise select uniformly among the 14 observed chord qualities, select uniformly among labeled training songs containing that quality, then select uniformly among matching frames in that song. Center a 512-frame window on the selected frame, clipped at song edges. Keep the original context padding and masked targets.

This mixture retains naturally occurring N and common chords while creating more opportunities to observe rare components. It balances the anchor choice; the 512-frame targets themselves still contain their natural neighboring chords, so total quality-frame counts are not uniform. No-chord is handled by the natural branch and surrounding frames. Sampling is with replacement. Source-song diversity is unchanged. For dim7 in particular, repetition of three songs can cause overfitting.

The train-only simulation with seed 20261119 yielded approximately 7.92 times the original aug frame exposures, 12.49 times dim7, 22.19 times minmaj7 and 3.50 times hdim7 in one sampled epoch. These are repeated observations, not new independent data. The exact training sequence can differ from this audit because model initialization, shuffling and training consume random numbers too. The actual per-quality frame exposures are therefore logged in each epoch's history.json.

### Fixed controls and tests

Use [p1-balanced.json](../../configs/p1-balanced.json). Component loss weights are disabled, matching the original baseline. Seed, optimizer, learning rate, epochs, architecture, window length, decoder, split and checkpoint selection criterion match the baseline. The different sampling changes the training examples seen at each step; total nonmasked frame counts can vary. Record sampling.json with the policy, quality support, windows and optimizer-step budget.

Seven unit checks include deterministic sampling for a fixed seed, preservation of rare anchors, the natural epoch budget and short-song padding. A real-data CPU batch passed forward, backward and optimizer update with finite loss and gradients. [Verification](../../reports/p1/sampling-setup/verification.json)

Status: completed on Kaggle and independently verified; see the returned results below. The weighting-only run remains a separate completed ablation. The next sampler result must be judged on complete rare-event recall, per-quality precision, common-quality accuracy and aggregate scores. A reduction in training loss alone does not establish improvement.

### Relevance to the second teacher

Teacher E can use the same train/validation split, vocabulary and independently fitted TCN backbone. For its oracle-event training, quality/song-balanced event sampling is a candidate counterpart, to be specified when P2 begins. Improvement in F would justify testing the shared strategy in E; it would not establish E's accuracy or teacher complementarity. Both teachers need standalone validation before pseudo-labeling.

### Weekly update

- The weighting trial improved seventh qualities but retained zero rare-event recall.
- Rare-event exposure is concentrated in very few windows and songs, including only three dim7 training songs.
- The new experiment changes sampling alone while retaining unweighted losses.
- Full sampler training has completed. Its first rare-event gain consists of two correct events; repeat-seed validation is the next proposed check.


## Balanced-sampling results and decision · 19 November 2026

The returned p1-balanced-results.zip completed 30 epochs on CUDA at revision d250916f82a1a5e2f0310920209450465636b23e. Checkpoint selection chose epoch 29. The training/evaluation IDs, source data, vocabulary, parameter count, environment and baseline hyperparameters match the original baseline; the added sampling policy is the intended changed factor. Actual per-quality target exposures are recorded for every epoch, with 3,471 window draws and 217 optimizer updates per epoch.

The ZIP passed integrity checks. All source hashes match the recorded Git commit. Independent recomputation from logits and detailed annotations reproduced the submitted event diagnostic and confusion matrix. The baseline and weighted outputs remain preserved.

| Metric | Unweighted baseline | Weighted loss | Balanced sampling |
|---|---:|---:|---:|
| Complete-chord frame accuracy | 57.11% | 54.12% | 56.87% |
| Macro recall over observed complete-chord classes | 19.38% | 22.73% | 23.05% |
| Oracle-event accuracy including N | 58.57% | 55.96% | 57.92% |
| Rare-quality macro oracle-event recall | 0% | 0% | 4.19% |

The balanced run correctly labeled 259,365 of 456,046 scored frames, 1,093 fewer than the original baseline. Compared with baseline, frame accuracy fell 0.24 percentage points, macro chord recall rose 3.67 points, and oracle-event accuracy fell 0.64 points. The sampled run therefore preserves much more of the original overall performance than the weighting-only run, while improving the predeclared rare-event score in this single-seed comparison.

### What produced the rare score?

| Rare quality | Reference events | Correct complete events | Recall | Predicted events of that quality | Complete-chord precision |
|---|---:|---:|---:|---:|---:|
| dim | 23 | 1 | 4.35% | 5 | 20.00% |
| aug | 5 | 0 | 0% | 0 | Undefined |
| sus2 | 62 | 0 | 0% | 2 | 0% |
| min6 | 6 | 0 | 0% | 0 | Undefined |
| minmaj7 | 4 | 1 | 25.00% | 9 | 11.11% |
| hdim7 | 9 | 0 | 0% | 9 | 0% |
| dim7 | 2 | 0 | 0% | 1 | 0% |

Here predicted events are predictions on the same scored reference intervals. A true positive requires the root and quality to match. Predictions on excluded OOV intervals are outside this precision denominator.

The macro score is (1/23 + 1/4) / 7 = 4.19%. Only 2 of 111 supported rare reference events are correctly classified, so pooled rare-event recall is 1.80%. The two correct intervals are C:dim in recording 0343 (16.742–19.974 s) and D:minmaj7 in recording 1253 (108.489–110.295 s). The gain is concentrated in two intervals from two songs. Five of seven rare qualities remain unrecognized at the event level. Single-seed and limited-support results should not be presented as robust rare-chord recognition.

For common and seventh qualities, frame recall is maj 75.69%, min 43.52%, 7 28.06%, min7 25.68%, maj7 16.99%. The balanced model improves all three seventh recalls relative to baseline, while reducing major/minor recall. Its standalone root/triad/extension frame accuracies are 80.76%/85.18%/77.50%.

[Three-run comparison](../../reports/p1/kaggle-balanced/comparison.csv) · [Verification](../../reports/p1/kaggle-balanced/verification.json) · [Quality frame scores](../../reports/p1/kaggle-balanced/per_quality.csv) · [Oracle-event recall and support](../../reports/p1/balanced-diagnostics/event_qualities.csv) · [Oracle-event precision](../../reports/p1/kaggle-balanced/event_precision.csv) · [Epoch history with actual sample exposures](../../reports/p1/kaggle-balanced/history.json)

### Decision and next evidence

Keep balanced sampling as the leading candidate for the next teacher-training checks, alongside the preserved unweighted reference. It is a candidate policy, with evidence of a small rare-event improvement; it is not a validated solution for both teachers.

The next proposed experiment is to repeat both the baseline and balanced configuration with matched additional seeds before further tuning. Report variation and per-quality correct events, since a two-event gain may be sensitive to initialization and sampling. Repeating both comparators avoids attributing ordinary seed variation to the sampler. No new repeated-seed runs have started.

After that check, P2 can train E with its own weights and reference-interval event classification, using an explicitly specified event-sampling counterpart. F's correct and incorrect oracle events provide the comparison basis. E still needs standalone validation; F's sampler result does not establish that E will improve.

| Date | Status | Evidence and decision |
|---|---|---|
| 19 November 2026 | Completed | Balanced sampling: 56.87% frame accuracy, 23.05% macro chord recall, 4.19% rare macro oracle-event recall from two correct events. |
| 19 November 2026 | Proposed | Repeat both baseline and balanced training with matched additional seeds to test stability before adopting a teacher-training policy. |

For the weekly presentation, state that sampling produced the first correct rare oracle events with a small overall-accuracy loss. Show the two-event count and the five unresolved qualities on the same slide as the 4.19% macro number. This is the current limit of the evidence.


## Proposed richer chord decomposition · 19 November 2026

The user proposed Root, Triad, Bass, Seventh, Ninth, Eleventh and Thirteenth components. This is a proposed representation revision; it has not been implemented or evaluated. The existing baseline remains root/triad/extension plus N, with slash bass collapsed and a 14-quality inventory.

A richer representation could preserve inversions and supervise higher tensions currently mapped to OOV. Specify each head's states and masks before implementation. Distinguish an absent degree from an unknown or unsupported annotation. Preserve diminished seventh (bb7), suspended triads and alterations. Decide how sixth chords are represented without silently equating the sixth and thirteenth roles, or diminished sevenths and major sixths, just because they share pitch classes.

If adopted, audit detailed symbols and training-only component/song support, define reconstruction and excluded-label policies, version the new vocabulary/cache, and evaluate against the current model on the same supported subset as well as expanded coverage. Existing checkpoints provide historical baselines but their heads and targets cannot be reused unchanged for the expanded prediction space. Additional heads do not by themselves establish improved recognition.

Reference: [McGill annotation conventions](https://ddmal.ca/research/The_McGill_Billboard_Project_%28Chord_Analysis_Dataset%29/), which use Harte chord syntax with additional shorthand. The next step for this proposal is a representation and coverage audit, before changing teacher F or teacher E.


## Proposed encoder comparison before richer decomposition · 19 November 2026

The user prefers keeping the richer Root/Triad/Bass/Seventh/Ninth/Eleventh/Thirteenth decomposition as a reserve proposal, subject to a label and coverage audit. A compact bidirectional Transformer is a candidate encoder experiment before expanding the target inventory. The architecture plan already lists Transformer substitution as optional experiment B1.

Research precedents include [BTC, Park et al., ISMIR 2019](https://arxiv.org/abs/1907.02698) and [Harmony Transformer, Chen and Su, ISMIR 2019](https://archives.ismir.net/ismir2019/paper/000030.pdf). BTC uses self-attention for chord recognition; Harmony Transformer jointly models segmentation and recognition. A compact encoder substituted into this project's component-head teacher would be an adaptation, requiring its own specification and comparison.

Proposed comparison: retain 24-dimensional chroma, the 14-quality vocabulary, component heads, split, balanced sampling policy, unweighted losses, decoder and evaluation metrics. Compare TCN and Transformer with explicitly reported parameter counts, input context, optimizer/training budget, runtime and memory. Use the same seeds for both configurations. First change F's encoder alone; E should later be trained independently and evaluated before attributing any complementarity to architecture.

Choose positional encoding, attention context, padding masks and overlap/cropping rules before implementation. Full-recording attention must be resource-tested; bounded inference windows may be needed. Report differences in effective context and capacity so improvements are not attributed solely to the model family.

Status: proposal only. No Transformer has been implemented or trained. A Transformer may improve contextual modeling but is not established as better on this dataset or rare-quality objective. The two-event balanced-TCN gain and limited independent training support remain relevant. Broader decomposition stays deferred while the encoder experiment is specified.


## Matched-seed repeat protocol · 19 November 2026

Status: four new Kaggle jobs completed and independently verified; see the repeated-seed results below. This repeats the existing TCN comparison before changing the encoder or chord decomposition.

Freeze additional seeds 20261120 and 20261121 in [p1-seeds.json](../../configs/p1-seeds.json). Each seed runs both the original baseline configuration and the balanced sampling configuration. Preserve 30 epochs, 367 labeled training songs, 110 validation songs, unweighted losses, model width, window/context, batch size, learning rate, decoder and best-checkpoint selection by validation frame accuracy. Each pair shares the initialization seed; their training sample sequences differ by design.

The notebook trains four jobs sequentially on one GPU session. It reuses the verified original seed 20261119 reports (baseline revision 8ec616c and balanced revision d250916) for the third pair. New pairs run from one common current revision, recorded in plan.json and each run.json. Original results come from earlier commits, so the summary explicitly records that provenance. Additional-seed results should also be inspected separately if the earlier-revision provenance affects a comparison.

Each new job retains its exact generated config, checkpoint, history, per-song logits and reference-event diagnostics. New job names are baseline-20261120, balanced-20261120, baseline-20261121 and balanced-20261121 inside runs/p1-seed-repeat/. Existing completed runs may be reused only with the same config/revision; incomplete jobs are reported explicitly.

### Planned summary and decision

Export per_seed.csv, mean and sample standard deviation (ddof=1) for each method, plus mean/std of paired balanced-minus-baseline differences. Summarize frame accuracy, macro complete-chord recall, oracle-event accuracy and rare macro oracle-event recall. Include positive, negative and tied seed counts for each difference. Keep per-quality event counts and precision available in the constituent run outputs.

Three seeds provide descriptive evidence about initialization/sampling variation on this fixed validation set. They do not quantify uncertainty over new songs or establish statistical significance. CUDA operations are not forced into deterministic mode, so seed equality alone is not a guarantee of bitwise reproducibility.

Before adopting balanced sampling for subsequent teachers, inspect whether rare-event improvements recur across seeds and qualities, whether precision remains usable, and the common-quality/overall tradeoff. Record any inconsistent or negative results. Do not change the checkpoint criterion or choose seeds after seeing their outcomes. No numeric improvement threshold beyond the existing evaluation protocol is introduced here.

The next weekly update should include the three paired rows per method, variability, correct rare-event counts and the decision. Transformer replacement and richer decomposition remain proposed experiments while these repeats are pending.


Local verification for the repeat workflow: all 11 tests passed. Hand-computed checks verified paired means and sample standard deviations, rejection of incomplete/duplicate pairs and nonfinite metrics. A mocked orchestration test verified exactly four new training launches, four diagnostics, inclusion of the original pair, and reuse of completed runs. Notebook cell syntax and local document links were checked. The mocked test does not validate Kaggle execution; the subsequent returned GPU runs are assessed below.


## Matched-seed repeat results · 19 November 2026

All four additional jobs completed at revision 777ec9133cc9408910d5575d43d4f96ebcdf10de. Source hashes match the recorded commit. Training/evaluation IDs, data, model capacity and configuration controls match the original experiment. Each job completed 30 epochs; the selected epochs were baseline-20261120: 27, balanced-20261120: 24, baseline-20261121: 30 and balanced-20261121: 23.

The archive passed integrity checks. Recomputed predictions from all saved logits and original annotations reproduced all four diagnostic summaries and confusion matrices. Each run scored exactly 456,046 validation frames and 9,673 reference-interval events. Mean, sample standard deviation and paired differences were independently recomputed and matched the returned summary.

### Individual seeds

| Seed | Baseline frame accuracy | Balanced frame accuracy | Baseline rare macro event recall | Balanced rare macro event recall |
|---|---:|---:|---:|---:|
| 20261119 | 57.11% | 56.87% | 0% | 4.19% |
| 20261120 | 57.20% | 56.45% | 0% | 0% |
| 20261121 | 56.79% | 57.10% | 0% | 3.57% |

The original seed pair is reused from the earlier verified runs; only the two additional seed pairs were newly trained. The original and repeat revisions are explicitly retained in the [plan and provenance](../../reports/p1/seed-repeat/plan.json).

### Three-seed descriptive summary

Values are mean ± sample standard deviation across the three seeds. Standard deviations are expressed in percentage points.

| Metric | Baseline | Balanced |
|---|---:|---:|
| Complete-chord frame accuracy | 57.03% ± 0.22 | 56.80% ± 0.33 |
| Macro complete-chord recall | 20.15% ± 0.78 | 21.83% ± 1.63 |
| Oracle-event accuracy including N | 58.03% ± 0.50 | 57.71% ± 0.28 |
| Rare macro oracle-event recall | 0% ± 0 | 2.59% ± 2.26 |

Balanced-minus-baseline paired changes averaged -0.23 percentage points for frame accuracy, +1.67 for macro chord recall, -0.32 for oracle-event accuracy and +2.59 for rare macro event recall. Balanced improved rare-event recall on two seeds and tied on one. Frame accuracy improved on one seed and declined on two. These are descriptive results on the same fixed validation songs, without a statistical significance claim.

### Rare-event detail and limitations

The baseline recognized no complete rare reference events in any seed. Balanced seed 20261119 recognized one C:dim interval in song 0343 and one D:minmaj7 interval in song 1253. Balanced seed 20261120 recognized none. Balanced seed 20261121 recognized the same D:minmaj7 interval in song 1253 (108.489–110.295 s). The seed-21 macro score is (1/4) / 7 = 3.57%.

Thus the rare gain appeared in two runs, but only two unique intervals across all balanced runs were ever correctly recognized. The dim success did not recur. The repeated minmaj7 success comes from the same recording and cannot establish generalization across additional independent songs. Five rare qualities remain at zero event recall in every run, and dim remains unstable.

Three seeds reveal sensitivity to initialization and sampling, but do not increase the independent validation-song sample. Per-quality event counts and false positives must accompany the aggregate score. More frequent exposure to the existing rare training samples has provided a limited improvement, rather than reliable rare-chord recognition.

[Per-seed metrics](../../reports/p1/seed-repeat/per_seed.csv) · [Means, standard deviations and paired deltas](../../reports/p1/seed-repeat/summary.json) · [Rare-event counts for every method/seed](../../reports/p1/seed-repeat/rare_events.csv) · [Independent verification](../../reports/p1/seed-repeat/verification.json)

### Decision after repetition

Balanced sampling remains a reasonable candidate setting for further encoder experiments because it occasionally recovers rare events with a small average frame-accuracy cost. It is not adopted as a proven solution for both teachers. Preserve the natural baseline and all balanced runs as comparators.

The matched-seed check is now complete. In line with the user's architecture discussion, the next proposed step is to specify a compact bidirectional Transformer encoder comparison while keeping the current root/triad/extension targets, data split and chosen sampling controls. Record capacity, attention context, runtime and checkpoint-selection policy before training. Do not expand decomposition simultaneously. Teacher E still requires an independent implementation and standalone validation.

### Weekly presentation update

- Completed four new runs and independently checked three matched seed pairs.
- Balanced increased average macro chord recall by 1.67 percentage points and rare macro oracle-event recall from 0% to 2.59%.
- Average frame accuracy decreased by 0.23 percentage points.
- Rare recognition remains concentrated in two unique validation intervals; the same minmaj7 interval recurred in two seeds.
- Next experiment proposal concerns encoder capacity/context through a compact Transformer, with the current label decomposition retained.

Full artifacts are stored locally in runs/p1-seed-repeat/. The original ZIP remains in runs/imports/p1-seed-repeat-results.zip. Selected metrics, histories and tables are versioned under reports/p1/seed-repeat/.


## Compact Transformer specification · 19 November 2026

The initial project-specific Transformer encoder is implemented for teacher F. It follows the broader self-attention direction established by [BTC](https://arxiv.org/abs/1907.02698); the implementation and component heads are this project's adaptation. BTC's published results cannot be compared directly with this chroma/component-head pilot without matching inputs and protocols.

Architecture: project each 24-bin chroma frame to width 48, add sinusoidal positions, pass through two independently initialized bidirectional TransformerEncoderLayer blocks with four attention heads, feed-forward width 96, GELU, pre-layer normalization and zero dropout, then apply final layer normalization and the existing four head dimensions (2,12,6,5). Parameters total 40,441 versus the TCN's 37,225 (about 8.6% more). No pretrained checkpoint is used.

[Implementation](../../src/acr/transformer_f.py) · [PyTorch encoder layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)

### Context, padding and inference

Training retains 512 target frames and 31 context frames on each side, with the same quality/song-balanced sampler and unweighted losses. Attention is unrestricted among the real frames within this 574-frame input, including future frames for offline analysis. Padding uses an explicit boolean key-padding mask. A real zero-chroma silence frame remains an unmasked observed frame. Padded targets stay ignored by the losses.

At validation, run bounded 574-frame windows with stride 512 and retain only each central target segment. This includes unlabeled windows and short final tails so timestamps, labels and logits remain aligned for every frame. Each window uses local sinusoidal positions. Full-song attention is not used.

The TCN sees at most 63 frames of contextual evidence per target, while this Transformer sees up to 574 real frames within the window. The experiment therefore compares encoder designs with different effective context and slightly different capacity, rather than isolating self-attention from every other factor. Transformer padding masks, layer normalization and GELU are also part of the encoder specification. Memory/runtime must be measured from the actual Kaggle execution.

### Training and evaluation controls

Use the existing two-song overfit protocol first (0003/0006, seed 20261119, 200 epochs, batch size 8, learning rate 0.001). Then train the full 367-song balanced run for 30 epochs, batch size 16, seed 20261119 and learning rate 0.001. Keep vocabulary, normalized features, optimizer, loss, gradient clipping, decoder, held-out validation songs and best-checkpoint criterion as in the balanced TCN.

Compare the full run against the balanced TCN seed 20261119 and the descriptive three-seed range. A single Transformer seed is an initial screen; it does not establish a robust architecture advantage. Evaluate full-chord frame scores, rare oracle-event recall, per-quality precision and component scores. Any learning-rate or context revision after viewing results must be logged as a new experiment.

### Local verification and status

Fourteen unit checks passed, including all earlier TCN checks, Transformer padding invariance on real frames, distinction between silence and padding, finite masked-loss gradients, all-padding rejection and complete inference coverage through unlabeled windows and final tails. A local CPU two-song run passed: 99.69% whole-chord training frame accuracy (5,792/5,810), with best epoch 184 of 200. Saved predictions independently reproduced this metric. Full Kaggle execution has since completed; its results are recorded below. The larger attention context and slightly higher capacity mean the overfit score alone cannot establish an architectural advantage. [Local run record](../../reports/p1/transformer-setup/local-overfit.json)

The TCN path retains full-recording convolutional inference and its default configuration; Transformer windows are used only when encoder=transformer. New checkpoints record encoder and window metadata together with the state dictionary. The existing target decomposition is retained.


## Compact Transformer Kaggle results · 19 November 2026

Both returned runs used revision 8bb198f4436e3bdf39f3973cb4240eaea1accfc5, PyTorch 2.11.0+cu130 and NumPy 2.1.3 on CUDA. The overfit check reproduced 99.69% training frame accuracy (5,792/5,810) at epoch 184. The full balanced run completed 30 epochs and selected epoch 29 by validation frame accuracy.

Verification passed ZIP integrity, committed source hashes, training/evaluation IDs, vocabulary, input hashes and the balanced-sampling controls. The full configuration matches the balanced TCN except encoder=transformer. All 110 validation prediction arrays preserve frame count and timestamp ordering. Recomputing their confusion matrix reproduced the saved one, and the independent reference-event diagnostic reproduced every submitted summary value.

### Comparison at seed 20261119

| Metric | TCN natural | TCN balanced | Transformer balanced |
|---|---:|---:|---:|
| Complete-chord frame accuracy | 57.11% | 56.87% | 55.38% |
| Macro complete-chord recall | 19.38% | 23.05% | 20.85% |
| Oracle-event accuracy including N | 58.57% | 57.92% | 55.63% |
| Rare macro oracle-event recall | 0% | 4.19% | 0% |

The Transformer correctly classified 252,553 of the same 456,046 scored validation frames, versus 259,365 for the balanced TCN. Relative to that matched-seed comparator, frame accuracy decreased 1.49 percentage points, macro chord recall 2.21 points and oracle-event accuracy 2.30 points. All seven rare qualities have zero correct complete-chord frames and zero correct reference events.

The per-quality frame recalls are maj 72.45%, min 44.82%, 7 28.30%, min7 27.10% and maj7 10.95%. Some individual qualities improve slightly relative to the balanced TCN, but aggregate and rare-objective outcomes do not improve.

The standalone root, triad and extension frame accuracies are 79.25%, 84.12% and 76.34%. For minmaj7 oracle events specifically, root and triad each reach 100% standalone accuracy, but the seventh/extension head has 0% accuracy, so no complete event is correct. This reinforces the importance of inspecting defining components rather than root alone.

[Run metrics and histories](../../reports/p1/kaggle-transformer/balanced/metrics.json) · [Matched-seed comparison](../../reports/p1/kaggle-transformer/comparison.csv) · [Component and quality diagnostics](../../reports/p1/transformer-diagnostics/frame_qualities.csv) · [Reference-event support and scores](../../reports/p1/transformer-diagnostics/event_qualities.csv) · [Verification](../../reports/p1/kaggle-transformer/verification.json)

### Resource observations

The Transformer has 40,441 parameters compared with the TCN's 37,225. Peak PyTorch CUDA allocated memory was 77,817,856 bytes versus 26,010,112 bytes for the balanced TCN, about three times as much. Its epoch loops totaled 203.83 seconds versus 67.41 seconds for the balanced TCN run. These are observed run measurements; they exclude preparation and do not control for GPU device differences, utilization or separate-session overhead.

Despite a broader attention window and slightly larger capacity, the trial did not improve the target metrics. The strong two-song overfit score establishes pipeline learnability, not validation generalization.

### Decision and remaining questions

Retain this Transformer run as a completed negative encoder trial. Keep the TCN runs as the current references; do not replace the teacher backbone based on this result. This is one compact configuration at one seed, not evidence that all Transformers are worse than TCNs. The unchanged learning rate and epoch budget constitute a first screening protocol, not architecture-specific optimal tuning.

Broader context, positional encoding, optimization, feature information and rare-label support remain possible explanations requiring separate evidence. The result does not identify one of them as the cause. Expanding decomposition has not yet been tested and cannot be presented as a confirmed fix.

A useful next project step is to specify P2's oracle-event E with the compact TCN and independent weights, then compare F and E on identical reference intervals. Event-level supervision is the planned distinction between teachers; its benefit remains unmeasured. In parallel, any further encoder tuning should have an explicit hypothesis and logged controls rather than replacing architectures repeatedly without a diagnosis.

| Date | Status | Evidence and decision |
|---|---|---|
| 19 November 2026 | Completed | Compact Transformer balanced trial: 55.38% validation frame accuracy, 55.63% oracle-event accuracy and 0% rare macro event recall. |
| 19 November 2026 | Recorded decision | Preserve the negative result and retain TCN references. Further Transformer tuning and P2 event-teacher implementation require their own specified experiments. |

### Weekly presentation notes

- Tried a compact self-attention encoder with the same components and balanced sampler.
- The training pipeline passed and overfit two songs at 99.69%.
- Validation performance fell relative to the balanced TCN, and rare-event recognition returned to zero.
- More context and a different architecture alone did not resolve the current rare-component failures.
- Preserve the original research question and test the planned frame-versus-event teacher distinction.

Raw outputs are in runs/p1-transformer/overfit/ and runs/p1-transformer/balanced/. The downloaded ZIP remains in runs/imports/p1-transformer-results.zip. Versioned selected artifacts are in reports/p1/kaggle-transformer/ and reports/p1/transformer-diagnostics/.


## Handoff to P2 · 19 November 2026

The event teacher implementation is now documented in [P2_REPORT.md](../p2/P2_REPORT.md). The fixed comparator is the original balanced F at seed 20261119, with its uncertainty retained in the repeated-seed analysis above. P2 supplies the same reference intervals to both teachers and fits E independently using event targets. The first P2 oracle trial has now been verified. Its measured F/E complementarity and rare-quality limitations are recorded in [P2_REPORT.md](../p2/P2_REPORT.md).

### Confidence audit follow-up · 20 November 2026

The original F checkpoint has now been audited with E25 at the same seed using saved logits on gold validation intervals. See the component confidence section in [P2_REPORT.md](../p2/P2_REPORT.md) for definitions, per-head precision/coverage and per-class results. This does not change F's training or frame-level metrics. At an exploratory threshold of 0.9, F/E agreement attains 96.54% overall triad precision but gets all 24 accepted triads on true rare-quality events wrong. Shared confident errors remain a material limitation; agreement alone is not a reliable rare-label filter. Investigate data diversity next while preserving the Billboard-only control; no augmentation, extra dataset or architecture change has been applied in this audit.


### External-data experiment for F and E — 20 November 2026

Extend the JAAH pilot to F as well as E. Train a fresh Billboard-only F control and a JAAH-mixed F with a separate 12-dimensional input projection and shared temporal trunk/heads. Keep F's 50% quality/song-balanced window sampler; replace 54 of 217 batches per full epoch with JAAH, holding the update/window budget fixed. This is a new experiment, not a replacement for the selected historical F. Compare both new F models with both new E models on the same gold validation events, including per-component confidence, precision, coverage and class-specific errors. See [the experiment protocol](../p2/P2_JAAH_EXPERIMENT.md). Full GPU results are pending.


### JAAH F pilot result — 20 November 2026

The full matched-seed pilot has been imported and independently rescored. F event accuracy falls from 58.348% to 57.118% with JAAH batch replacement; rare macro recall increases from 0.621% to 1.242%, representing only one additional correct dim event. Both runs complete 30 epochs at 217 updates/epoch; the mixed arm replaces 54 batches per epoch. Retain the selected F. The four-pair analysis still finds no correct accepted triads on true rare events at confidence 0.9. See [P2 results and limitations](../p2/P2_REPORT.md) for per-head precision/coverage, E's different tradeoff and the next diagnostic plan. This single-seed replacement-budget result does not establish that external data is generally ineffective.
