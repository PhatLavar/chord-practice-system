# Stage B: Literature baselines and corpus generalization

Established: 21 November 2026, following the user's request to separate the new direction from P2. Status: B0 planning/source review; no new baseline training launched.

## Research scope

Develop credible frame-based and event-based audio baselines, establish their behavior within each admitted corpus, then test cross-corpus transfer and multi-corpus training. Overall recognition quality is primary; rare-quality recall, precision, component reliability and segmentation remain necessary diagnostics.

This is a reset of research organization and experiment contracts, not deletion of data, checkpoints, failed experiments or provenance. Preserve P0/P1/P2 as historical exploratory evidence and reusable infrastructure. Their results do not certify production-ready teachers or a final benchmark. Do not append subsequent baseline/corpus experiments to P2.

Use stage IDs B0–B3 rather than renaming the existing P3, which originally denotes predicted boundaries. Event-sequence baselines touch that subject, but do not automatically complete the original P3 or student milestones. A later architecture-plan revision must reconcile those deliverables explicitly.

## Milestones and exit conditions

| Milestone | Question | Required evidence before advancing |
|---|---|---|
| B0: Protocol and compatibility | What exactly are the inputs, targets, splits and model interfaces? | Pinned source revisions, source/label admission audit, vocabulary/time contract, forward/backward and known-answer I/O checks |
| B1: Single-corpus baselines | Can each baseline learn and generalize within each admitted source? | Source fit, full-timeline recognition, event segmentation where applicable, per-quality support, reproducible screening and matched repeats for promising setups |
| B2: Transfer and multi-corpus | Does adding another source help, and on which evaluation domain? | Fixed per-corpus evaluation sets, overlap checks, source ratios, matched budgets, per-source and pooled results, negative-transfer analysis |
| B3: Consolidation | Which teacher policies and representations deserve further system integration? | Consolidated ablations, justified F/E policies, confidence/coverage evaluation under the chosen boundary protocol and documented remaining limitations |

Baseline choices carried forward from [the source review](FRAME_EVENT_BASELINE_REVIEW.md): BTC large-vocabulary as the initial frame reproduction anchor; TE-DS as the initial genuine event-sequence baseline, then pTE-DS. ChordFormer is a modern structural frame follow-up. These are candidates, not demonstrated improvements on our data. Keep current TCN controls, including F-natural/E-transposition GuitarSet results, as controls rather than new-baseline checkpoints.

## Corpus admission and current feasibility

The table records local project status, not fresh claims that other sources are unavailable publicly. A dataset is admitted to a training condition only after matching input/annotation semantics and recording identities have been checked.

| Source | Currently usable project inputs | Stage B role / dependency |
|---|---|---|
| GuitarSet | Pinned microphone audio and performed labels; known errata exclusions | Immediate audio compatibility and single-corpus development source; limited repertoire and historical exposure |
| McGill Billboard | Official detailed labels and 24-dimensional Chordino features | Existing feature-based control; matching audio is required before treating it as a common-CQT audio source |
| JAAH | Audited labels and 12-dimensional NNLS features, with exclusions | Existing jazz feature-based diagnostic; audio-to-label version matching needed for the common audio baseline |
| Isophonics / USPOP / Robbie Williams | Literature and annotation leads; user has no matching audio | Conditional pop/rock expansion; no usable audio tracks assumed |
| Other sources, including synthetic candidates | Prior literature leads only | Audit separately before admission; synthetic-to-real transfer must be an explicit condition |

Do not manufacture 144-bin CQT by duplicating or reshaping 12/24-dimensional chroma. Multiple front ends or source-specific adapters are legitimate experiments, but add architectural changes and must be distinguished from a common-front-end corpus comparison. F and E need not ultimately use the same training mixture; first compare each architecture on a declared common source so architecture and data effects can be separated.

There is currently only one admitted raw-audio corpus in this project. Therefore B2 common-CQT multi-corpus training is conditional on admitting at least a second compatible audio corpus. Admission can proceed alongside B1; it need not wait until all single-corpus experiments are exhausted. A chroma-only multi-source study can proceed under its own input contract, not as an author audio-baseline reproduction.

## Required experiment matrix

For each baseline, define A and B as independently audited sources. Use separate, fixed train/development/evaluation manifests per source, with matching task vocabulary and explicitly reported excluded labels.

| Training source | Evaluate on A | Evaluate on B | Interpretation |
|---|---|---|---|
| A only | Within-source generalization | Zero-shot cross-source transfer | Source A baseline and domain sensitivity |
| B only | Zero-shot cross-source transfer | Within-source generalization | Source B baseline and domain sensitivity |
| A + B | Evaluate held-out A | Evaluate held-out B | Added-source benefit or harm on each domain |

Zero-shot means no training/checkpoint/threshold tuning on the named evaluation source. Any tuning there creates an adapted condition and must be relabeled. Train A+B never includes the held-out A/B recordings. Freeze mixture selection rules using development data and do not optimize one pooled score using final evaluation sets.

Run the matrix independently for frame and event systems, keeping oracle-event classification separate from free-running event prediction. Only after those controls, consider different optimal corpus mixtures for F and E and measure complementarity on the same evaluation recordings. Different training corpora may improve individual teachers but also make architecture comparisons ambiguous.

## Controls that make the comparisons interpretable

- Group duplicate recordings, editions and related composition/lead-sheet versions according to the stated generalization question before assigning partitions, including across corpora. Keep every augmented derivative with its original recording. Fit normalization and sampling weights on training only.
- Existing GuitarSet player validation has been used repeatedly, and old progression holdouts have entered later training. A new phase name, seed, split or five-fold rotation cannot make that historical pool pristine again. Retain it as development evidence; reserve a genuinely new evaluation source/recordings for later untouched claims when available. Preserve historical exposure records instead of silently replacing them.
- Report both fixed-update source replacement and added-data/added-compute conditions when relevant. More songs with more optimization steps does not isolate a corpus-diversity effect. Prespecify natural/proportional or source-balanced sampling and record per-source exposure; tune mixture on development data.
- Produce per-source duration-weighted Root/MajMin/Triads/Tetrads and common-label event diagnostics, support and OOV/eligible-duration counts. Show both pooled-duration results and equal-corpus summaries; neither alone describes all domains. Do not hide a small corpus regression behind a large corpus's weight.
- For event systems, evaluate free-running boundaries and segmentation as well as chord recognition. Oracle-boundary and teacher-forced results are separate diagnostics. Different output decompositions must be reconciled by explicit symbol-level mapping, not equal integer IDs.
- Separate a common-input architecture ablation from an author-recipe adaptation. Record parameters, updates, runtime, memory, feature recipe and pretrained-data overlap. More capacity or a modern architecture is not guaranteed to improve a small dataset.
- Establish a primary metric/selection criterion before each experiment, retain negative results and use matched seeds for promising comparisons. Seeds measure optimization variation; grouped folds measure a different source of variation and create no new training recordings.

## Organization and next action

| Location | Purpose |
|---|---|
| docs/baselines/ | Stage B human-readable plans, instructions and research log |
| reports/baselines/ | Machine-readable source audits and future experiment evidence |
| Existing src/acr/ | Shared implementations; no bulk code relocation required |
| Existing data/, runs/, splits/ | Preserve current versioned sources and run history; new manifests/outputs get distinct stage names |
| Future B notebooks/configs | Use a B-specific prefix rather than extending 02/P2 numbering |

Do not create empty placeholder trees or rename historical run folders merely for cosmetic consistency. Future B artifacts should carry stage, model family, corpus condition, split version and seed in metadata. New notebooks can use b01_, b02_, etc.; names will follow actual implemented experiments.

Immediate B0 work: pin and check BTC/TE-DS author code; specify common symbol/time interfaces and known-answer tests; audit matching-audio availability in parallel. Start B1 with the admitted GuitarSet source once compatibility checks pass. Plan B2 explicitly now, but launch it only when two corpus input contracts and clean grouping rules exist. No training or data deletion is performed in this documentation reset.
