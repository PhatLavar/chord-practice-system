# Accuracy-first multi-corpus research plan

Date: 20 November 2026. Status: literature/code review and prespecified plan; no new corpus training or five-fold scores yet.

## Objective and interpretation

Following the user's clarification, prioritize overall recognition accuracy and credible comparison with published systems for both F and E. Track rare quality recall and component precision/coverage as secondary diagnostics. Diversity is valuable, but more source domains can cause negative transfer; disagreement alone is not useful unless one teacher is correct. Retain per-source scores, common-label metrics and F-only/E-only correct counts.

The current models are compact chroma TCN research baselines. E receives gold event boundaries, and F/E event accuracy weights each event equally. These scores cannot be compared numerically with published audio-to-timeline WCSR. A small Transformer trial in this project does not establish that published BTC or Transformer architectures are inferior.

## Evidence from the matrix and author code

Re-read the ACR entries in the user's [SYNTHESIS MATRIX](https://docs.google.com/spreadsheets/d/1Y8wUc8XaKuiAit_q-Je09pxsIWVyVc9S/edit). BTC and the event-based paper use the 471-song setting in the matrix; other entries use different pools and protocols. Matrix summaries guide reproduction, but paper claims require checking their original evaluation protocol.

The [BTC author repository](https://github.com/jayg996/BTC-ISMIR19) supplies CQT preprocessing, BTC, CNN/CRNN baselines and evaluation code. Its [dataset implementation](https://github.com/jayg996/BTC-ISMIR19/blob/master/audio_dataset.py) divides sorted song identities into five folds; [training code](https://github.com/jayg996/BTC-ISMIR19/blob/master/train.py) combines separate Isophonics/USPOP/Robbie datasets and computes normalization from training batches. Reuse and adapt these documented components rather than inventing a new benchmark architecture. Audit legacy dependencies and split construction before execution. Author code is a reproduction starting point, not an automatically leakage-safe split for an expanded corpus.

Five-fold evaluation rotates held-out songs; it does not create independent training data. For our extension, group duplicate recordings, covers/compositions and all augmentations together before assigning folds. Use inner development data for checkpoint/hyperparameter selection and outer folds for evaluation. Report published-protocol reproduction separately if its selection procedure differs. Preserve the existing frozen Billboard holdout; do not absorb it into new training folds. Fit normalization only on each fold's training partition.

## Dataset order

| Source | Purpose for F and E | Admission work |
|---|---|---|
| Isophonics, then USPOP/Robbie | Closest connection to BTC's pop/rock benchmark; primary accuracy expansion | Obtain exact matching audio, inventory labels, align versions, deduplicate across Billboard and source collections |
| GuitarSet 1.1.0 | Accessible audio-to-feature experiment and guitar application domain | Audit performed/instructed labels and timing errata, group lead sheets and player identities according to the evaluation question |
| JAAH | Retain as tagged jazz source and source-fit diagnostic | Finish upstream semantics check and source optimization analysis; avoid assuming that its current mixture is beneficial |
| RWC-Pop | Further pop diversity | Resolve audio access, exact recording/annotation pairing and overlap |
| AAM | Later synthetic augmentation ablation | Start from the published baseline/recipe, check vocabulary and evaluate transfer on real held-out recordings |

[Isophonics](https://isophonics.net/content/reference-annotations.html) gives annotation confidence and specific CD editions. Annotation access alone is not training audio access. [GuitarSet's pinned release](https://zenodo.org/records/3371780) supplies annotations and microphone audio plus known timing-error links; use the microphone subset rather than all multichannel stems for the first experiment. GuitarSet is not a substitute for reproducing the commercial-music benchmark. No corpus is counted as added until matched usable recordings have been audited.

## Feature and baseline protocol

Prefer a common audio front end across new corpora when audio is available. Build a CQT baseline using the BTC author's preprocessing and architecture as a separate experiment. Preserve the existing chroma TCN as a measured control. Keep feature choice and architecture changes explicit; a CQT-versus-chroma difference cannot be attributed solely to the Transformer. Where only compatible documented features exist, use a separate input branch and report that experimental condition.

Use duration-weighted chord evaluation with explicit Root, MajMin, Triads and Tetrads reductions, plus the project's complete-label metric and vocabulary coverage. Define unknown/N handling before training; unsupported labels must not silently become N. Report per-song/per-corpus scores as well as aggregate scores. For E, label gold-boundary results as oracle classification; comparisons to end-to-end event models require predicted boundaries and segmentation metrics. Match datasets, splits, vocabulary, input access and metrics before claiming parity with prior papers. Published headline scores on other datasets are context only.

## Next executable stages

1. Verify JAAH extraction semantics and run a quality-diverse small source-fit diagnostic for both teachers. Track all heads; record memorization as training fit only.
2. Inventory matching audio availability for pop corpora; audit GuitarSet annotations as the immediately downloadable audio candidate. Pin files/hashes, map labels, quantify usable durations and leakage groups before feature extraction.
3. Add standard duration-weighted evaluation and reproduce one author baseline on the exact admitted split. Preserve a separate oracle E track.
4. Compare single-source and added-source training for F/E separately, with matched seeds. Distinguish replacement-budget experiments from added-data/added-compute experiments. Select source proportions on development data.
5. Once preprocessing and a baseline work, run grouped five-fold evaluation on the newly admitted benchmark pool, with explicit outer/inner roles. Report uncertainty and source-specific results. Revisit rare-quality optimization after establishing the overall-accuracy baseline.

No new GPU run is needed until the input and evaluation contracts are ready. The unresolved matching-audio availability is a concrete dependency, not a reason to assume a dataset is unusable.
