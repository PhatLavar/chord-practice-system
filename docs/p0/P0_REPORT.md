# Billboard P0 corpus audit

The frozen pilot contains 734 song groups from 883 eligible chart slots. Use the [manifest](../../splits/billboard-v1.csv) and filter to rows where `id == representative` when constructing examples. Read identifiers as four-character strings. P0 ran locally on CPU.

## Inputs and reproduction

Follow the [P0 instructions](P0_INSTRUCTION.md) for official downloads, local filenames, commands and expected outputs.

McGill releases the data under CC0. Cite Burgoyne, Wild and Fujinaga (ISMIR 2011). McGill describes phrase-internal timings as interpolated. Frame coverage checks establish timestamp compatibility; musical alignment to the original audio was not measured. [Publisher and citation](https://ddmal.ca/research/The_McGill_Billboard_Project_%28Chord_Analysis_Dataset%29/)

## Eligibility and partitions

Of 1,300 indexed slots, 410 lacked both inputs and 890 had pairs. We required finite 24-bin frames, strictly increasing nonnegative timestamps, valid ordered intervals, at least 99% annotation coverage, and an absolute difference of at most one second between the last frame and annotation end. The last-frame difference is signed in the [inventory](../../reports/p0/inventory.csv). Six slots failed timing checks (0218, 0308, 0669, 0723, 0794, 0823). Slot 0974 contains four reversed and four empty intervals. It was excluded as a whole. For eligible slots, 13 of 4,114,582 frames are outside annotation; mask them during training. Empty annotation and the observed no-chord class N have different meanings.

Group slots transitively by case-folded artist/title or identical SHA-256 of the feature file. Use the smallest eligible ID as the representative. This controls the duplicates identifiable by these keys, not every possible cover or alias. Sort group IDs, shuffle with Python seed 20261117, then partition at the 50%, 70%, and 85% cutoffs (rounded down). These are pilot choices. Preserve this manifest across experiments.

| Partition | Independent song groups | Permitted use |
|---|---:|---|
| train_labeled | 367 | Fit the supervised model and derive vocabulary statistics |
| train_hidden | 146 | Reserve for pseudo-labeling; withhold reference labels from model training |
| validation | 110 | Model selection and calibration |
| test | 111 | Final evaluation after the protocol is frozen |

Artist credits may recur across splits. For instance, 40 distinct artist-credit strings occur in both labeled train and test. [All pairwise overlaps](../../reports/p0/summary.json)

## Vocabulary and event counts

Use 12 pitch classes and the 14 qualities in [qualities.csv](../../reports/p0/qualities.csv), plus N. Collapse slash bass and enharmonic roots. Preserve distinct sixth, minor seventh, major seventh and diminished seventh component targets. Accept exact quality shorthand. Altered, extended or explicitly spelled variants outside that inventory are conservatively marked OOV, including some musically equivalent spellings. X means unknown. Mask X, OOV and uncovered frames in supervised loss and primary scoring; report excluded coverage. N is a supervised whole-label class with masked root, triad and extension. For supported chords all three components are observed. The [symbol map](../../reports/p0/symbols.csv) lists every raw symbol seen on labeled training representatives, including excluded ones.

Merge touching intervals when their mapped root and quality match; keep distinct OOV symbols separate. Count events, seconds, distinct songs and distinct roots **only on labeled training representatives**, before augmentation. Of 36,084 events in 22.01 annotated hours, 31,679 events (87.79%) and 87.77% of duration are retained including N.

For this pilot, define a rare quality as observed in **less than 1% of supported chord events** on labeled train (exclude N, X and OOV from this denominator). This yields aug, dim, dim7, hdim7, min6, minmaj7 and sus2. In particular dim7 occurs in only three independent training songs. Freeze this rule for the pilot and report per-quality support when evaluating. The five-quality subset in the architecture plan remains available for a small pipeline check.
