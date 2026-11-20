# Dataset review for F/E development

Review date: 20 November 2026. Status: source-level feasibility review, not a completed corpus import or a training result.

Follow-up: JAAH has since been selected and audited at file level for a two-input E pilot. The later results and setup are in [P2_JAAH_EXPERIMENT.md](P2_JAAH_EXPERIMENT.md). The source-level limits below describe the original review stage; they do not supersede that subsequent inventory. JSIR/GuitarSet remain later candidates.

## Recommendation

Investigate **JSIR and JAAH for rare-chord supervision**, **GuitarSet for an accessible guitar-domain pilot**, and **Isophonics for a pop/rock expansion when matching audio is available**. Keep RWC-Pop and USPOP as conditional candidates. Consider SWD for a small classical transfer experiment and AAM for a separate synthetic-data ablation. None is yet certified as a complete, compatible addition to the current training set.

This ordering follows the confidence audit: confident F/E agreement supplies only major/minor triads and almost exclusively `none` extensions. Additional songs are useful only if their usable labels and independent recordings address this deficit. Jazz is a plausible source of seventh/diminished harmony, not a guarantee of better Billboard accuracy.

## Connection to the user's literature matrix

Read the ACR portion of [SYNTHESIS MATRIX.xlsx](https://docs.google.com/spreadsheets/d/1Y8wUc8XaKuiAit_q-Je09pxsIWVyVc9S/edit) through the Google Drive connector. The file is an Excel workbook displayed through a Sheets URL; it was read only.

The six ACR entries cover BTC, pseudo-labeling/KD, event-based non-triad recognition, ChordFormer, Harmony Transformer and Lardet's project report. Their dataset notes identify Isophonics, Robbie Williams, USPOP, Billboard, RWC-Pop, an aggregated Humphrey–Bello collection, unlabeled audio pools and synthetic music. These are literature leads, not proof that every referenced audio collection is downloadable or every count denotes new songs. Individual paper statistics were not all independently reproduced in this review.

BTC's [author repository](https://github.com/jayg996/BTC-ISMIR19) confirms its use of Isophonics, Robbie Williams and USPOP annotations with separately obtained audio. Its 471-song experimental subset in the matrix must not be confused with a complete downloadable audio release. Similarly, an aggregate collection containing Billboard does not supply that many additional independent recordings to our existing corpus.

## Candidate comparison

| Candidate | Confirmed annotation/access properties | Project role and main obstacle |
|---|---|---|
| JSIR (2026) | 267 recordings of 44 standards; time-aligned chord, beat and structure labels; audio identifiers instead of distributed audio | Strong rare-harmony candidate; obtain exact recordings, respect valid segments, group versions by composition |
| JAAH (2018) | 113 jazz tracks; musician-created aligned harmony annotations; JSON/LAB and extracted chroma | Established jazz candidate; verify detailed label mapping and feature compatibility/audio access |
| GuitarSet (2018; release 1.1.0) | 360 short acoustic-guitar excerpts; audio and JAMS labels available | Practical guitar pilot; performed labels are derived, limited progression diversity, timing errata |
| Isophonics | Harte start/end/label annotations; reference CD editions listed | Pop/rock expansion; obtain matching audio and measure rare support |
| Robbie Williams | Used in classic ACR studies; annotations available through research collections | Conditional pop expansion; resolve release/subset counts and audio correspondence |
| USPOP chord subset | Provider lists 195 manually annotated songs; no audio in annotation repository | Additional pop labels, not the entire USPOP audio collection |
| RWC-Pop | Chord annotation provider covers 100 songs; audio access handled separately | Useful controlled repertoire; audio acquisition and annotation pairing required |
| Schubert Winterreise (SWD 2.1) | 24 compositions; annotations for nine performances; only two audio performances distributed | Small classical transfer study; multiple performances are not independent compositions |
| Artificial Audio Multitracks (AAM 1.1.0) | 3,000 generated tracks; mixes, stems, MIDI and chord annotations | Synthetic-data ablation; audit vocabulary and synthetic-to-real transfer |
| ChoCo | Standardized aggregation of audio-aligned and score-based harmony datasets | Useful discovery/conversion resource; not an independent pool of new audio |

## Priority candidate details

### JSIR: promising new rare-harmony source

The [October 2026 paper](https://arxiv.org/abs/2610.15356) describes 25.6 hours with chord labels revised against performances by a jazz musician. Audio is not bundled; identifiers locate the annotated versions. Labels represent jazz chord symbols rather than literal sounding pitch sets, which matters when instruments omit chord tones.

The [official repository](https://github.com/MALerLab/JSIR) reports 3,249 half-diminished, 556 diminished-family and 207 augmented events. These are **publisher family counts before our filtering**, not locally verified training counts; diminished-family combines dim/dim7. The repository requires sampling only valid segments. Chords are onset CSVs, so interval endpoints must respect those masks. Annotations use CC BY-SA 4.0.

**Decision:** prioritize a label/valid-segment audit. Check exact quality counts after our mapping, independent-song support, and overlap with JAAH/Billboard. Group all versions of one standard together in any new evaluation split. Do not treat 267 performances as 267 independent compositions or infer sus2 coverage from the family table. No local file-level inventory has yet been performed.

### JAAH: established jazz annotations, incompatible feature assumptions

The [dataset paper](https://zenodo.org/records/1291834) describes annotations for 113 Smithsonian jazz recordings, created by a jazz musician, with meter, structure and chords across tracks. The [repository](https://github.com/MTG/JAAH) publishes JSON annotations, LAB exports and chroma features.

The [feature documentation](https://mtg.github.io/JAAH/) describes 12-dimensional NNLS chroma and beat-predominant/root-relative processing for statistical visualization. Our input is 24-dimensional bass plus treble chroma. Inspect the distributed feature files and their generation code before using them; a documented visualization representation is not proof that every downloadable array has identical semantics. Do not duplicate a 12-vector to manufacture a 24-vector or feed root-relative features built with reference labels into recognition training.

**Decision:** prioritize annotation inspection alongside JSIR. Matching audio and re-extraction may be necessary. Check licensing of each artifact; this review does not establish permission to redistribute audio. Jazz seventh-chord prevalence is useful motivation, but coverage of all seven project rare qualities remains unmeasured.

### GuitarSet: accessible audio for the practice-system domain

The [project page](https://guitarset.weebly.com/) provides 360 roughly 30-second excerpts from six players and 30 lead sheets, in comping/soloing versions. It distinguishes instructed harmony from performed labels inferred from note annotations using lead-sheet roots and segmentation. The latter is not fully independent manual chord ground truth.

The [1.1.0 release](https://zenodo.org/records/3371780) offers annotation.zip (39.1 MB) and microphone audio (656.9 MB), so the full multichannel audio download is unnecessary for an initial mic-audio pilot. Check the release's artifact license and include attribution when importing.

Published [timing errata](https://github.com/marl/GuitarSet/issues/5) identify 04_BN3-154-E_comp (+0.409 s) and 04_Jazz1-200-B_comp (+0.309 s). Confirm whether the selected release contains these issues; document exclusion/correction explicitly rather than silently shifting labels.

**Decision:** best candidate here for testing a complete audio-to-feature pipeline without acquiring commercial recordings. Audit performed labels, chord vocabulary and progression/player grouping first. It is a domain-transfer pilot, not evidence that rare qualities are adequately covered.

## Pop/rock and other candidates

**Isophonics:** [official documentation](https://isophonics.net/content/reference-annotations.html) specifies Harte LAB intervals and the CDs used. Beatles chord annotations have undergone repeated checks; annotation confidence differs across collections. Use a pinned archive and actual file inventory instead of assuming the paper's chosen subset equals the complete release. Audio version mismatches can shift every label. This is the most natural genre expansion, conditional on access to matching recordings.

**USPOP and RWC-Pop:** the [annotation provider](https://github.com/tmc323/Chord-Annotations) lists 195 USPOP tracks and all 100 RWC-Pop tracks, manually annotated and cross-checked, in LAB/SVL formats. It explicitly distributes no audio. The 185 USPOP tracks in the matrix's BTC entry are an experimental subset, not a contradiction of the provider's 195. Consult the [RWC access procedure](https://staff.aist.go.jp/m.goto/RWC-MDB/) separately. Song lists and fingerprints must be compared against protected Billboard partitions before adding any training item.

**Robbie Williams:** [Korzeniowski and Widmer's reproduction page](https://fdlm.github.io/post/auditorymodel/) confirms use of this collection. Counts vary between experimental subsets and transformed collections; do not silently equate the matrix's 65 with ChoCo's 61 annotations. Pin the exact release, original labels and matching recordings before counting its contribution.

**SWD:** the [official 2.1 release](https://zenodo.org/records/10839767) provides audio-aligned chord annotations and two performances of the 24-song cycle, despite annotations for nine performances. Score-only Winterreise corpora are different resources. Use detailed chord annotations, not the major/minor reduction. Split by composition, not performance. Classical voice/piano transfer and actual rare-class support require measurement. Read recording-specific redistribution conditions, including the Scarlata/Denk restriction on modified audio, before planning an audio augmentation release.

**AAM:** the [official release](https://zenodo.org/records/5794629) contains generated audio/MIDI plus chord labels and provides a small demonstration version. The full release is approximately 209.6 GB including stems; a small mix-only pilot is sufficient for feasibility. The [2025 ACR study](https://arxiv.org/html/2508.05878v1) evaluates AAM with BTC/HT, but uses a 25-class major/minor/N vocabulary. Its positive findings do not establish improved recognition of our 14 qualities. Audit the original annotation vocabulary before selecting it to address rare triads/extensions; keep synthetic examples tagged separately from human-labeled recordings.

**ChoCo:** the [authors' repository](https://github.com/smashub/choco) distinguishes audio annotations in seconds from symbolic annotations in score coordinates. It reuses several candidates above, including Billboard. Its headline size is not that many new aligned audio recordings. Use it for provenance-aware label normalization/discovery; deduplicate against original corpora and preserve source identities.

**Unlabeled pools from the matrix:** FMA, MAESTRO and DALI are discussed there as pseudo-label-stage sources. Their presence in an ACR experiment does not make them verified chord-ground-truth datasets. They belong to a later weak-supervision/pretraining study, after addressing the current teacher errors. Likewise, a generation prompt or MIDI note track alone does not prove correct performed chord labels.

## Admission criteria before any training

1. Pin source version, URL, checksums and artifact-specific licenses. Inventory annotation/audio/features separately; publish actual matched counts, exclusions and missing files.
2. Check finite, ordered, positive intervals; gaps/overlaps; duration coverage; valid-segment masks; exact audio edition and alignment. Keep unknown/OOV regions masked instead of converting them to N.
3. Count exact symbols and mapped root/triad/extension targets by events, duration, recordings and independent compositions. Report all 14 qualities and zero-support classes. Current exact-shorthand mapping excludes altered/9th/11th/13th forms. Do not collapse them silently merely to increase usable counts.
4. Standardize the audio front end to our documented bass/treble NNLS representation, checking bin order, hop/timestamps, tuning, normalization and frame alignment. A different chroma extractor is an additional experimental change. No feature extraction has been validated in this review.
5. Detect duplicate recordings and versions across corpora; group compositions conservatively. Preserve Billboard validation/test and train-hidden roles. Do not tune selection by inspecting hidden/test chord outcomes.
6. Compare Billboard-only against one added corpus at a time using matched seeds and a controlled update budget. Report source-specific scores, per-class precision/recall and partial-target coverage; a higher overall score is insufficient.

Recommended next execution: small annotation/provenance inventories for JSIR, JAAH and GuitarSet, then choose the first training addition using usable support and audio feasibility. Pitch transposition may diversify roots but preserves chord quality, so it cannot by itself create missing dim/aug/sus2/extension examples.

## Review limits

This review checked the user's matrix and primary project/paper/release pages. It did not download or validate complete candidate corpora, listen to their alignments, verify every audio link, or measure mapped support. A local in-memory JAAH download attempt failed at DNS resolution; it produced no corpus or statistics. The GuitarSet ZIP exceeds the web reader's file-size limit. These are limits of this review, not evidence that the datasets themselves are unavailable. Training data, splits and models remain unchanged.
