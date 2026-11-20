# GuitarSet inventory and admission status

Pinned source: [GuitarSet 1.1.0](https://zenodo.org/records/3371780), CC BY 4.0 according to release metadata. Cite Xi, Bittner, Pauwels, Ye and Bello (ISMIR 2018). Downloaded annotation.zip (39,132,574 bytes) and audio_mono-mic.zip (656,927,981 bytes); both match the publisher MD5 checksums. SHA-256 values are in reports/p2/guitarset-inventory/summary.json. Archives remain under ignored data/raw/guitarset/; no raw audio is committed.

## Findings

All 360 JAMS files pair with microphone WAVs. WAV headers, durations and label bounds were checked; total audio duration is 10,968.44 seconds (about 3.05 hours). Header/duration checks do not prove perceptual alignment. Exclude 04_BN3-154-E_comp and 04_Jazz1-200-B_comp because of the [reported timing errors](https://github.com/marl/GuitarSet/issues/5), leaving 358 candidates. No timing corrections were applied.

Each file has instructed and performed chord layers, selected by metadata rather than annotation position. Performed labels are derived from note annotations with lead-sheet roots and boundaries; they are not independent manual ground truth. Among retained recordings, the existing exact-shorthand mapping supports 100% of instructed labeled duration and 66.34% of performed labeled duration. Instructed labels use maj/min/7/hdim7. Performed labels add further qualities but have 3,678.62 seconds of OOV labels, including explicit power chords such as C#:(1,5)/1 and altered chords. OOV is a representation limitation, not necessarily an annotation error.

Do not silently map power chords to major/minor or switch to instructed labels solely to improve coverage. Next compare exact pitch-set equivalence via the established Harte parser and explicit partial-target supervision; preserve distinctions where the third is absent or ambiguous. This requires a separate tested mapping decision before training admission. Current audit reuses the existing mapping without changing it.

The [project documentation](https://guitarset.weebly.com/) identifies six players, 30 lead sheets and only three underlying progressions. recordings.csv contains player, mode and conservative candidate style/progression groups, collapsing tempo/key variations within each style/code. Cross-style versions still need grouping; these are not certified composition IDs. Five-fold progression-independent evaluation cannot be assumed appropriate with only three underlying progressions. Choose unseen-player or unseen-progression evaluation explicitly and keep comp/solo counterparts together as required. No folds have been assigned.

## Reproduce

```powershell
.venv/Scripts/python.exe src/acr/audit_guitarset.py --download --output runs/guitarset-audit-repeat
```

The command verifies both pinned archives, reads headers and JAMS directly from ZIPs, then writes summary.json, recordings.csv, symbols.csv and qualities.csv to a fresh output directory. No audio extraction or training cache is produced. Install requirements.txt first. Existing full archives are reused and verified; interrupted downloads restart to a .part file.

## Next gate

Resolve vocabulary coverage and grouping, then extract a consistent feature representation for an audio baseline. Keep the source tagged so both F and E can be evaluated separately. This inventory is complete; feature extraction, audio spot checks, train/validation splits and training remain pending. See STANDARD_EVALUATION.md for the interval metric contract.


## Label contract and progression folds — 20 November 2026

The follow-up is implemented in src/acr/guitarset_labels.py, using mir_eval 0.8.2's Harte parser. Run with the optional evaluation dependencies installed:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-eval.txt
.venv/Scripts/python.exe src/acr/guitarset_labels.py --output runs/guitarset-labels-repeat
```

Relative pitch sets retain explicit additions/omissions; upper degrees wrap by pitch class. The annotated root is preserved. Bass is excluded from quality comparison, consistently with the existing root/quality task, rather than inserted into the pitch set. Compare the resulting set with all 14 quality templates: only exact equality produces a complete class. This does not infer an absent third. Unlike mir_eval's evaluation encode helper, the construction does not reinsert the bass/root after an explicit root omission. Malformed syntax fails loudly.

Results: exact equivalence leaves performed coverage at 66.34% (7,250.48 seconds); 3,678.62 seconds remain outside the complete vocabulary. There are no additional equivalent spellings to recover in this release under this policy. Every previously supported symbol keeps its complete class ID. Inversions retain the existing collapsed-quality semantics; no bass head is introduced.

For unsupported but valid chord labels, event_targets.csv proposes gate=chord and the annotated root, while triad/extension/complete are -100. These roots may originate from the lead sheet; they are not independently verified sounding bass notes. N has gate=no-chord and no component targets; X masks everything. This is a proposal artifact only: current trainers still derive targets from complete labels and have not been changed to consume partial targets. The initial audio pilot may use complete-supported intervals, with OOV masked; partial supervision requires a separately tested training change and ablation.

### Split contract

All 360 instructed sequences were normalized by subtracting the first root, dropping duration and collapsing adjacent equal chord tokens. This yields exactly three distinct progression signatures. Each filename code ending in 1/2/3 agrees with its corresponding signature across all five styles, six players, keys, tempi and comp/solo modes. Group identifiers are therefore supported by annotation evidence, not filename intuition alone. The source describes these as three underlying progressions; the manifest uses numeric groups to avoid assigning unverified musical names.

Created three outer folds. For outer fold k, progression k is test, (k+1) modulo 3 is validation and the remaining progression is train. The two timing-erratum recordings are excluded in every fold. Train/validation/test counts are 119/120/119 in fold 0, 119/119/120 in fold 1 and 120/119/119 in fold 2. No performance results informed this allocation. All variants of a progression and comp/solo pairs stay together. The same players appear across partitions: this tests unseen progression, not unseen performer. With only three progression groups, fold uncertainty will be large and the training pool small; do not describe it as the BTC five-fold protocol.

The authoritative manifest is reports/p2/guitarset-labels/folds.csv. Training must select one fold explicitly and use only its train rows; normalization fits train only, model selection uses validation only, and test outcomes are withheld until choices are frozen. Do not train on the union of all folds before reporting outer-test results. The annotation inventory has exposed label distributions for auditing; no model has been evaluated on these test partitions.

Files under reports/p2/guitarset-labels/: summary.json, symbol_mapping.csv, event_targets.csv, folds.csv, fold_quality_support.csv and verification.json. The latter records 48 passing tests and full-inventory checks for disjoint progression groups, paired modes and unchanged legacy class IDs. Earlier statements that folds were pending describe the initial inventory stage.

### Next implementation boundary

Feature extraction remains pending. Prefer one documented CQT audio front end for a BTC-style baseline and both teacher branches, with training-only normalization. Preserve these manifests and source identities. The existing 24-bin chroma models cannot accept CQT directly without an explicit new input-model configuration. Retain their previous results as controls; do not load their checkpoints into a different front end and call it the same experiment. No GPU training notebook is ready from this mapping-only change.
