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
