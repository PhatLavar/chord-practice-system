# P2 JAAH auxiliary-data pilot

Status (20 November 2026): data inventory and implementation prepared locally. Full Kaggle training and any improvement claim are pending.

Local verification: 36 unit tests pass, including unchanged 24-channel outputs/RNG, 12-channel gradients into the shared trunk, mixed-channel batch rejection, external-label masking and the fixed replacement budget. A tiny training-only memorization check reached 8/8 correct Billboard events and 8/8 correct JAAH events after 84 alternating epochs (168 optimizer steps). See [overfit_check.json](../../reports/p2/jaah-inventory/overfit_check.json). This is an execution/learnability check, not validation performance or evidence that adding JAAH improves ACR.

## Selected data and research question

Use JAAH as the first external training source. Its pinned public release supplies frame-level chroma and aligned LAB annotations, allowing an experiment without acquiring additional audio. Keep JSIR as a promising rare-harmony candidate requiring matched audio, and GuitarSet as a later guitar-domain experiment requiring feature extraction. Do not combine all three in this first pilot.

Question: can auxiliary jazz supervision through a separate 12-channel input projection improve the shared event teacher's rare-chord recognition on the unchanged Billboard validation set?

This tests a combined auxiliary-domain/representation intervention. It does not isolate the contribution of genre, amount of data or input information individually. It does not test the 12-channel branch on unseen JAAH recordings or establish cross-dataset generalization.

## Audited JAAH inventory

Source: [MTG/JAAH](https://github.com/MTG/JAAH/tree/6fec3f6dd1d60c21cf9af855ea173be38e6d7243), revision 6fec3f6dd1d60c21cf9af855ea173be38e6d7243. Source-file hashes, exclusions and cache hashes are recorded in [summary.json](../../reports/p2/jaah-inventory/summary.json).

- Downloaded metadata, LAB archive and 12-dimensional frame chroma for 113 recordings.
- Kept 103 recordings with 15,491 supported, nonempty merged events, including N.
- Excluded ten recordings because annotation/metadata duration and feature duration differ beyond the conservative tolerance. Exclusion does not prove which source is wrong. No timestamp shifts were inferred or applied.
- Retained 1,611 unsupported annotation intervals as masked regions; they are not N targets. Supported consecutive identical chord targets are merged; unsupported intervals remain breaks.
- Bare Harte roots such as `F` are canonicalized to `F:maj` for JAAH only. Altered, extended and explicit-interval labels outside the existing exact vocabulary remain masked. Billboard mapping remains unchanged.
- Verified 13 CSV columns (timestamp + 12 bins), finite nonnegative features, strictly regular timestamps, ordered nonoverlapping LAB intervals and duration consistency. Both feature clocks have a step of 2048/44100 seconds, approximately 46.44 ms.
- Applied per-frame L1 normalization to the JAAH 12-bin vector. Billboard keeps its existing separate normalization of its two 12-bin blocks.

| Quality | Usable merged events | Recordings |
|---|---:|---:|
| dim | 76 | 11 |
| aug | 17 | 8 |
| min6 | 237 | 28 |
| minmaj7 | 42 | 8 |
| hdim7 | 429 | 38 |
| dim7 | 617 | 46 |
| sus2 | 0 | 0 |
| sus4 | 0 | 0 |

See [qualities.csv](../../reports/p2/jaah-inventory/qualities.csv) for all classes and normalized-title composition counts, [inventory.csv](../../reports/p2/jaah-inventory/inventory.csv) for inclusion decisions and [symbols.csv](../../reports/p2/jaah-inventory/symbols.csv) for source-symbol counts. Raw symbol counts and merged event counts are different units.

Normalized titles were compared conservatively against every title in the Billboard index, including protected partitions. No exact normalized-title overlap was found. This is a metadata screen, not audio fingerprint verification or proof that all alternate-title/cover overlaps are absent. Only train_labeled and validation Billboard features/labels are loaded; hidden/test labels remain unused. JAAH is auxiliary training only, with no held-out JAAH performance claim. Duration agreement also does not establish perfect local alignment; further listening/alignment review would require matching audio.

## Model and controlled comparison

```text
Billboard 24 bins -> original 24-to-48 projection --+
                                                 +-> shared TCN -> event pooling -> four shared heads
JAAH 12 bins ----> new 12-to-48 projection ---------+
```

The new projection does not reconstruct missing bass/treble information. It learns a representation that can feed the shared trunk. Each batch has a single input representation; mixed-channel batches are rejected. Teacher F stays fixed and independently trained. Its frozen event decisions are used for complementarity after each full E run.

Both arms start from scratch at seed 20261119, width 48, batch size 32, learning rate 0.001, 30 epochs, Adam, gradient clipping 5, unweighted four-head losses and E25's within-dataset sampling policy. Both models contain the same additional projection; the control never uses it. Construction of that projection preserves the original model's initialization and CPU sampling RNG stream. Auxiliary sampling also preserves the Billboard CPU sampling RNG stream.

| Arm | Billboard batches/epoch | JAAH batches/epoch | Total optimizer steps/epoch |
|---|---:|---:|---:|
| billboard_control | 990 | 0 | 990 |
| jaah_mixed | 742 | 248 | 990 |

The 25% fraction rounds to 248 of 990 batches (25.05%). JAAH batches replace Billboard batches, not add extra optimizer steps. Batch sizes follow the Billboard schedule, including the final partial batch, so both arms draw 31,679 supervised events per epoch. Actual per-source/per-quality exposure is recorded each epoch. JAAH natural draws use sampling with replacement; its balanced branch uses the same quality/song policy as E25. Billboard retains the original shuffled-event loader.

Validation uses the same 110 Billboard songs, 9,673 reference intervals and original 169-class output vocabulary. The highest validation complete-event accuracy selects the checkpoint, matching the earlier policy. Keep the historical E25 result as context, but use the freshly trained control for the primary paired comparison.

## Run on Kaggle

1. Import [02f_teacher_e_jaah_pilot.ipynb](../../notebooks/02f_teacher_e_jaah_pilot.ipynb).
2. Enable Internet and a GPU; stop an older GPU session if Kaggle's concurrent-session limit is reached.
3. Run cells in order. The notebook updates the repository, prepares the official Billboard cache if absent, downloads only the required pinned JAAH files and verifies the resulting sources against the committed inventory.
4. A short CPU smoke run checks both branches. Its scores are not research results. Then run the two full GPU jobs.
5. Save the executed notebook version. Download `/kaggle/working/p2-jaah-pilot-results.zip` and place it in local `runs/imports/`.

Output directories are protected against overwriting. An interrupted full pilot is not automatically resumed; preserve completed artifacts and use a fresh output root for a deliberate retry. The exported ZIP contains both runs, checkpoints, event logits, predictions, histories, source exposure, per-quality precision/recall and F/E complementarity. It contains no raw JAAH data.

Local commands, from the repository root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests
# First preparation only; use fresh output/report paths for another preparation.
.venv/Scripts/python.exe src/acr/prepare_jaah.py --download --report runs/p2-jaah-preparation
.venv/Scripts/python.exe src/acr/experiment_jaah.py --output runs/p2-jaah-check --smoke
# Full run requires an available CUDA device.
.venv/Scripts/python.exe src/acr/experiment_jaah.py --output runs/p2-jaah-pilot --device cuda
```

Raw inputs live in data/raw/jaah/; normalized cache in data/processed/p2-jaah/; these are ignored by Git. Human-readable instructions stay in docs/ and machine-readable evidence in reports/. Models/results stay in runs/.

## Decision after training

Compare complete-event accuracy, macro quality recall, rare macro recall, per-quality precision and correct rare-event/song counts. Inspect whether rare successes add information beyond F, and whether maj/min or sus classes regress. Re-run confidence/coverage analysis on saved outputs before accepting any pseudo-labeling policy.

An increase in rare recall alone is insufficient if it comes from many false positives or a large common-chord regression. A promising single-seed result must be repeated with matched seeds before changing the preferred teacher. A negative result should be retained and interpreted as evidence about this specific mixture, mapping and input-adapter setup, not proof that external data cannot help. No threshold, dataset or model is selected based on hidden/test outcomes.
