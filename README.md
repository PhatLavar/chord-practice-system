# Chord Practice System

**English** · [Tiếng Việt](docs/README-VIE.md)

This project aims to help musicians practice with a recording by providing two independent timelines. One shows the chords and when they change. The other marks boundaries between musical sections. The corpus audit (P0) is complete. Model training and evaluation are planned. [Model Architecture Plan v1.2, sections 7 and 13](https://drive.google.com/file/d/1gmz-JaN6I_5oCi65DYxmTpX5nlbG2_tO/view)

## Windows setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select the interpreter from `.venv` in your IDE. Place the detailed McGill Billboard annotations and Chordino features in `data/raw/`. Configure input and output paths when moving to Kaggle so the same code runs in both environments.

## Run the first P0 audit

From the project root, run `python src/acr/audit.py 0003 --output runs/p0-0003` or open [the audit notebook](notebooks/00_data_audit.ipynb) in Jupyter. The CLI uses only the Python standard library. The demo reads one recording directly from the official archives. See the [P0 instructions](docs/p0/P0_INSTRUCTION.md) for downloads and the full corpus command.

Open `timeline.svg` in the output directory to inspect interval and frame coverage. `frames.csv` retains all 24 chroma bins and assigns each timestamp the reference label at that instant (start inclusive, end exclusive). A blank label means no annotation covers the frame. `N` is an annotated no-chord label. `summary.json` records the counts.

The full CPU audit is complete. See the [P0 report](docs/p0/P0_REPORT.md) for data sources, exclusions, vocabulary and reproduction steps, and the [frozen song split](splits/billboard-v1.csv) for P1.

## P1 training

The first Kaggle baseline is complete: 57.11% whole-chord validation frame accuracy. See the [P1 report](docs/p1/P1_REPORT.md) for the verified results and rare-quality limitations. Follow [P1 instructions](docs/p1/P1_INSTRUCTION.md) and import [the Kaggle notebook](notebooks/01_teacher_f.ipynb). Store each run under a new name in `runs/`.

## P2 oracle event teacher

The three-seed oracle-event comparison is verified: natural E averages 63.33% overall accuracy; the 25% mixture averages 60.58% overall accuracy and 7.88% rare macro recall. The report records precision and support limitations. See [P2 instructions](docs/p2/P2_INSTRUCTION.md), [P2 report](docs/p2/P2_REPORT.md) and [the oracle notebook](notebooks/02_teacher_e_oracle.ipynb).

## Project layout

| Path | Purpose |
|---|---|
| `docs/` | Markdown guides and written reports. |
| `reports/p0/` | Versioned P0 audit tables and summary. |
| `src/acr/` | Dataset readers, time alignment, label mapping, models, and evaluation. Add modules as each milestone begins. |
| `notebooks/` | Planned entry points include `00_data_audit`, `01_teacher_f`, and `02_teacher_e_oracle`. Shared logic belongs in `src/`. |
| `configs/`, `splits/` | Experiment settings and manifests that assign songs to splits. |
| `data/raw/`, `data/processed/` | Original data and local caches. |
| `runs/<run_id>/` | Configuration, Git commit, data version, metrics, figures, and checkpoints for one run. |

## Implementation milestones

1. **P0 · Data audit.** Pair annotations with `bothchroma.csv` by recording ID and timestamp. Plot one recording to check alignment before processing the corpus. Group duplicate songs and freeze the split. For each chord quality, count events, total duration, and distinct songs in the training set before augmentation. Use the audit to define the vocabulary and the criterion for rarity.
2. **P1 · Frame teacher F.** Overfit a small temporal convolutional network on a few training songs to check the pipeline. Then train on the labeled split and evaluate on unseen validation songs. Record per-quality scores and support, timelines, runtime, and memory use.
3. **P2 · Event teacher E with reference boundaries.** Train E with independent weights by averaging hidden features within each reference interval. At evaluation, aggregate F’s probabilities over the same intervals. For each quality and chord component, count where both teachers are correct, only F is correct, only E is correct, or both are wrong.
4. **P3 · Predicted boundaries.** Train E’s boundary head. Repeat the comparison using a fixed event-matching rule and measure any drop in performance from P2.
5. **Student experiments.** Compare S0 (human labels only), S1 (one teacher), S2 (two teachers agreeing on the whole chord), and S3 (agreement on individual components). Hold the labeling budget, dataset, and student capacity fixed. The primary metric is macro event recall across rare qualities, scored on complete chords. Selective knowledge distillation (S4) and encoder substitutions are optional experiments.

The music structure analysis (MSA) branch identifies practice sections independently of chord changes. The demo will identify whether each timeline comes from annotations or predictions.

## Commit conventions

Give each commit one verifiable purpose. Examples include `[DOCS] document the Billboard split protocol`, `[FEAT] align Billboard features for one recording`, and `[FIX] handle the final chord interval`. Record the commit ID for each experiment run.

## Documentation language

English is the primary language for project documentation. Place new documents under `docs/` and group them by topic as needed. Give Vietnamese translations the suffix `-VIE.md` and link them to the English original. Update both READMEs in the same commit when instructions or plans change.
