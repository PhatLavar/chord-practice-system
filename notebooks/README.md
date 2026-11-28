# Notebook entry points

Notebooks are grouped by phase and source. Import the desired file into Kaggle and run cells in order; use the current file rather than an old saved draft with flat script paths.

| Folder | Role |
|---|---|
| p0/ | Initial Billboard audit and timeline visualization |
| p1/ | Historical frame-teacher baselines/ablations |
| p2/billboard/ | Oracle event teacher and sampling/seed studies |
| p2/jaah/ | Auxiliary-source F/E experiments |
| p2/guitarset/ | Audio CQT, player split, augmentation and repeats |

The latest historical repeat notebook is [02k](p2/guitarset/02k_guitarset_transposition_repeats.ipynb); its results are already verified. Current baseline development is Stage B and has not introduced a new GPU notebook. See [the structure guide](../docs/architecture/PROJECT_STRUCTURE.md) and [Stage B instructions](../docs/baselines/BASELINE_AND_CORPUS_PLAN.md).

Setup pulls current source and installs the acr package. Project CLI calls use python -m acr..., while reusable implementations stay under src/acr/. Historical run data/config hashes are not rewritten during the file migration.
