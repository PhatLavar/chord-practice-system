# Project structure and refactor contract

Updated 28 November 2026. This structure separates model research responsibilities while preserving the original experiment history. The refactor changes imports, invocation and file ownership; it does not change architectures, mathematical kernels, training hyperparameters, split membership or recorded results.

## Source ownership

```text
src/acr/
  vocabulary.py          # Shared symbols, decomposition and complete-label IDs
  provenance.py          # Historical config-path resolution
  data/
    billboard/           # Audit, corpus manifest and feature-cache preparation
    guitarset/           # Release audit, label mapping and CQT/cache time alignment
    jaah.py              # Jazz source reader/preparation
    events.py            # Event dataset and collation
    windows.py           # Frame windows and balanced sampling
    augmentation.py      # CQT/root transposition
  models/
    tcn.py               # FrameTeacher + EventTeacher sharing the TCN family
    components.py        # Factor targets, losses and constrained decoding
    transformer.py       # Historical compact project Transformer
    btc.py               # Project adapter for the pinned author BTC
  training/              # Frame, event and GuitarSet optimizer loops
  evaluation/            # Interval metrics, saved-logit diagnostics and train-fit checks
  experiments/
    billboard/           # Repeat controllers, mixture comparisons, frozen reference
    jaah/                # Frame/event auxiliary-source experiments and pair analysis
    guitarset/           # Memorization and seed-repeat orchestration
    baselines/           # B0 BTC compatibility/learning diagnostic
  vendor/btc/            # Pinned upstream code and MIT/provenance files, unchanged
```

Group code by responsibility and corpus, not by a new class/file for each run. FrameTeacher and EventTeacher now live together because they share the TCN backbone. The window datasets no longer live inside the training script. Vocabulary no longer depends on Billboard preparation. Interval reading belongs to data, while scoring and confusion analysis belong to evaluation. Data/model modules must not import optimizer-loop modules; a test checks this boundary.

Do not merge unrelated CLI scripts into one large class just to reduce the file count. Sampling policies, training and statistical comparisons have different state and verification needs. Add configuration to an existing experiment family when behavior is shared; create a new module only for a distinct responsibility. Package __init__.py files mark boundaries and do not eagerly import GPU/audio dependencies.

## Experiments, instructions and evidence

```text
configs/{p1,p2,baselines}/
notebooks/
  p0/
  p1/
  p2/{billboard,jaah,guitarset}/
docs/
  architecture/          # This guide
  evaluation/            # Common metric contract
  p0/, p1/
  p2/{guitarset,jaah,research}/
  baselines/             # Current Stage B plan, source review, instructions and log
reports/                 # Versioned machine evidence, grouped by experiment
data/{raw,processed}/    # Local sources/caches, ignored by Git
runs/                    # Local immutable run artifacts, ignored by Git
tools/verify_structure.py
```

Historical report JSON/CSV, raw data, split manifests and checkpoints retain their paths and contents. Numerous history/metric files are evidence, not necessarily redundant code: do not merge or delete them because source folders were reorganized. Future experiment evidence should use a run/condition subfolder when it grows. Changing recorded source hashes or old metadata paths would destroy provenance, so those literal historical values are intentionally not rewritten. Documentation links and current executable recipes are updated.

The exact old→new file/module map is [structure-migration.json](../../reports/maintenance/structure-migration.json), including symbol extractions. This is an intentional CLI migration with no collection of duplicate flat-script wrappers. Reproduce an archived run with its recorded source revision; use current commands and fresh outputs for new runs. The config resolver recognizes earlier flat p1/p2 references when reading archived cache_config metadata without modifying that metadata.

## Install and invoke

From the repository root:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-audio.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
.venv/Scripts/python.exe -m acr.training.frame --help
.venv/Scripts/python.exe -m acr.training.event --help
.venv/Scripts/python.exe -m acr.training.guitarset --help
.venv/Scripts/python.exe -m acr.experiments.baselines.btc_check --help
```

Use requirements.txt instead of requirements-audio.txt when audio processing is unnecessary. pyproject.toml defines the editable src-layout package; it does not introduce a second set of runtime dependency versions. Imports are fully qualified, such as from acr.models.tcn import FrameTeacher. Run project commands at the repository root because relative data/config paths intentionally refer to that root.

For example, the player experiment config is configs/p2/p2-guitarset-cqt-player.json; the notebook is notebooks/p2/guitarset/02i_guitarset_player_teachers.ipynb. Kaggle notebook setup clones/pulls the repository, installs the local package and then imports project code. The P0 audit notebook also locates the repository root when opened from its nested folder.

Reimport updated notebooks into Kaggle. A previously saved draft with flat script paths cannot be updated by moving files in this checkout. Completed-run provenance guards still reject stale code/config reuse; preserve old artifacts and select a fresh run path when required. Do not disable those guards to make a saved notebook appear compatible.

## Verification commands and limits

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe tools/verify_structure.py
.venv/Scripts/python.exe tools/verify_structure.py --smoke --output runs/notebook-smoke-fresh
```

Static validation compiles every notebook's Python cells, resolves their package CLI modules, checks current config references and checks local Markdown links. The optional smoke command requires existing local source archives/caches. It executes the actual P0 audit cell without IPython rendering and input/cache plus CPU smoke cells 2/3 from 02h–02k, routing output into a fresh directory. These include natural/transposition F/E training and matched-seed repeat orchestration. It does not clone/pull remotely, reinstall dependencies, render plots, replay full GPU training or claim a Kaggle browser execution.

Pre/post proof uses the same deterministic inputs/seeds to compare exact logits, state-dict keys and parameter counts for F, E, compact Transformer and author BTC. Sixteen moved model/data/scoring kernels retain identical ASTs; all 19 configs retain identical values after reversing path translation. Packaging, CLI import checks, unit tests and real notebook smoke execution complement that proof. Verification evidence is [structure-verification.json](../../reports/maintenance/structure-verification.json); it records observed completion and limitations rather than treating syntax checks as a full training run.
