# P1 instructions · Frame teacher F

P1 trains a compact non-causal TCN to label each chroma frame. Begin with the two-song overfit check, then run the full labeled baseline. Both configurations are initial pilot choices. A passing overfit result establishes that the model can learn its training examples. Use validation results to assess generalization.

## Files and inputs

| File | Purpose |
|---|---|
| src/acr/p1_data.py | Verify official source checksums, load the frozen representatives, cache normalized chroma and targets. |
| src/acr/teacher_f.py | TCN, component targets, masked losses and complete-chord decoder. |
| src/acr/train_f.py | Train, select checkpoints and export predictions and scores. |
| configs/p1-overfit.json | Two labeled training songs, 200 epochs, fixed seed. |
| configs/p1-baseline.json | All 367 labeled training songs, 110 validation songs, 30 epochs. |
| notebooks/01_teacher_f.ipynb | Kaggle entry point with downloads and commands. |

P0's 14 qualities and N define 169 whole-chord classes. Use the frozen manifest and the three official files listed in [P0 instructions](../p0/P0_INSTRUCTION.md). Each 12-bin chroma block is normalized by its own sum. Zero blocks stay zero. X, OOV and uncovered frames are masked. The cache contains labeled training and validation representatives.

## Model and pilot decisions

Five residual convolutions use dilations 1, 2, 4, 8 and 16, with width 48. The receptive field covers 63 frames. Each 512-frame training target window includes 31 frames of neighboring context on both sides. Tail targets are padded with the ignore label. Four heads estimate no-chord, root, triad and extension. The loss sums each head's mean cross-entropy over its observed targets. N trains the no-chord head and masks its other components.

The no-chord gate chooses N when its N logit exceeds its chord logit. Otherwise, the decoder selects the root and the supported triad/extension tuple with highest summed log probability. Ties follow the fixed vocabulary order. This baseline uses Adam and gradient clipping, with no augmentation or class balancing. Add those as recorded experiments after the initial pipeline check.

## Local preparation

Run from the repository root after installing requirements into the project environment.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/acr/p1_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe src/acr/train_f.py --config configs/p1-overfit.json --output runs/p1-overfit-local --device cpu
```

Preparation creates data/processed/p1. Output directories must be new to protect earlier runs. A failed preparation may leave an incomplete cache; remove that specific incomplete directory before retrying. A complete cache has metadata.json. The full cache takes a CPU pass through the archives.

## Kaggle walkthrough

1. Open Kaggle, create a notebook and import notebooks/01_teacher_f.ipynb.
2. Enable Internet for the public GitHub clone and official data downloads. Select a GPU accelerator for training.
3. Run the first cell. Confirm the printed Git commit, PyTorch/NumPy versions and CUDA status. The notebook uses Kaggle's installed PyTorch build. When CUDA is unavailable it uses CPU; enable the accelerator before starting the full run.
4. Run the data cell. It downloads the three official archives, verifies their hashes during preparation and runs the focused tests. The reduced-label Kaggle mirror is unnecessary.
5. Run the overfit cell. Inspect history.json and metrics.json. The pilot pass threshold is at least 95% whole-chord frame accuracy on the same two training songs.
6. Once the check passes, set RUN_BASELINE = True and execute the baseline cell. It selects the best epoch by validation frame accuracy. The test and hidden-label partitions are reserved for later stages.
7. Run the final cell to create /kaggle/working/p1-results.zip. Save the executed notebook version and download that archive to retain the run.

## Interpret the outputs

run.json records source hashes, split hash, exact song IDs, configuration, revision, dirty-worktree flag, source-code hashes, environment and parameter count. history.json records per-head training losses, epoch runtime and evaluation scores. best.pt stores the selected weights and vocabulary. metrics.json identifies whether evaluation used training or validation songs and records CUDA peak allocated memory where available.

predictions/ contains timestamps, reference labels, predictions and all four heads' logits per song. It also contains a confusion matrix, per-song counts and per-quality recall requiring the complete chord to match. Frame accuracy and macro chord recall exclude masked frames. These are frame-based P1 diagnostics. Rare-quality event recall, event matching and temporal decoding require a separately fixed evaluation protocol before the student comparisons.

The returned Kaggle runs have been verified. See the [P1 report](P1_REPORT.md) for the baseline results and limitations. Pin the recorded code revision when reproducing an experiment.


## Local verification on 19 November 2026

The cache contains all 367 labeled training and 110 validation representatives. The focused tests passed for all 169 supported chord targets, masked losses, enharmonic/inversion mapping, and tail padding. The local CPU model has 37,225 parameters.

The first overfit run used learning rate 0.003 for 100 epochs and reached 94.51% training frame accuracy. The selected configuration uses learning rate 0.001 for 200 epochs and reached 95.58% at epoch 195 on recordings 0003 and 0006 (5,810 scored frames). The full histories, checkpoints and predictions remain in local runs/. [Recorded configurations and results](../../reports/p1/local-overfit.json)

These local results are training-set pipeline checks. The Kaggle overfit check and full baseline have since completed. See [P1_REPORT.md](P1_REPORT.md).


## Reproduce the saved-logit diagnostic

After importing the Kaggle baseline into runs/p1-baseline-kaggle/, run from the repository root:

```powershell
python src/acr/diagnose_f.py --output reports/p1/kaggle-diagnostics
```

Choose a new output directory if it already exists. The script checks the detailed-label archive hash, reconstructs frame references and predictions, and reproduces the stored confusion matrix before reporting diagnostics. It averages probabilities on reference intervals and reports standalone component accuracy, per-quality event support, and a 0.1-second boundary diagnostic. See [P1_REPORT.md](P1_REPORT.md) for denominators, excluded events, limitations and the evolving experiment log.


## Weighted component-loss experiment

Import [01b_teacher_f_weighted.ipynb](../../notebooks/01b_teacher_f_weighted.ipynb) into Kaggle. Enable Internet and GPU, then run its cells in order. The setup clones the repository or fast-forwards an existing checkout, prints the revision and reuses a completed P1 cache. If a previous preparation was interrupted, verify that data/processed/p1/metadata.json exists before reusing it.

The notebook trains with configs/p1-weighted.json, records the actual training counts and weights in loss_weights.json, evaluates the validation checkpoint, then runs the reference-event diagnostic. It produces /kaggle/working/p1-weighted-results.zip. Save the executed Kaggle version and download that archive for comparison.

On an existing Kaggle notebook with the data prepared, you can instead add and run these cells:

```python
subprocess.run(["git", "pull", "--ff-only", "origin", "main"], check=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], check=True)
weighted_run = project / "runs/p1-weighted-kaggle"
subprocess.run([sys.executable, "src/acr/train_f.py",
                "--config", "configs/p1-weighted.json",
                "--output", str(weighted_run), "--device", "cuda"], check=True)
subprocess.run([sys.executable, "src/acr/diagnose_f.py",
                "--run", str(weighted_run),
                "--output", str(weighted_run / "diagnostics")], check=True)
import shutil
print(shutil.make_archive("/kaggle/working/p1-weighted-results", "zip",
                         project, "runs/p1-weighted-kaggle"))
```

Choose a fresh run name for a retry. Review the [P1 experiment log](P1_REPORT.md) for the weighting formula, fixed controls and interpretation rules. This is a new training run from the seeded initialization.
