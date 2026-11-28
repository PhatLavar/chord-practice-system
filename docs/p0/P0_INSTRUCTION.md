# P0 instructions · Prepare Billboard data for model training

P0 pairs each recording’s 24-bin Chordino features with detailed chord labels, checks timestamp coverage, groups repeated chart entries, and freezes the split and vocabulary statistics used by P1. Run P0 on CPU. A GPU and original audio are unnecessary for this stage.

## 1. Prepare the three inputs

From the [McGill Billboard release page](https://ddmal.ca/research/The_McGill_Billboard_Project_%28Chord_Analysis_Dataset%29/), download the following files into `data/raw/`. Use the local names in the table and keep both archives compressed. Check SHA-256 against [the recorded hashes](../../reports/p0/summary.json) with `Get-FileHash data/raw/* -Algorithm SHA256` in PowerShell.

| Download | Local filename | Used for |
|---|---|---|
| [Index v2.0](https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1) | index.csv | Slot IDs, artists and titles |
| [Chordino v2.0](https://www.dropbox.com/s/e9dm23vbawg9dsw/billboard-2.0-chordino.tar.xz?dl=1) | billboard-2.0-chordino.tar.xz | Timestamp and 24 chroma values per frame |
| [Detailed LAB v2.0.1](https://www.dropbox.com/s/t390alzrkx0c9yt/billboard-2.0.1-lab.tar.xz?dl=1) | full.tar.xz | Start, end and detailed chord symbol for each interval |

The original P0 run used chroma from the Kaggle mirror. All 890 chroma files matched the McGill release byte for byte. The current workflow reads the publisher’s archives directly.

## 2. Inspect a single recording (optional)

Run commands from the repository root with Python 3.11 or newer. On Windows you can replace `python` with `.\.venv\Scripts\python.exe`.

```powershell
python -m acr.data.billboard.audit 0003 --output runs/p0-0003
```

The reader opens recording 0003 inside both archives. Open `runs/p0-0003/timeline.svg` to inspect the reference intervals against frame timestamps. In `frames.csv` each frame uses the interval containing its timestamp, including the start and excluding the end. An empty label means uncovered; N is an annotated no-chord target. For 0003 expect 95 intervals and 3,250 covered frames. The notebook calls the same reader.

## 3. Run the corpus audit

```powershell
python -m acr.data.billboard.corpus --raw data/raw --output runs/p0-corpus
```

The command reads every available pair, records failures, then groups eligible slots into songs. One representative per song supplies examples. Only labeled training representatives contribute to the quality counts.

| Generated file under runs/p0-corpus/ | What to check |
|---|---|
| inventory.csv | 1,300 index rows, 890 pairs, 883 eligible slots. Inspect the status column for each exclusion. |
| manifest.csv | 734 groups: 367 labeled train, 146 hidden train, 110 validation, 111 test. |
| qualities.csv | Training events, seconds, songs, root counts and rare-quality flags. |
| symbols.csv | Mapping of observed training symbols to quality, pitch class, triad and extension. |
| summary.json | Group counts, artist-credit overlap and input checksums. |

The versioned audit results are stored under `reports/p0/`. Each execution writes its outputs to `runs/p0-corpus/`. Written documentation is stored under `docs/p0/`.

## 4. Verify the split and hand it to P1

In PowerShell compare the generated manifest with the [frozen manifest](../../splits/billboard-v1.csv). The following command prints any differing lines; a matching pair produces no output.

```powershell
Compare-Object (Get-Content splits/billboard-v1.csv) (Get-Content runs/p0-corpus/manifest.csv) -SyncWindow 0
```

For P1, read IDs as strings to preserve leading zeros. Select rows with `partition == train_labeled` and `id == representative` to get 367 training songs. Select validation representatives for checkpoint selection. Use the ID to read feature and label entries from the same archives. Recording 0003 is already a training representative and needs no separate copy. The [P0 report](P0_REPORT.md) records the exclusion rules, label masks and rare-quality definition used in this split.
