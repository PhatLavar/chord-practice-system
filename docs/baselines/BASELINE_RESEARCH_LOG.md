# Stage B research log

## 21 November 2026 — Stage separated from P2

The user requested a new phase for the literature baseline direction and required both single-corpus and multi-corpus work to be considered. Established Stage B, with B0 protocol/compatibility, B1 single-source baselines, B2 cross-source/multi-source experiments and B3 consolidation. The [plan](BASELINE_AND_CORPUS_PLAN.md) defines the matrix, admission dependencies, budget controls and evaluation limits.

Moved the baseline source review to [FRAME_EVENT_BASELINE_REVIEW.md](FRAME_EVENT_BASELINE_REVIEW.md) and its machine inventory to reports/baselines/source_review.json. Historical P2 results, checkpoints, datasets and code remain as evidence. P2 is an archived exploratory stream for subsequent baseline work; this does not assert that all original teacher-reliability goals were met.

Decision: a new research stage is appropriate because the model families, event inference task and corpus-transfer questions differ from the original oracle E milestone. A destructive reset is not justified. A new name does not erase prior validation exposure; more corpora or stronger models must demonstrate benefit under controlled comparisons. F and E may eventually prefer different corpus mixtures, after separately identifying architecture and data effects.

Status: documentation and organization only. No new model imported/trained; no new corpus admitted; no new split claimed as untouched. Next action is B0 compatibility and input-contract work, with audio admission investigated alongside it. Future weekly observations and decisions belong here or in experiment-specific Stage B reports, not P2_REPORT.md.

## 21 November 2026 — B0 author BTC compatibility completed

Pinned jayg996/BTC-ISMIR19 at 2682317be668032e6e4b269ded36adaa2ad57df0 and included only its MIT-licensed model/transformer modules with attribution and source hashes. All model class ASTs match upstream. Changes are package-relative imports, removal of unused YAML/demo code and the modern NumPy equivalent of np.float. Preserved author-specific normalization, attention, final feed-forward activation/dropout and unused LSTM parameters; this is not a generic replacement Transformer.

Added btc_adapter.py with explicit project/BTC integer mapping, 170 raw logits and fixed-window/full-track I/O. Verified every supported chord ID against the pinned author conversion method, plus N and unknown distinctions. B0 retains ignored OOV training targets rather than silently creating N labels; native-unknown supervision and published-recipe equivalence remain separate decisions. Existing GuitarSet cache/splits/normalization and TCN controls are preserved.

Ran the full 3,036,842-parameter model on CPU from scratch. Four predetermined training windows contain 240 scored frames across six qualities. The final count-based diagnostic passes at the first checked 95% result: 228 correct frames at update 160; no validation score or new recording selected the stop. Full-track tiling produces 170×241 finite logits on a selected training recording. A preliminary floating-point-mean check was superseded by integer correct/total stopping to handle the exact threshold reliably. This is concentrated fitting evidence, not a generalization score or a B1 training run.

Checkpoint reload reproduces the score. Dominant 7 remains only 5/16 correct while the other selected qualities are mostly/all correct. The predefined global gate therefore establishes basic learning compatibility, not uniform quality mastery. Report that concentration explicitly before a full experiment. Superseded fitting artifacts and the temporary upstream-source snapshot were removed; the final diagnostic run and vendored provenance are retained.

All 66 tests pass, with a clean five-test BTC subset rerun. Added [B0_BTC_INSTRUCTION.md](B0_BTC_INSTRUCTION.md), config configs/baselines/btc-b0.json and reports/baselines/b0-btc/verification.json. Checkpoint and raw fitting artifacts stay under ignored runs/. No new dependencies, pretrained weights or corpus were added. Both READMEs link the new guide.

Next: specify and check TE-DS token/time/OOV contracts and pinned source compatibility, then consolidate B0 before prespecifying B1 training. Common-audio admission beyond GuitarSet remains pending and must be addressed alongside model bring-up; current feature-only Billboard/JAAH inputs are not silently treated as CQT audio corpora. No user Kaggle action is needed for this local compatibility step.
