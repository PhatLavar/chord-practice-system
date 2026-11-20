# Pop/rock matching-audio status

Checked 20 November 2026. The user confirms they do not currently have audio for Isophonics, USPOP or Robbie Williams. These corpora have not been admitted for training.

| Collection | Available source evidence | Missing dependency |
|---|---|---|
| Isophonics | Official annotation pages and reference CD issue numbers | Matching recordings and edition/alignment checks |
| USPOP | Provider publishes 195 song annotations and a song list; explicitly no audio | Acquire corresponding audio; resolve exact versions |
| Robbie Williams | Research reproduction page links annotations | Matching recordings and annotation-version pairing |
| RWC-Pop | Provider covers 100 tracks; audio access is handled by RWC | Obtain audio through the provider and verify pairing |

Sources: [Isophonics](https://isophonics.net/content/reference-annotations.html), [USPOP/RWC annotation provider](https://github.com/tmc323/Chord-Annotations), [Korzeniowski/Widmer reproduction page](https://fdlm.github.io/post/auditorymodel/).

No purchase or external contact was made. Do not substitute arbitrary streaming recordings without checking timing/version correspondence. Annotation song counts are not counts of locally usable training examples. GuitarSet is the immediate accessible-audio pilot; it does not reproduce the pop/rock benchmark.

The reproduction page also warns that its released madmom model was trained on several of these corpora including Billboard. Evaluate pretrained baselines only after checking training-set overlap; otherwise report them as potentially contaminated references, not clean held-out competitors.
