# BTC source provenance

Source: [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19), commit [2682317be668032e6e4b269ded36adaa2ad57df0](https://github.com/jayg996/BTC-ISMIR19/commit/2682317be668032e6e4b269ded36adaa2ad57df0).

Paper: Jonggwon Park, Kyoyun Choi, Sungwook Jeon, Dokyun Kim and Jonghun Park, [A Bi-Directional Transformer for Musical Chord Recognition](https://zenodo.org/records/3527886), ISMIR 2019. Use the upstream paper citation when reporting results. Source is MIT licensed, Copyright (c) 2019 Jonggwon Park; the complete notice is preserved in LICENSE.

Only btc_model.py and utils/transformer_modules.py are included. UPSTREAM.json records upstream/vendored hashes, the pinned revision, reference-file hashes and compatibility changes. File hashes for vendored sources normalize CRLF to LF so Windows checkout conversion does not create a false source change.

Changes: package-relative imports; remove unused HParams import and standalone demonstration from btc_model.py; replace the removed NumPy np.float alias with equivalent np.float64 in timing generation. Model class ASTs match the pinned upstream classes exactly. The author's sample-standard-deviation LayerNorm, feed-forward final ReLU/dropout, attention masking and unused output-layer LSTM parameters are deliberately preserved. Those choices must not be silently modernized while claiming the same model.

Project adapters, masked-target loss and cache/window inference live outside this directory. No dataset readers, third-party pretrained weights or original audio were copied. This source pin does not make our GuitarSet front end a reproduction of the original published data/preprocessing protocol.
