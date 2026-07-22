# M1 baseline inputs

`corpus/` is the tracked, model-specific input corpus frozen by decision 0002. It contains
source records, exact checkpoint-template renders, little-endian `u32` token IDs, and a
SHA-256 manifest for the 512/1K/4K/8K/16K/32K contexts. These files are benchmark inputs, not
performance results.

Model-free verification:

```sh
scripts/verify-m1-corpus.sh
scripts/test-m1-corpus.sh
```

Reproducibility verification against the pinned tokenizers and local model artifacts:

```sh
scripts/build-m1-corpus.sh --check
```

`--write` intentionally regenerates the tracked corpus and is only valid before a corpus
version is accepted. Raw M1 traces belong under `benchmarks/raw/m1/` and remain ignored.
