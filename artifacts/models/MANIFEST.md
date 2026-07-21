# Model artifact manifest

Tensor payloads under this directory are gitignored. This tracked manifest is the source of
truth for identity, acquisition status, license review, local hashes, and milestone ownership.

| Milestone | Repository | Immutable revision | Local status |
|---|---|---|---|
| M0 | `google/gemma-4-12B-it-qat-q4_0-unquantized` | `b6ed86275a6a5735884e208bfed95b445a684ca2` | verified, converted, real generation passed |
| M2 | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | deferred; BF16 spot checks only |
| M1 | `google/gemma-4-E4B-it-qat-q4_0-unquantized` | `476025a01dbf99361c062bbeca3d6a76bb4c4566` | deferred to M1 baseline acquisition |
| M7 | `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` | `18934064dd4c5c6cc3621f6381e7d377fc8cb7bd` | deferred for disk budget |
| M8 | `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` | `27f8d204f09f2be353d6ff0bf0d012792b13c79f` | deferred for disk budget |

## M0 primary artifact

- Source directory: `artifacts/models/gemma4-12b-qat-source/`
- Converted directory: `artifacts/models/gemma4-12b-qat-mlx-g64-b4/`
- Conversion identity: affine Q4, group size 64, 4 bits, using the locked oracle environment
- Source dry-run payload: 23.9 GB `model.safetensors` plus tokenizer/config metadata
- Source verification: `hf cache verify --fail-on-missing-files` checked all nine remote files;
  an independent root allowlist rejected unexpected non-cache files
- Source SHA-256 manifest digest:
  `6a07a92df9260b71117b113a8ad0b305432a48f895abd850a7616241a636ebed`
- Source weight SHA-256:
  `26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0`
- Source config/tokenizer SHA-256: `a323d02f68420f6fa3a3548130a0d36356075a4047a622e57148558f8eee7077`,
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`
- Source chat-template SHA-256:
  `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`
- Converted SHA-256 manifest digest:
  `9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144`
- Converted shard SHA-256:
  `318f06775a7c234e0c31c1f9971a38b6c3217d5c5afe2be8a8286fdfe4015dd9`,
  `755c80994e9c8dc7c9491d5d01c1472c152da3055ce9fd35832f0c2c12f3c39f`
- Converted config/tokenizer SHA-256:
  `257501c3412dd0c5645c56a47b6c5752fbc416c586d97534bca696668644b7b0`,
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`
- Converted chat-template SHA-256:
  `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`
- Conversion reported `4.501` bits/weight; converted payload is 6.3 GiB on disk
- Strict smoke: thinking disabled through the checkpoint chat template, greedy seed 0, response
  exactly `HYPERION_M0_OK`
- License/frontmatter review: local model-card frontmatter declares `apache-2.0` and links the
  Gemma 4 license; final source hash binds that reviewed card to the snapshot

The four deferred repositories remain required. Decision 0001 schedules them at the first
milestone that consumes each artifact because downloading all five plus conversion output at
session zero would leave inadequate local working space. Official Hub metadata for all five
reviewed revisions declares `apache-2.0`; each local model card is rechecked when acquired.
