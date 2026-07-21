# Model artifact manifest

Tensor payloads under this directory are gitignored. This tracked manifest is the source of
truth for identity, acquisition status, license review, local hashes, and milestone ownership.

| Milestone | Repository | Immutable revision | Local status |
|---|---|---|---|
| M0 | `google/gemma-4-12B-it-qat-q4_0-unquantized` | `b6ed86275a6a5735884e208bfed95b445a684ca2` | acquisition/verification in progress |
| M2 | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | deferred; BF16 spot checks only |
| M8 | `google/gemma-4-E4B-it-qat-q4_0-unquantized` | `476025a01dbf99361c062bbeca3d6a76bb4c4566` | deferred for disk budget |
| M7 | `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` | `18934064dd4c5c6cc3621f6381e7d377fc8cb7bd` | deferred for disk budget |
| M8 | `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` | `27f8d204f09f2be353d6ff0bf0d012792b13c79f` | deferred for disk budget |

## M0 primary artifact

- Source directory: `artifacts/models/gemma4-12b-qat-source/`
- Converted directory: `artifacts/models/gemma4-12b-qat-mlx-g64-b4/`
- Conversion identity: affine Q4, group size 64, 4 bits, using the locked oracle environment
- Source dry-run payload: 23.9 GB `model.safetensors` plus tokenizer/config metadata
- Verification: `hf cache verify` plus a local SHA-256 manifest; final hashes are appended only
  after acquisition and conversion complete
- License/frontmatter review: local model-card frontmatter declares `apache-2.0` and links the
  Gemma 4 license; final source hash binds that reviewed card to the snapshot

The four deferred repositories remain required. Decision 0001 schedules them at the first
milestone that consumes each artifact because downloading all five plus conversion output at
session zero would leave inadequate local working space. Official Hub metadata for all five
reviewed revisions declares `apache-2.0`; each local model card is rechecked when acquired.
