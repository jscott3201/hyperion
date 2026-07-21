# Model artifact manifest

Tensor payloads under this directory are gitignored. This tracked manifest is the source of
truth for identity, acquisition status, license review, local hashes, and milestone ownership.

| Milestone | Repository | Immutable revision | Local status |
|---|---|---|---|
| M0 | `google/gemma-4-12B-it-qat-q4_0-unquantized` | `b6ed86275a6a5735884e208bfed95b445a684ca2` | verified, converted, real generation passed |
| M2 | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | deferred; BF16 spot checks only |
| M1 | `google/gemma-4-E4B-it-qat-q4_0-unquantized` | `476025a01dbf99361c062bbeca3d6a76bb4c4566` | verified and converted for the M1 stock baseline |
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

## M1 E4B baseline artifact

- Source directory: `artifacts/models/gemma4-e4b-qat-source/`
- Converted directory: `artifacts/models/gemma4-e4b-qat-mlx-g64-b4/`
- Conversion identity: affine Q4, group size 64, 4 bits, using the same locked oracle as the
  M0 primary conversion; converter reported `4.501` bits/weight
- Source repository/revision: `google/gemma-4-E4B-it-qat-q4_0-unquantized` at
  `476025a01dbf99361c062bbeca3d6a76bb4c4566`
- Source verification: all nine remote files verified; an independent root allowlist rejects
  unexpected non-cache files
- Source SHA-256 manifest digest:
  `d86886b83233724bbfd0bbf6f033bcd7b2862c16206f62b6d7c01826fef44c95`
- Source weight SHA-256:
  `ad8ef515194b15ab13b6f98d54d37652b691c9a2c787cdf7ed5f951f5ed2c7fa`
- Source config/tokenizer/template SHA-256:
  `f2db6d6e24cb4b695308897d82c8424fae946e95272a27de68b6e908c77b0f95`,
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`,
  `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`
- Converted SHA-256 manifest digest:
  `9ba65423d3b2bab1e7c52ea88a1a2b0a33c1f51909b1df66330bf872b7a6c2b0`
- Converted model/index SHA-256:
  `58612593e62f7488c7fc467b7c7248d67f063f0f5af33da02b3bcd64df876147`,
  `4fc339ac8c0e2b0846fe9b20ee8879767dd6fecd6ca4b48372173ef4e67163f3`
- Converted config/tokenizer/template SHA-256:
  `4ef2407f45352f03b2a0cd959c07956e5db7cefdc2202272b25ad74011892e93`,
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`,
  `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`
- Geometry bound by the verifier: 42 layers (35 sliding / 7 full), hidden 2560, 8 Q / 2 KV
  heads, local/global head dimensions 256/512, sliding window 512, 18 shared-KV layers,
  256-dim PLE, K≠V, 128K context, and vocab 262144
- License/frontmatter review: the pinned local model card declares `apache-2.0`; its hash is
  part of the source manifest

The three deferred repositories remain required. Decision 0001 schedules them at the first
milestone that consumes each artifact because downloading all five plus conversion output at
session zero would leave inadequate local working space. Official Hub metadata for all five
reviewed revisions declares `apache-2.0`; each local model card is rechecked when acquired.
