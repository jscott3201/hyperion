# ADR 0005: public source under MIT OR Apache-2.0

- **Status:** accepted
- **Date:** 2026-07-31
- **Owners:** Hyperion team
- **Supersedes:** O-2's private-repository default
- **Related:** `Cargo.toml` workspace package metadata; `artifacts/models/MANIFEST.md`

## Context

The workspace has declared `MIT OR Apache-2.0` since M0, but the repository did not contain
the corresponding root license texts and O-2 still described the repository as private. That
was incomplete for a public release and left the boundary between project source, model
artifacts, and third-party material implicit.

The owner has chosen to open-source Hyperion. The license should be familiar to Rust users,
compatible with the existing Cargo metadata, and explicit about which material the project
can actually license.

## Decision

Unless a file says otherwise, first-party source code and documentation owned by Justin Scott
are offered under either of these licenses, at the recipient's option. This includes material
reused or adapted from his earlier Helios and mlx-bonsai prototypes:

- the MIT License in `LICENSE-MIT`; or
- the Apache License, Version 2.0, in `LICENSE-APACHE`.

The MIT notice names Justin Scott as the 2026 copyright holder.

The SPDX expression remains `MIT OR Apache-2.0` in the workspace package metadata, and every
workspace crate continues to inherit it.

This project license does **not** relicense model checkpoints, converted weight artifacts,
third-party dependencies, evaluation data, or future imported source. Those materials retain
their own terms. Model payloads remain gitignored; the tracked artifact manifest records the
reviewed license frontmatter and immutable revision for each checkpoint. Four committed
real-model-derived oracle fixtures are Apache-2.0-only and identified in `PROVENANCE.md` and
their directory-level notice. Any future source-level adoption must preserve applicable
copyright, license, attribution, and NOTICE material before it is merged or redistributed.

## Consequences

- The public README links both complete license texts and states the model/dependency boundary.
- O-2 is closed as a ruled owner decision: public source, dual licensed at the user's option.
- No project `NOTICE` file is created now because neither predecessor is third-party material
  and the reviewed model snapshot does not include an upstream `NOTICE`. The narrower fixture
  terms and attribution are recorded in `PROVENANCE.md` and the fixture-directory sidecar.
  If imported NOTICE content appears later, notice handling becomes part of that change's gate.
- Binary or packaged distributions need a fresh third-party license and notice audit; this
  ADR covers the repository's project-authored source and documentation, not a future bundle.

## What this does not decide

- Whether future releases use a DCO, CLA, or another contribution-attestation mechanism.
- Whether any model checkpoint may be redistributed with a Hyperion release.
- Whether a future file must be Apache-2.0-only because of incorporated upstream material.
