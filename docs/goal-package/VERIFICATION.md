# VERIFICATION — adversarial review pass (2026-07-21)

> Historical scope: this verifies the v0.1/v0.2 Gemma-only package. ADR 0006 supersedes its
> forward scope and milestone order. Its arithmetic/evidence remains valid only for the named
> Gemma configurations until the dual-family phases receive their own review record.

An adversarial reviewer (Opus 4.8 subagent) re-derived every number, cross-checked all files
against the CONFIRMED reference sheets, and hunted for contradictions, overclaims, and
executability blockers. **Verdict: executable as-is by a competent coding agent** — Gemma-4
geometry (all fields), the milestone DAG (no forward dependency; E1's lossy comparator is
correctly built in M2), and the serving/cache/T0-dodge designs are faithful and coherent.

## Checks that PASSED (no change)

Global KV 8 KiB/token and 0.5/1.05/2.1 GB @ 64/128/256K; weights 6.7 GB@4.5b / 7.5 GB@5.0b;
decode ceiling 153÷7 ≈ 22 tok/s; budget @32K < 12.06 GB; MTP economics (α·γ ⇒ 2.3–3.05
tok/verify, +40–80% at ≤1.6× verify cost); global-vector ineligibility (correctly falls
back); ALL model geometry vs config ground truth (zero violations); ALL MEASURED history
numbers internally consistent (21.874 GB, p99 510.9/p50 82.4, XR86 +19.969%/0.706/85.76%,
T0 79.077%/−81.6%/2048-exact, ceiling 12,713,115,648 B).

## Findings fixed

| # | Sev | File | Issue → fix |
|---|---|---|---|
| 1 | HIGH | 05 | Governor `attention_transient` summed Σ over all 48 layers; attention evals sequentially (one score tensor live) → **max-over-layer-types**, not Σ. (As written it predicted ~17 GB vs ~7.5 GB actual and would false-reject feasible prefill — a form error no calibration constant fixes.) |
| 2 | HIGH | 08, 10 | G1 said "bit-exact logit frames" for kernel changes; flash kernels reorder FP reductions so bit-exact logits are physically impossible → G1 split three ways: **exec-model/cache/MTP = bit-exact tokens+logits; kernels = tokens-exact + logit within fault-boundary threshold; lossy quant = top-1 + KL.** |
| 3 | MED | 00, 04 | bf16 12B (~24 GB) can't be resident on 16 GB → marked **off-device / streaming-only**; on-device oracle uses the SAME Q4 both sides. |
| 4 | MED | 03, 10 | M2's 16K-peak gate was framed as needing K2 (M4) → clarified the **execution-model change (chunked prefill + no-concat-grow KV) carries it at M2**; K2 removes the residual ctx factor at M4. |
| 5 | MED | 05, 10 | O(1) local ring at capacity 1024 with no slack lets rejected MTP drafts overwrite in-window KV → **ring capacity = 1024 + γ_max(8)**, reads exclude the speculative region. |
| 6 | MED-LOW | 05 | At v0.1 the base 1024-token ring was corrected from 0.67 GB to 40×8×256×2×2×1024 = **0.335 GB** (snapshot image ≤0.34 GB). The later +8 speculative slack is not part of this historical figure; active `05` accounts for the full 1032-token allocation at 0.338 GB. |
| 7 | LOW | 03, 10 | K1 titled "decode kernel (q_len=1)"; MTP verify is q_len=γ and global exits the vector path at q_len≥3 → K1 **covers q_len 1..γ_max**. |
| 8 | LOW | 03 | "halving global-KV bandwidth" (K1) stated as payoff → relabeled: **CONFIRMED 37.5% storage reduction (arXiv); bandwidth win is ASPIRATIONAL, G2-gated.** |

Biggest residual risk called out by the reviewer and now mitigated in-spec: the governor Σ
form (finding 1). No BLOCKER-severity findings remained after the pass.

## v0.2 — NunSpark incorporation (2026-07-21)

After v0.1 verification, an exploration of NunSpark (independent MLX/Apple-Silicon engine)
surfaced evidence that **corrected one v0.1 invariant** and added several mechanisms. Folded in
(see `references/nunspark-adoptions.md` for measured evidence + landing sites):

- **A1 (corrects v0.1):** the v0.1 exactness gate — "MTP-on greedy == MTP-off greedy,
  token-identical, auto-disable on first divergence" (old 07/08/AGENTS.md) — was **unachievable**
  at fp16/bf16 scale (NunSpark MEASURED ~1 near-tie flip/110 tokens from verify-pass shape;
  auto-disable-on-divergence is self-defeating). Replaced with a target-verified three-tier G1 +
  a `near_tie_events` counter (07, 08, 10-M2/M7, 11-R11).
- **A2** append-only KV rollback (never trim a rotated ring) — 05, 07, 10-M7, 11-R12.
- **A3** masks built per layer-kind — 02, 10-M2.
- **A4** governor targets the throughput optimum, not the OOM ceiling; M1 budget sweep — 05,
  10-M1, 11-R13.
- **A5** spec OFF for MoE — 07, 10-M9, 11-R14.
- **B1** n-gram adaptive drafter lane — 06/07, 10-M5. **B3** confirmed drafter details — 04/07,
  10-M7. **B4** prefix rotation safety — 05, 10-M6. **B5** corrected tree-spec reasoning — 07.
- **M9** new owner-gated post-v1 milestone: 26B-A4B on 16GB via expert streaming (O-8 upgraded,
  O-10 added).

No BLOCKER introduced; the A1 change is a correction to an over-strong v0.1 gate, not a new
defect. The v0.2 spec is internally consistent (near_tie counter referenced by 07/08/10/11;
M9 referenced by 07/10/INDEX; A-labels cross-linked to the adoptions sheet).
