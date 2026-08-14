# 09 — Agent eval (re-domained to the actual workload)

The task/grader layer is shared, but each artifact runs through its exact conversation profile
and receives a separate frozen baseline/floor. Cross-family aggregate scores never substitute
for tokenizer/template/tool fidelity or family-specific regression rows.

## What carries from mlx-bonsai (mechanism ≈ verbatim)

- The **turn-loop runner**: streaming chat request → assemble tool calls → execute in
  sandbox → append the selected conversation profile's rendered tool result → repeat (≤10 turns)
  → grade. Zero-dependency Rust; no Gemma marker is embedded in the shared runner.
- **Sandbox safety**: fixed path, delete+recopy reset, path-escape rejection (abs/`..`/
  backslash/NUL/symlink), protected grader surfaces restored before every grade.
  **Upgrade (closing bonsai's admitted gap):** network isolation enforced, not
  conventional — run graders under `sandbox-exec`/no-network profile; a task that attempts
  network access fails loudly.
- **Determinism**: canonicalized tool IDs/results, FNV-1a-64 transcript fingerprint,
  k=2 byte-identical repeat gate (greedy arm), `PYTHONHASHSEED=0` where Python graders run.
- **Honesty split**: raw-model fidelity vs server-repaired outcome reported separately
  (`wellformed`, `repaired`, `expected_tool_share`, `parsed_call`, per-tool arg-shape
  validation). Server repair is allowed in production and measured in eval — never
  conflated (the bonsai "93.06% is a repaired upper bound" discipline).
- Multi-arm support (`HYPERION_AGENT_EVAL_ARM=`): native server vs mlx-lm server vs (later)
  hosted tuples — same tasks, same grader, engine-identity recorded per row.

## What changes: the task suite (coding suite → smart-buildings suite)

Tool surface (replaces read_file/write_file/run_tests; schemas frozen as fixtures):

| Tool | Shape (summary) |
|---|---|
| `get_points(filter)` | query normalized point list (id, label, kind, unit, tags) |
| `get_timeseries(point_ids, window, agg)` | bounded series slices (synthetic fixtures) |
| `propose_point_tags(point_id, tags[], confidence)` | W5 tagging proposal (pointforge shape) |
| `submit_fault_finding(equip_id, rule_id, evidence{...})` | W1 FDD explain output |
| `submit_recommendation(action, rationale, savings_estimate)` | W2 output |
| `set_setpoint(point_id, value, revert_after)` | bounded-write action with guard args |
| `lookup_rule(rule_id)` | FDD reference retrieval (64-rule corpus subset as fixtures) |

Task families (target ~26 tasks, mirroring bonsai's ladder shape: t=toy, s=single-tool,
r=multi-step, x=hard/compound):

- **t0x smoke (3):** single well-formed call, echo-shaped.
- **s0x single-tool (6):** clamp/screen-out analog → e.g., propose tags for one point with
  vocab in-context; reject-on-insufficient-evidence (NO_EVAL discipline — the model must
  refuse, mirroring pointforge's closed refusal enum).
- **r0x multi-step (8):** get_points → get_timeseries → submit_fault_finding chains
  (economizer stuck-damper, simultaneous heat/cool, sensor drift — drawn from the FDD
  reference's AHU/RTU rules); tool_response digestion; thinking-mode ON variants.
- **x0x compound (6):** cross-equipment reasoning, conflicting evidence, required
  parallel calls (two independent lookups in one turn), long-transcript continuation (grader
  checks the conversation-cache path produces identical outcomes to cold path).
- **adv (3):** prompt-injection in tool_response content (must not execute embedded
  "instructions"), oversized args, duplicate-call bait (dedupe telemetry asserted).

Graders are deterministic Rust/JSON-schema checks + golden expected-call sets (no pytest,
no LLM judge). Rubric stays 0–5 ordinal per task. Each artifact/profile freezes its own G3 floor
from a preregistered baseline and permits zero regressions on previously green tasks. The former
24/26 proposal is not a cross-family gate and cannot become a floor without accepted measurement.

## Quality gates beyond agentic

- **gemma-challenge/eval-prompts** (128 MMLU-Pro/GPQA-Diamond/AIME26 prompts, harness
  format): retained as the Gemma kernel-CI quality canary (the logit-saturation catch), not as a
  Qwen acceptance substitute. License note:
  keep the prompt payload out of the public repository until its redistribution terms have
  been independently verified; Hyperion's repository license does not relicense external data.
- **Thinking-mode A/B:** suite runs thinking-on vs thinking-off arms; ledger records
  quality-vs-token-budget curves (feeds deployment profiles' budget defaults).
- A 12B-versus-E4B capability delta remains an optional legacy Gemma-family comparison if that
  tier is reopened; it is not a P0–P7 dependency.

## Relationship to project-level evals

Engine-repo eval answers "does the ENGINE serve the model faithfully and fast" (fidelity,
latency, memory, determinism). Product-level model-capability admission lives in its own
project per the owner's scoping — this suite feeds it evidence rows but does not replace it.
