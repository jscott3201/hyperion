# Hyperion evaluation ledger

This file is append-only once rows are accepted. Raw outputs are stored under
`eval-results/raw/` and ignored. Agent-eval floors begin at M5; M0 contains only the schema so
no fixture can masquerade as a real evaluation.

Every future row must include classification, immutable case/corpus hashes, model and quant
identity, engine commit, exact command, safe machine state, trial policy, full metrics, and a
gate verdict.
