# SycoScope v2

Linear probes for sycophancy in LLMs. Contrastive probes (one per taxonomy cell, trained on system-prompt pairs) are compared with a universal probe and with skyline probes (trained on benchmark labels), scored on out-of-distribution benchmarks.

This repo is a clean rebuild of `github.com/oscaryas/SycoScope` (the "old repo", read-only reference; `main` at commit `4132611`). Code is ported from the old repo one component at a time and verified against its results.

## Read first

- `docs/SPEC.md`: the research spec. Scope is the spec plus skyline cross-generalization.

Current step: **Step 1 only** (probe training). Do not start Step 2 or 3 until asked.

## Principles

**Precision over robustness.** When something is wrong, stop. A crash is better than a wrong number.

- Errors propagate. No `except Exception`, no silent fallbacks, no placeholder values such as `"unknown"`.
- Every stage reports rows in, rows excluded (with reason) and rows out. Counts must add up.
- No guessed defaults for anything that affects results (layer, position, seed, rubric, split). Missing fields raise.
- Undefined metrics are reported as undefined, never filled in.
- Assert invariants where a mistake would pass silently: index and array lengths match, no training ID in evaluation, both classes present, activation model matches probe model.

**Lightweight code.**

- Plain functions and plain data: dicts in JSONL, numpy arrays in `.npz`. No class hierarchies, registries, plugin systems or config frameworks.
- One implementation per job. Move code to a shared module only when a second real caller needs it.
- Each stage is a script with `argparse` that reads files and writes files, readable top to bottom: load, compute, save.
- A new research variant should need a new input file or flag, not a new layer of code.

**Comments.**

- State what the code does, briefly and precisely.
- A one-line reason is fine where code would otherwise look wrong or removable, e.g. `add_special_tokens=False  # chat template already includes BOS`.
- No citations, research notes, history or long module docstrings in code. Those go in `docs/`.

**Tests.**

- Only where a mistake gives wrong numbers silently. Small `pytest` functions, most needing only a tokenizer.
- When a test fails, fix the code.

## Working rules

- Port from the old repo's best existing version of each piece; do not rewrite from memory.
- Anything that produces numbers gets a parity check against the old repo before it is used.
- Every activation file gets a sibling `*.meta.json`: input paths with SHA-256, git commit (a dirty tree is recorded as dirty), command-line arguments, row counts.
- Generations and judged data are committed (Git LFS for large files). Activations are never committed.
- Model runs (generation, activation extraction) need a GPU machine; unit tests must run on CPU.
