# Self-prompt: HNSW implementation kickoff

**Purpose:** This file is the prompt to paste at the start of a new Claude
Code session when picking up HNSW implementation work. It's self-contained —
assumes no memory of prior sessions. Read top to bottom, then start.

---

## You are picking up HNSW work for `tokenfs-algos` v0.7.0

You are working in `/home/mjbommar/projects/personal/tokenfs-algos`. The
crate is at v0.5.0 (audit-R10 closed, iai-callgrind regression gate live,
gh-pages bench-history publishing at `mjbommar.github.io/tokenfs-algos/`).
The HNSW landing is the next major work item; it spans 7 weeks across
5 phases per [`docs/hnsw/`](.).

**Before doing anything else, read in this order:**

1. [`docs/hnsw/README.md`](README.md) — index + headline findings from
   the research deep-dives.
2. [`docs/HNSW_PATH_DECISION.md`](../HNSW_PATH_DECISION.md) — the parent
   decision: fully native walker AND builder, usearch v2.25 wire format,
   no libusearch dependency in this crate.
3. [`docs/hnsw/00_ARCHITECTURE.md`](00_ARCHITECTURE.md) — layer map,
   module tree, posture matrix.
4. The phase doc for the work you're picking up:
   [`docs/hnsw/phases/PHASE_<N>.md`](phases/) — deliverables, tests,
   demos, acceptance criteria.

Don't start coding until you've read 1–4. They prevent re-litigating
decisions and re-doing research.

## Find your phase

Run `git log --oneline -20` and look for the most recent commit prefixed
`feat(hnsw): phase <N>` or `release: v0.7.0-phase<N>`. Phase numbering:

- **Phase 1** — wire format + scalar walker skeleton (week 1)
- **Phase 2** — SIMD distance kernels (weeks 2-3)
- **Phase 3** — filter primitives + AVX-512 (week 4)
- **Phase 4** — native deterministic builder (week 5)
- **Phase 5** — kernel-FPU bracketing + tokenfs_writer integration (weeks 6-7)

If no `feat(hnsw)` commits exist, you're starting Phase 1.

## Inviolable contracts

These are load-bearing for everything we shipped through audit-R10. Break
any of them and you must surface it explicitly to the user before
continuing — do NOT silently work around.

1. **`panic_surface_allowlist.txt` stays at 0 entries.** Every new public
   function follows the `try_*` / `_unchecked` / `_inner` shape from
   [`docs/KERNEL_SAFETY.md`](../KERNEL_SAFETY.md). The lint runs in
   `cargo xtask check`; failing is a hard stop.

2. **`cargo xtask check` must be green before every commit.** No
   exceptions for "I'll fix it next commit." If a fix needs more thought,
   stash and re-commit cleanly.

3. **Kernel-safe surface (`--no-default-features --features alloc`)
   continues to compile.** The `tokenfs-algos-no-std-smoke` crate's
   `smoke()` is the canary; extending it as you add new kernel-safe entry
   points is part of the deliverable, not a follow-up.

4. **Per-backend kernels follow the established `_unchecked` sibling
   pattern** from `bits::popcount`, `bits::streamvbyte`, `vector::distance`.
   Read those before writing the first HNSW kernel. Don't invent a new
   shape.

5. **iai-callgrind regression gate at 1% IR.** Every new SIMD kernel gets
   an `iai_hnsw_*` bench row in `benches/iai_primitives.rs`. The gate
   is auto-active on every push; first regression hit IS signal, not noise.

6. **Wire format is usearch v2.25, byte-for-byte.** We do NOT invent a new
   format. We do NOT extend usearch's format. We do NOT support pre-v2.25
   formats. New format versions land at a new section ID per
   `IMAGE_FORMAT_v0.3 §11` — out of scope for this v0.7.0 landing.

7. **Determinism: single-thread + sorted input + `rand_chacha::ChaCha8Rng`
   seeded from `image_salt`.** Per
   [`docs/hnsw/research/DETERMINISM.md`](research/DETERMINISM.md). Not
   negotiable for SLSA-L3.

## The work loop, per phase

For each phase, follow this loop. Don't skip steps.

### Step 1 — Re-read the phase doc + relevant research

Open [`docs/hnsw/phases/PHASE_<N>.md`](phases/) and the research files it
cites in "Required research input." Don't trust your prior reading — re-read.
The research docs are 500-1100 lines each; the phase doc tells you which
sections to actually consume.

### Step 2 — Confirm the prior phase landed

Run `git log --oneline | head -20`. If picking up Phase N, you should see
`feat(hnsw): phase <N-1>` (or `release: ... phase<N-1>`) and the prior
phase's tests should pass:

```bash
cargo test -p tokenfs-algos --features arch-pinned-kernels --test hnsw_walker_parity
cargo xtask check
```

If either fails, stop and surface the issue to the user. Don't paper over
broken state with new code.

### Step 3 — Read the component spec(s) for this phase

Each phase touches specific components in [`docs/hnsw/components/`](components/).
The component specs document API skeletons + key findings folded from research.
Treat them as the contract — implementation matches the API skeleton; if it
needs to diverge, update the spec first and surface the divergence.

### Step 4 — Sketch the implementation, get user buy-in for non-trivial choices

Before writing 500+ LOC, draft the module layout + key type signatures
in a short text response to the user. Specifically check in on:

- Any divergence from the component spec
- Any new public API surface beyond what's in the spec
- Anything the research docs flagged as "decide later" (e.g. exact tie-break
  ordering, brute-force fallback threshold, FPU-bracket granularity)
- Any pull of an outside dependency (`Cargo.toml` change)

Don't ask permission for routine implementation work — write the code.
Do check in on shape choices, dependency adds, and anything that changes
the audit posture.

### Step 5 — Implement + test in tight loops

Write code in small commits, each with passing tests:

- **Parity tests are non-negotiable** — every new kernel has scalar parity;
  every walker change has a test against the libusearch reference fixture.
  Phase docs specify the fixture path.
- **Bench rows added in the same commit as the kernel** — not deferred.
  iai-callgrind needs to start tracking from the first commit, otherwise
  the regression gate has nothing to gate against.
- **`cargo xtask check` runs before EVERY commit.** Make it muscle memory.

### Step 6 — When you hit a real blocker, stop and ask

Real blockers (vs. inconveniences):
- A research finding contradicts the component spec
- A determinism gap that can't be resolved with the rules in
  [`research/DETERMINISM.md`](research/DETERMINISM.md)
- A kernel-safety constraint that can't be satisfied without changing the
  module's posture
- A wire-format question not answered in
  [`research/USEARCH_DEEP_DIVE.md`](research/USEARCH_DEEP_DIVE.md)

For real blockers, write a short summary (what you tried, what failed,
what alternatives you see) and surface to the user. Don't grind.

### Step 7 — End of phase: demo + commit + ritual

When the phase's acceptance criteria pass:

1. Run `cargo xtask check` one more time.
2. Run the phase's parity / round-trip tests one more time.
3. Run `cargo xtask bench-iai` locally if you have valgrind; otherwise
   confirm CI's iai-bench workflow will pick it up.
4. Update [`docs/hnsw/README.md`](README.md) phase table — mark the
   phase complete with a date and commit SHA.
5. Update [`docs/HNSW_PATH_DECISION.md`](../HNSW_PATH_DECISION.md) §9
   — mark the phase row complete.
6. Commit with `feat(hnsw): phase <N> — <theme>` and a body describing
   what landed + what tests pass.
7. Push.
8. Verify the gh-pages bench-history publishes a new dot for the new
   kernels (give it ~10 min after push).

After Phase 5: full v0.7.0 release ritual per
[`phases/PHASE_5.md`](phases/PHASE_5.md) §"v0.7.0 release checklist" —
Cargo.toml bump, CHANGELOG, AGENTS.md, README, tag, push.

## Iteration patterns to expect

- **Parity test fails on a SIMD kernel.** Default first move: write a tiny
  test reproducing the failure with hand-picked inputs (not random). Add
  to `tests/parity.rs`. Then fix. Don't tune SIMD without a deterministic
  failing test.
- **iai-callgrind regression hits.** Inspect the IR delta per-bench-row.
  If the regression is real (algorithm change), accept and re-baseline. If
  it's spurious (compiler version change, target-feature flag drift),
  document in the commit message.
- **libusearch round-trip diverges in Phase 4.** Bisect: build N=10, 100,
  1000 progressively. The first divergence point usually identifies which
  algorithm step (level assignment, neighbor selection, edge pruning) is
  off. The HNSW paper is precise enough that a correct implementation
  should match libusearch on deterministic input; divergences are bugs.
- **Cross-arch f32 results differ.** Expected. Document the divergence;
  constrain SLSA-L3 builds to integer metrics per
  [`research/DETERMINISM.md`](research/DETERMINISM.md) §9.
- **A primitive you wanted to reuse turns out to need extension.** Extend
  the underlying primitive (`vector::*`, `bits::popcount`, etc.) in a
  separate commit, with its own audit-R10 discipline. THEN use it from
  HNSW. Don't fork the primitive.

## Things you should NOT do

- **Do not pull `usearch` (the C++ library) as a dependency.** Anywhere.
  Decision in `HNSW_PATH_DECISION.md` §2; do not re-litigate.
- **Do not extend the wire format.** New format = new section ID, out of
  scope for v0.7.0.
- **Do not add multi-threaded build to the v0.7.0 builder.** Single-thread
  is the SLSA-L3 contract; `parallel` feature defers to v0.8+.
- **Do not write Rust code with `unsafe fn` accessors lacking bounds
  checks** unless they're the established `_unchecked` sibling of an
  asserting variant. The CVE-2023-37365 risk row exists for this exact
  reason.
- **Do not design for hypothetical future scalar types** (f16, bf16, e5m2,
  e4m3, e3m2, e2m3). v1 is f32 / i8 / u8 / binary; defer the rest until a
  consumer asks.
- **Do not silently ship f32 metrics in the kernel-safe path** without
  the FPU bracketing macro from Phase 5. Fail closed.
- **Do not delegate the "decide what's reusable" question.** The primitive
  inventory at [`research/PRIMITIVE_INVENTORY.md`](research/PRIMITIVE_INVENTORY.md)
  marked every primitive as Direct reuse / Pattern reuse / Inspiration.
  Trust those classifications; if you find one wrong, update the inventory.
- **Do not skip the no_std smoke crate update.** Every new kernel-safe
  entry point gets a call from `tokenfs-algos-no-std-smoke::smoke()` in
  the same commit.

## Cross-cutting reminders

- **AGENTS.md** ([`../../AGENTS.md`](../../AGENTS.md)) — agent guidance for
  this codebase. Particularly the sed/perl preview discipline; bulk regex
  substitutions in this codebase are dangerous (docstrings, string
  literals, adjacent `try_*` calls). Use `grep -n` first, prefer surgical
  Edit calls.
- **Saved feedback memories** — auto-loaded into context. They include the
  `_inner` helper extraction pattern and the "don't snapshot HIGH-severity
  audit gaps" rule. Both apply directly to HNSW work.
- **`docs/PRIMITIVE_CONTRACTS.md`** — every primitive must clear the
  Queue-Pruning Gate (4 questions). HNSW already cleared it in
  `HNSW_PATH_DECISION.md` §3 — don't re-derive.
- **`docs/PROCESSOR_AWARE_DISPATCH.md`** — kernel buffet pattern + planner
  integration. HNSW kernels plug in via the same shape.

## When in doubt

When in doubt about an algorithmic question, the order of consultation is:

1. The HNSW paper (Malkov & Yashunin 2018), cited in
   [`research/HNSW_ALGORITHM_NOTES.md`](research/HNSW_ALGORITHM_NOTES.md).
2. usearch source at `_references/usearch/include/usearch/`, with line
   numbers cited in
   [`research/USEARCH_DEEP_DIVE.md`](research/USEARCH_DEEP_DIVE.md).
3. NumKong source at `_references/NumKong/` for SIMD kernel patterns
   (per [`research/SIMD_PRIOR_ART.md`](research/SIMD_PRIOR_ART.md)).
4. The user. Only after 1-3 fail to answer.

When in doubt about a Rust API shape, the order of consultation is:

1. The component spec in [`components/`](components/).
2. The pattern reference in
   [`research/PRIMITIVE_INVENTORY.md`](research/PRIMITIVE_INVENTORY.md)
   §10 (the canonical `bits::popcount` pattern).
3. [`docs/KERNEL_SAFETY.md`](../KERNEL_SAFETY.md) for the
   `try_*` / `_unchecked` / `_inner` shape.
4. The user.

## End-of-session ritual

Before ending a session:

1. Commit any in-progress work, even if not at a phase boundary. Mark
   commits `wip(hnsw): ...` so they're identifiable.
2. If a phase landed, push.
3. Update [`docs/hnsw/README.md`](README.md) phase table with current state.
4. Tell the user: which phase is in progress, what's left in it, and
   what the next session should pick up first.

## A note on the user

The user (Michael Bommarito) prefers terse, surgical responses. They have
deep Rust expertise, deep systems-programming context, and have been doing
this codebase for several months. Don't over-explain audit-R10 patterns;
they helped design them. Do explain non-obvious algorithmic choices, since
HNSW has subtleties that aren't load-bearing in the rest of the crate.

The user pushed back hard during audit-R10 about partial work being
shipped under a "we'll defer the rest" frame. That precedent applies to
HNSW phases: a phase is "complete" when its acceptance criteria pass on
green CI, not when the happy-path test passes locally.

The user is the final word on scope. If you want to expand a phase's
scope (add a metric, add a backend), surface and ask. Don't expand
unilaterally.

---

**Now: read the four docs at the top of this file, find your phase, and
start the work loop.**
