# Autoresearch Ideas Queue

## Prerequisites

- Build a low-overhead hot-path profiler before selecting another speed
  candidate. This is infrastructure for choosing ideas, not itself an
  optimization idea. The profiler should be disabled by default and should not
  be judged by Modal speedup unless it accidentally changes the normal hot path.
- Profiler evidence now exists. Policy-completion profile:
  `artifacts/benchmarks/policy-profile-level1-1-native-maxpool-levelchange-strict-20260701-224421.json`.
  Canonical local validation profile:
  `artifacts/benchmarks/local-profile-validation-20260701-224610.json`.
  These are local diagnostic artifacts only; they rank candidate mechanisms but
  do not establish accepted speed wins.

## Ready

Curated order: highest estimated ROI first. ROI is judged from local profiler
evidence, expected Modal signal, implementation size, correctness risk, and the
existing reject/keep ledger. Local profiles rank candidates only; Modal remains
the acceptance source of truth.

### IDEA-20260701-002: Fast-Forward SMB Sprite-0 Polling

- Status: ready
- Perspective: emulator-core
- Estimated ROI: highest. Hot in both policy-completion and grouped-lane
  profiles, narrow PC/routine target, and likely enough CPU-step removal to show
  through Modal variance.
- Hypothesis: SMB has a known sprite-0 wait loop that repeatedly reads
  `PPUSTATUS` until the sprite-0 bit is set. The emulator already has a coarse
  PPU event model and a committed SMB idle-jump fast-forward; a ROM-signature
  guarded sprite-0 polling fast-forward could remove many interpreter dispatches
  while preserving frame timing.
- Target files: `src/emulator.rs`
- Prior evidence: `src/emulator.rs` models `PPU_SPRITE0_DOT` and has
  `SMB_IDLE_JMP_PC` fast-forward support, but no committed sprite-0 poll
  fast-forward. Prior ROM analysis identified the loop around `$8150`. The
  2026-07-01 profiler now makes this the highest-priority candidate: policy
  completion top exact PCs were `$8150`, `$8153`, and `$8155`, and the hottest
  16-byte range was `$8150-$815F`; the canonical local grouped-lane profile
  showed the same hottest exact PCs and range.
- Plan: Guard on exact PRG bytes and current PC. Only fast-forward when the
  sprite-0 status bit is clear. Advance pending PPU cycles to the next event,
  update CPU cycle guard, and preserve `PPUSTATUS` read side effects including
  clearing vblank and resetting the first-write latch.
- Contract risks: NMI timing drift, incorrect `A`/ZN flags after the skipped
  reads, or missing `PPUSTATUS` side effects.
- Required checks: Add a targeted regression for the guarded loop if practical,
  then run `cargo fmt --check`, `cargo check --release`,
  `.venv/bin/python -m maturin develop --release`, `make test`, and the Modal
  paired benchmark before accepting.
- Expected benchmark signal: Candidate should show a robust paired speedup if
  this loop is hot in the first four level states; rough expectation `+3%` to
  `+12%`.

### IDEA-20260701-004: Specialize Controller Polling Loop Safely

- Status: ready
- Perspective: emulator-core
- Estimated ROI: high. Smaller expected upside than sprite-0, but the hot
  range is narrow and the emulator already stores controller state compactly.
- Hypothesis: SMB polls controller state through repeated `$4016` reads in NMI.
  For this emulator, controller state is already a compact byte. A guarded
  routine-level specialization could materialize the same input result with far
  fewer interpreted instructions.
- Target files: `src/emulator.rs`
- Prior evidence: The core already implements serial controller reads, and the
  benchmark action set uses deterministic per-frame button bytes. Prior ROM
  analysis identified controller polling around `$8E5C`. The 2026-07-01
  profiler strengthens this: `$8E70-$8E7F` was the third-hottest 16-byte range
  in the policy-completion profile and was also hot in the canonical local
  grouped-lane profile.
- Plan: Use profiling first. If the poll loop is hot, add a narrowly guarded
  fast path for the standard SMB controller routine that preserves RAM outputs,
  accumulator flags, X/Y effects, and strobe semantics. Avoid changing the
  public action mapping.
- Contract risks: Very high if edge-triggered button handling, two-player reads,
  or A/B/start filtering differs from the ROM routine. Reject on any parity
  mismatch.
- Required checks: Add a focused controller-routine equivalence test or a
  step-sequence parity check with `noop`, `right`, `right_a`, and `right_a_b`,
  then run required checks and paired Modal benchmark.
- Expected benchmark signal: Rough expectation `+1%` to `+5%`; the value is
  mainly compounding with other PC-specific fast-forwards.

### IDEA-20260701-005: Cache Sprite Overlay Background Priority Data

- Status: ready
- Perspective: ppu-render
- Estimated ROI: medium-high. Rendering/resize is a material bucket, but prior
  broad resize rewrites were rejected, so keep this narrow and measurement-led.
- Hypothesis: The default grayscale renderer writes a cropped native frame and
  then area-resizes. Sprite priority handling can trigger repeated background
  opacity/color lookups. A measured cache of background opacity/color for the
  default crop may reduce sprite overlay work without revisiting previously
  losing direct-resize experiments.
- Target files: `src/emulator.rs`
- Prior evidence: Prior fused/default-area and direct nametable variants were
  discarded, so this should target sprite/background priority lookups rather
  than another broad resize rewrite. The 2026-07-01 policy-completion profile
  measured rendering/maxpool capture as the largest non-CPU bucket
  (`396ms` rendering plus `55ms` resize over 481 env steps). The canonical
  local grouped-lane validation profile also showed render and resize as
  material buckets, so a narrow sprite/background priority cache is worth
  measuring once CPU fast-forwards are tried.
- Plan: Add or use profiler counters to isolate sprite overlay/background
  priority work. If significant, prototype a default-path scratch representation
  that records background gray and opacity once for the cropped region and lets
  sprite overlay reuse it. Do not combine with resize or layout rewrites in the
  same candidate.
- Contract risks: Sprite priority, transparency, left-edge clipping, palette
  mirroring, or scroll behavior can regress visually.
- Required checks: Existing scratch-resize and stable-retro parity tests are
  mandatory; add a targeted sprite-priority parity case if needed, then run the
  required checks and paired Modal benchmark.
- Expected benchmark signal: Rough expectation `+3%` to `+8%` only if sprite
  overlay is a measured hotspot; discard quickly if the extra cache hurts
  locality.

### IDEA-20260701-003: Profile-Guided Audio Routine Fast-Forward

- Status: ready
- Perspective: emulator-core
- Estimated ROI: medium. The `$F200` page is hot, but the side-effect proof is
  harder than sprite-0/controller polling and the safe skip boundary is less
  obvious.
- Hypothesis: The emulator ignores APU audio output, but the SMB sound engine
  still burns CPU instructions and writes audio registers. If profiling shows a
  hot, side-effect-limited audio routine, a ROM-signature guarded fast-forward
  could preserve gameplay-facing RAM while skipping audio-only work.
- Target files: `src/emulator.rs`
- Prior evidence: The performance plan explicitly accepts no-audio scope. ROM
  analysis found the NMI path calling the sound routine around `$F2D0`, while
  `cpu_write` ignores most APU registers except controller/DMA surfaces. The
  2026-07-01 policy-completion profile had `$F200-$F2FF` as the second-hottest
  PC page, while the canonical local grouped-lane profile also kept `$F200`
  hot. This should be investigated after the narrower `$8150` sprite-0 loop.
- Plan: First use the prerequisite profiler to confirm hot PCs. If audio work
  is hot, identify a narrow routine or loop whose only observable effects are
  APU writes and internal sound RAM. Prototype a guarded fast-forward or no-op
  return only for the expected SMB PRG signature.
- Contract risks: SMB sound RAM may interact with timers, pause state, or other
  gameplay flags. This should be abandoned unless the profiler and parity tests
  make the side effects clear.
- Required checks: Add a targeted smoke/parity check over Level1-1 through
  Level1-4 startup and gameplay frames, then run the full required checks and
  paired Modal benchmark.
- Expected benchmark signal: Only worth keeping if robust paired speedup is
  clearly positive; rough expectation `+2%` to `+8%` if audio is hot.

### IDEA-20260701-007: Hot Basic-Block Interpreter Prototype

- Status: ready
- Perspective: emulator-core
- Estimated ROI: medium-low. Possible high upside, but implementation cost and
  silent-corruption risk are much higher than PC-specific fast-forwards.
- Hypothesis: The CPU core is still a large opcode `match`. For a fixed SMB
  NROM, hot side-effect-free basic blocks could be decoded once and executed
  with fewer fetch/decode branches while preserving cycle counts and memory
  side effects.
- Target files: `src/emulator.rs`
- Prior evidence: A broad "inline CPU step interpreter" candidate was already
  discarded. This idea should only proceed after profiler data identifies a
  small set of hot PCs that are stable across Level1-1 through Level1-4. The
  2026-07-01 profiles show a concentrated hot set (`$8150-$815F`,
  `$8220-$822F`, `$8E70-$8E7F`, `$F200-$F2FF`), but the safer first pass is
  still PC-specific fast-forwarding before a general block-cache prototype.
- Plan: Build a tiny, opt-in block cache for one or two hot blocks first.
  Restrict blocks to instructions with well-understood RAM/PRG accesses, stop
  at branches, JSR/RTS, PPU/controller reads, DMA, and interrupt boundaries.
- Contract risks: High. Any incorrect cycle count, flag update, interrupt
  boundary, or memory side effect can silently corrupt gameplay.
- Required checks: Add targeted CPU-block equivalence tests if implemented,
  run the full required checks, and require a strong paired Modal result before
  accepting.
- Expected benchmark signal: Unknown; possible `+5%` to `+15%` for a successful
  very narrow block cache, but likely discard if complexity grows.

### IDEA-20260701-006: Tune Rayon Chunking For Group Leaders

- Status: ready
- Perspective: vec-env
- Estimated ROI: lowest among current ready ideas. Group sharing itself already
  won; local profiling shows copy cost is tiny and the mask-reuse follow-up was
  a large Modal regression.
- Hypothesis: The accepted grouped-lane candidate wins by stepping repeated
  state group leaders in parallel, then copying peer outputs. The remaining
  overhead may be Rayon scheduling and cache behavior for a small number of
  leaders. A fixed chunking or small-group path could reduce scheduling cost
  without changing semantics.
- Target files: `src/vec_env.rs`
- Prior evidence: The single-thread grouped attempt was slower, while the
  parallel grouped-leader candidate was kept. The follow-up mask-reuse attempt
  was discarded, so avoid that mechanism. The 2026-07-01 canonical local
  profiler showed `group_hit_rate=1.0`, `6000` group leaders, `18000` peer
  copies, and about `460ns` grouped-copy time per env step. This makes the idea
  plausible but lower priority than CPU/render work; do not chase observation
  copy overhead first.
- Plan: Only try after the CPU/render candidates stall. Test one scheduling
  change at a time: fixed small-group leader arrays, explicit leader index list
  reuse, or a branch that uses sequential stepping when the leader count is
  below the measured break-even point.
- Contract risks: Accidentally serializing too much work, changing reset/group
  materialization semantics, or repeating the discarded mask-reuse mechanism.
- Required checks: `scripts/check_vec_env_equivalence.py`, full required checks,
  and paired Modal benchmark.
- Expected benchmark signal: Rough expectation `+1%` to `+4%`; discard quickly
  if host variance or scheduling overhead hides the signal.

## In Progress

## Done

### IDEA-20260630-D01: Group Repeated Saved-State Lanes

- Status: discard
- Perspective: vec-env
- Result: commit `c75909e`, artifact
  `artifacts/benchmarks/grouped-synced-state-lanes-2026-06-30-2135.json`.
- Reason: Modal result was substantially slower than the baseline despite the
  plausible mechanism. Do not repeat a single-thread grouped-lane design.

### IDEA-20260630-D02: Parallelize Repeated Saved-State Group Leaders

- Status: keep
- Perspective: vec-env
- Result: commit `68bea25`, artifact
  `artifacts/benchmarks/grouped-synced-state-lanes-parallel-2026-06-30-2142.json`.
- Reason: Parallel group leaders preserved Modal CPU parallelism while avoiding
  duplicate emulator work for repeated saved-state lanes.

### IDEA-20260630-D03: Reuse Persistent Grouped-Action Mask

- Status: discard
- Perspective: vec-env
- Result: commit `40089b0`, artifact
  `artifacts/benchmarks/reuse-synced-group-action-mask-2026-06-30-2147.json`.
- Reason: Clean Modal run was substantially slower than the accepted parallel
  grouped-state baseline. Avoid this mask-reuse follow-up shape.
