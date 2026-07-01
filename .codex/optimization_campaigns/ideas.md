# Autoresearch Ideas Queue

## Prerequisites

- Build a low-overhead hot-path profiler before selecting another speed
  candidate. This is infrastructure for choosing ideas, not itself an
  optimization idea. The profiler should be disabled by default and should not
  be judged by Modal speedup unless it accidentally changes the normal hot path.

## Ready

### IDEA-20260701-002: Fast-Forward SMB Sprite-0 Polling

- Status: ready
- Perspective: emulator-core
- Hypothesis: SMB has a known sprite-0 wait loop that repeatedly reads
  `PPUSTATUS` until the sprite-0 bit is set. The emulator already has a coarse
  PPU event model and a committed SMB idle-jump fast-forward; a ROM-signature
  guarded sprite-0 polling fast-forward could remove many interpreter dispatches
  while preserving frame timing.
- Target files: `src/emulator.rs`
- Prior evidence: `src/emulator.rs` models `PPU_SPRITE0_DOT` and has
  `SMB_IDLE_JMP_PC` fast-forward support, but no committed sprite-0 poll
  fast-forward. Prior ROM analysis identified the loop around `$8150`.
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

### IDEA-20260701-003: Profile-Guided Audio Routine Fast-Forward

- Status: ready
- Perspective: emulator-core
- Hypothesis: The emulator ignores APU audio output, but the SMB sound engine
  still burns CPU instructions and writes audio registers. If profiling shows a
  hot, side-effect-limited audio routine, a ROM-signature guarded fast-forward
  could preserve gameplay-facing RAM while skipping audio-only work.
- Target files: `src/emulator.rs`
- Prior evidence: The performance plan explicitly accepts no-audio scope. ROM
  analysis found the NMI path calling the sound routine around `$F2D0`, while
  `cpu_write` ignores most APU registers except controller/DMA surfaces.
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

### IDEA-20260701-004: Specialize Controller Polling Loop Safely

- Status: ready
- Perspective: emulator-core
- Hypothesis: SMB polls controller state through repeated `$4016` reads in NMI.
  For this emulator, controller state is already a compact byte. A guarded
  routine-level specialization could materialize the same input result with far
  fewer interpreted instructions.
- Target files: `src/emulator.rs`
- Prior evidence: The core already implements serial controller reads, and the
  benchmark action set uses deterministic per-frame button bytes. Prior ROM
  analysis identified controller polling around `$8E5C`.
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

### IDEA-20260701-005: Profile Sprite Overlay And Background Priority Cost

- Status: ready
- Perspective: ppu-render
- Hypothesis: The default grayscale renderer writes a cropped native frame and
  then area-resizes. Sprite priority handling can trigger repeated background
  opacity/color lookups. A measured cache of background opacity/color for the
  default crop may reduce sprite overlay work without revisiting previously
  losing direct-resize experiments.
- Target files: `src/emulator.rs`
- Prior evidence: Prior fused/default-area and direct nametable variants were
  discarded, so this should target sprite/background priority lookups rather
  than another broad resize rewrite.
- Plan: Use profiler counters to isolate sprite overlay cost. If significant,
  prototype a default-path scratch representation that records background gray
  and opacity once for the cropped region and lets sprite overlay reuse it.
- Contract risks: Sprite priority, transparency, left-edge clipping, palette
  mirroring, or scroll behavior can regress visually.
- Required checks: Existing scratch-resize and stable-retro parity tests are
  mandatory; add a targeted sprite-priority parity case if needed, then run the
  required checks and paired Modal benchmark.
- Expected benchmark signal: Rough expectation `+3%` to `+10%` only if sprite
  overlay is a measured hotspot.

### IDEA-20260701-006: Tune Rayon Chunking For Group Leaders

- Status: ready
- Perspective: vec-env
- Hypothesis: The accepted grouped-lane candidate wins by stepping repeated
  state group leaders in parallel, then copying peer outputs. The remaining
  overhead may be Rayon scheduling and cache behavior for a small number of
  leaders. A fixed chunking or small-group path could reduce scheduling cost
  without changing semantics.
- Target files: `src/vec_env.rs`
- Prior evidence: The single-thread grouped attempt was slower, while the
  parallel grouped-leader candidate was kept. The follow-up mask-reuse attempt
  was discarded, so avoid that mechanism.
- Plan: Profile the accepted path, then test one scheduling change at a time:
  fixed small-group leader arrays, explicit leader index list reuse, or a
  branch that uses sequential stepping when the leader count is below the
  measured break-even point.
- Contract risks: Accidentally serializing too much work, changing reset/group
  materialization semantics, or repeating the discarded mask-reuse mechanism.
- Required checks: `scripts/check_vec_env_equivalence.py`, full required checks,
  and paired Modal benchmark.
- Expected benchmark signal: Rough expectation `+2%` to `+8%`; discard quickly
  if host variance or scheduling overhead hides the signal.

### IDEA-20260701-007: Hot Basic-Block Interpreter Prototype

- Status: ready
- Perspective: emulator-core
- Hypothesis: The CPU core is still a large opcode `match`. For a fixed SMB
  NROM, hot side-effect-free basic blocks could be decoded once and executed
  with fewer fetch/decode branches while preserving cycle counts and memory
  side effects.
- Target files: `src/emulator.rs`
- Prior evidence: A broad "inline CPU step interpreter" candidate was already
  discarded. This idea should only proceed after profiler data identifies a
  small set of hot PCs that are stable across Level1-1 through Level1-4.
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
