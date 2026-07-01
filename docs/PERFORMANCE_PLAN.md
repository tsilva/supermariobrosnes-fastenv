# Fastest SMB RL Env Performance Plan

Objective: make the fastest Super Mario Bros NES environment for RL, while
preserving the Gymnasium-style step/reset contract and the training-facing
preprocessing contract: frame skip, grayscale, and frame stacking.

## Current Hot-Path Rules

- Python crosses into Rust once per vectorized batch step.
- Actions enter Rust as a contiguous `uint8[num_envs]` array.
- Observations, rewards, done flags, and scalar info arrays are preallocated
  NumPy buffers that Rust mutates in place.
- `step_fast()` avoids per-env Python `info` dictionaries.
- Rust releases the GIL during reset and step.
- Frame skip, grayscale conversion, frame stacking, reward extraction, and
  termination checks belong in Rust.
- The persistent observation buffer is the frame-stack state for the fast path.
- Identical reset lanes may be represented by one shared emulator state while
  every lane receives the same action. This is exact for deterministic SMB/NROM
  execution: the first lane is stepped once, and its observation, reward, done
  flags, `x_pos`, and `lives` are copied to the other identical lanes. If an
  action vector diverges, the shared state is cloned into independent lanes
  before the mixed action step runs.
- Repeated saved-state groups may use the same deterministic sharing rule per
  group. Mixed-level benchmark lanes still receive fully materialized
  observations and scalar info; if any action diverges inside a repeated-state
  group, the group is materialized before stepping.
- The default cropped grayscale renderer uses SMB/NES tile structure directly:
  background pixels are emitted in 8-pixel tile-row runs, then the existing
  sprite overlay path applies priority and palette semantics.

## Bottlenecks To Eliminate

### 1. Pixel Work That The Policy Never Needs

Fastest pixel path:

- PPU produces the final frame in an indexed/native palette format.
- Grayscale is generated directly from palette/index values.
- RGB is never materialized unless the caller explicitly asks for RGB.
- During frame skip, do not copy or convert intermediate frames.

Optional fastest observation path:

- Add RAM or compact feature observations for policies that do not require
  pixels.
- Keep pixel obs available for apples-to-apples Gym Retro comparisons.

### 2. Full-Stack Copies

Current optimized path uses the output buffer as stack state:

- Shift old frames left inside the preallocated observation buffer.
- Write only the newest frame into the final stack slot.
- Avoid an internal ring buffer plus full-stack copy into Python.

Future option:

- Add a channels-last output mode if the training stack consumes NHWC without
  transposes.
- Add a torch/DLPack path only if the learner can consume it without extra
  CPU-side copies.

### 3. Generic NES Dispatch

Super Mario Bros is mapper 0 / NROM. The first serious emulator core should be
SMB/NROM-specialized:

- No mapper trait-object dispatch in the hot path.
- Fixed PRG/CHR mapping.
- PRG and CHR reads use precomputed power-of-two address masks for mapper-0
  SMB/NROM ROMs, and instruction fetches go directly to PRG ROM when the
  program counter is in the cartridge range.
- Precomputed mirroring behavior.
- Direct CPU memory fast paths for RAM, PPU registers, controller, and PRG ROM.

General mapper support can come later, behind a slower compatibility path.

### 4. Python Object Allocation

The training path should not allocate per step:

- No `list[dict]` infos in the fast loop.
- No new obs/reward/done arrays in `step_wait_fast()`.
- Scalar info uses typed arrays such as `x_pos`, `lives`, `world`, `level`,
  `timer`, `score`, and `flags`.

Gymnasium-compatible `info` dictionaries should remain a convenience wrapper,
not the benchmark path.

### 5. Reward And Termination In Python

Reward and termination must be derived in Rust:

- Track SMB RAM variables directly.
- Compute all-time max x-position, score deltas, life loss, level completion,
  timeout, and death in Rust.
- Return compact typed arrays to Python.

This avoids Python RAM reads, dict parsing, and wrapper chains.

### 6. Interpreter Overhead In The CPU Core

The 6502 core should be designed for branch predictability and locality:

- Use a static opcode dispatch table or generated opcode match.
- Keep registers in plain fields.
- Inline addressing modes on hot opcodes.
- Avoid bounds checks in profiled ROM/RAM access paths where invariants are
  already proven.
- Benchmark with instruction counters, not only env SPS.

### 7. PPU Cost

For pixel policies the PPU is likely to become the largest true cost:

- Render only the final observable frame after frame skip.
- Still advance PPU state correctly for skipped frames.
- Use scanline/tile caches for background where SMB makes this profitable.
- Convert sprite/background palette indices directly to grayscale.
- Keep render buffers lane-local and reused.

### 8. Vector Lane Parallelism

Once single-thread hot paths are clean:

- Parallelize env lanes in Rust, not Python.
- Use a fixed thread pool.
- Chunk lanes to avoid per-env scheduling overhead.
- Keep each lane's emulator state independent and cache-local.
- Benchmark `num_envs x threads` grids to find saturation points.

Additional deterministic-lane fast path:

- Reset creates identical lanes for the benchmark and for many evaluation
  starts.
- Uniform action batches can share one emulator state until the first mixed
  action batch.
- Mixed action batches materialize independent lane states before stepping, so
  the public vector-env contract stays the same.

Intentionally unsupported cases:

- The emulator remains scoped to Super Mario Bros mapper 0 / NROM. Other NES
  mappers, save-state formats, audio, and general Gym Retro compatibility are
  outside this fast path.
- The primary fast path assumes standard power-of-two NROM PRG/CHR sizes.
- The shared-lane optimization assumes deterministic emulator state and no
  per-lane RNG, wrappers, or hidden side effects outside `NesEmulator`.
- The tiled renderer is the optimized grayscale cropped path used by the RL
  benchmark. RGB and uncropped compatibility renderers are kept separate and are
  not the primary optimization target.

### 9. Benchmark Integrity

Report two classes of numbers separately:

- Emulator hot-path diagnostics from direct ROM boot/state.
- Training-representative benchmarks from real gameplay states with the actual
  preprocessing contract.

For RL relevance, always report both:

- isolated env SPS
- learner-loop SPS with the intended algorithm stack

## Current Real-Core Benchmark

Real SMB/NROM no-audio core, grayscale, `frame_stack=4`, `frame_skip=4`,
alive lanes using `noop` actions:

- `num_envs=1`: about `1.18k` env steps/sec / `4.73k` emulated frames/sec
- `num_envs=8`: about `4.10k` env steps/sec / `16.42k` emulated frames/sec
- `num_envs=16`: about `2.47k` env steps/sec / `9.87k` emulated frames/sec

The first real bottleneck has moved into CPU/PPU/render work and cache behavior,
not Python/Rust switching. The current host favors an 8-lane batch over 16 lanes
for this build; thread/batch scaling should be treated as a benchmarked tuning
knob, not a constant.
