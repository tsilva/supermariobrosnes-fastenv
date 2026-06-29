# supermarioemu

Fast Rust-first Super Mario Bros NES environment for RL.

The main performance rule is simple: Python should call Rust once per vectorized
batch step. Frame skip, grayscale conversion, frame stacking, reward extraction,
termination checks, and observation-buffer writes happen on the Rust side.

Current emulator scope:

- Super Mario Bros / mapper 0 NROM fast path
- 6502 CPU interpreter
- no-audio PPU timing, VRAM, OAM DMA, controller input, vblank/NMI
- grayscale and RGB frame output
- Rust-side vector stepping with lane parallelism for larger batches

## Intended hot path

```python
env = SuperMarioBrosVecEnv(
    rom_path="~/Desktop/roms/SuperMarioBros.nes",
    num_envs=64,
    frame_skip=4,
    grayscale=True,
    frame_stack=4,
    crop_top=32,
    resize_width=84,
    resize_height=84,
)

obs = env.reset()
env.step_async(actions)
obs, rewards, terminated, truncated, infos = env.step_wait()
```

Under the hood `step_wait()` calls a single Rust `step_into(...)` method that
fills preallocated NumPy arrays in place.

## Build

```bash
uv sync --extra dev
uv run maturin develop --release
```

## Smoke And Benchmark

```bash
uv run python scripts/smoke_smb.py
uv run python scripts/benchmark_vec_env.py --num-envs 8 --frame-skip 4 --frame-stack 4
uv run python scripts/benchmark_sps.py --num-envs 16 --steps 500 --repeats 3
```

The `start` action is included so the raw ROM can leave the title screen without
special Python-side reset logic.

### Clean CPU Benchmarks On Modal

Use Modal for the canonical optimization baseline so timings come from a fresh
CPU-only machine instead of a busy local workstation. The default profile is the
standard 16-env benchmark. Run this from the repo root with an authenticated
Modal CLI (`modal setup` if this is the first time):

```bash
modal run scripts/modal_benchmark_sps.py \
  --output-json artifacts/benchmarks/modal-baseline.json
```

For each optimization attempt, run the same command again and save a new JSON
artifact:

```bash
modal run scripts/modal_benchmark_sps.py \
  --output-json artifacts/benchmarks/modal-optimization.json
```

The launcher sends the local ROM bytes to the remote container at run time, so
the Modal image stays generic while the benchmark still uses the same ROM as the
local scripts. Override it with `--rom-path` if needed. The JSON includes the
benchmark config, per-repeat timings, summary statistics, Modal CPU/memory
metadata, local Git commit/status, and the ROM SHA-256.

## Play

```bash
uv run python scripts/play.py --mode external
uv run python scripts/play.py --mode external --view preprocessed --scale 4
```

`external` mode reads keyboard input in Python, maps it to the discrete
Gymnasium-style action IDs, and calls `SuperMarioBrosEnv.step(action)` each
frame. The default raw view disables RL preprocessing for play: no frame stack,
no grayscale, and no frame skip. Both views disable RL flagpole termination so
the game can continue through SMB's own end-of-level sequence. Play mode uses
the native SDL2 backend for fast scaled display. `--view preprocessed` instead
shows the Gym observation tensor with grayscale, frame stacking, area resize to
84x84, and the default 32-pixel HUD crop; stacked frames are tiled in the SDL
window so you can see exactly what the agent would receive.
Controls: arrows or A/D move, X/J/Space jump, Z/K/Shift run, Enter start,
Esc quit.
