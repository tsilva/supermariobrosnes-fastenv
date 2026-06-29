---
name: optimize-sps
description: Benchmark-driven SuperMarioEmu optimization workflow. Use when Codex is asked to improve, profile, verify, or continue optimization of `/Users/tsilva/repos/tsilva/supermarioemu`, especially work involving `scripts/benchmark_sps.py`, Super Mario Bros NES throughput, `env_steps_per_sec`, Rust emulator/vector-env hot paths, preprocessing correctness, or future optimization rounds after an existing speedup.
---

# Optimize SPS

## Operating Contract

Optimize the live repo, not a remembered snapshot. Preserve the public benchmark command and the externally observed benchmark contract unless the user explicitly changes the goal:

```bash
.venv/bin/python scripts/benchmark_sps.py --num-envs 16 --steps 500 --repeats 3
```

Expected benchmark contract:

- `obs_shape=(16, 4, 84, 84)`
- `obs_dtype=uint8`
- real Super Mario Bros NES reset/step behavior
- correct frame skip, frame stack, grayscale/crop/resize, action mapping, rewards, dones/truncations, reset behavior, and info scalar semantics

Do not fake throughput by skipping required emulator progression, returning stale observations, changing the public command, or silently weakening the benchmark workload.

Every optimization round must run this benchmark first as the baseline, optimize only after recording that baseline, and then run the same benchmark again for final acceptance. An optimization is only a success if the final arithmetic mean is more than 10% higher than the baseline arithmetic mean. Treat `<= 10%` gain, noisy evidence, or best-sample-only improvement as not good enough.

## Workflow

1. Inspect the live hot path before editing:
   - `scripts/benchmark_sps.py`
   - `python/supermarioemu/env.py`
   - `src/py_api.rs`
   - `src/vec_env.rs`
   - `src/emulator.rs`
   - `Cargo.toml`, `pyproject.toml`, and relevant docs
   - `git status --short`

2. Establish current timing:
   - Run the exact benchmark command with at least 3 repeats.
   - Record individual samples, arithmetic mean, sample stdev, and best.
   - Save this measurement as the baseline before making optimization edits.
   - If machine load is obviously noisy, say so and avoid treating noisy samples as final truth.
   - Run final benchmark commands sequentially, not in parallel with other CPU-heavy checks.

3. Profile or instrument enough to identify the bottleneck:
   - Use small controlled probes such as frame-skip, frame-stack, resize dimensions, `include_info`, lane count, or targeted counters.
   - Separate Python boundary cost, Rust vector-env scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack movement, and output-buffer copying.
   - Prefer measured evidence over generic optimization lists.

4. Optimize aggressively but honestly:
   - Favor Rust-side changes in `src/emulator.rs`, `src/vec_env.rs`, and `src/py_api.rs`.
   - Mario/NES-specific shortcuts are allowed when they preserve observed SMB behavior for this repo's supported game.
   - Keep generic emulator compatibility only when cheap.
   - If a shortcut relies on an assumption, document it in `docs/PERFORMANCE_PLAN.md`.

5. Rebuild before testing Python behavior:
   - Run `.venv/bin/python -m maturin develop --release`.
   - If sandboxing blocks uv/pip cache writes, rerun with escalated filesystem approval and explain that maturin needs cache access outside the repo.

6. Prove correctness before calling a speedup real:
   - Run `.venv/bin/python scripts/check_vec_env_equivalence.py` when present.
   - Run `.venv/bin/python scripts/smoke_smb.py`.
   - Add or update targeted checks when a new optimization changes semantics, especially for observations, rewards, termination flags, reset behavior, noop stepping, uniform-action lanes, and divergent-action lanes.
   - Compare optimized behavior to an independent reference path where practical.

7. Run final acceptance timing:
   - Run the exact benchmark command with at least 3 repeats.
   - Compare final mean against the recorded baseline mean from this same optimization round.
   - Compute gain as `(final_mean / baseline_mean - 1) * 100`.
   - Call the optimization successful only when gain is greater than 10%.
   - If gain is `<= 10%`, report the result as an insufficient speedup and do not present it as a win, even if correctness checks pass or one final sample was faster.
   - Report baseline samples, final samples, both means, both stdevs, both best values, gain percentage, speedup multiplier, checks run, and intentionally unsupported cases.

8. Leave the repo reviewable:
   - Run `cargo fmt`.
   - Run `cargo check --release`.
   - Note that plain `cargo test --release` may fail to link a PyO3 extension test harness unless Python symbols are configured; do not hide this if it happens.
   - Show `git status --short` and summarize changed files.

## Existing Fast-Path Assumptions

The repo currently includes two important optimization ideas that future work must preserve or deliberately replace with stronger checks:

- Deterministic synced lanes: after reset, identical lanes can share one emulator state while all actions are uniform; the state must materialize into independent lanes before mixed actions.
- Cropped grayscale tile rendering: the RL benchmark path emits SMB/NES background tile-row runs and then applies sprite overlay semantics.

Unsupported or intentionally narrow cases are acceptable when documented:

- only Super Mario Bros mapper 0 / NROM is in scope
- no audio requirement
- no general Gym Retro or arbitrary NES mapper compatibility requirement
- RGB and uncropped renderers are compatibility paths, not the primary optimized RL benchmark path
