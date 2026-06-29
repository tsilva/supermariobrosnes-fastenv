---
name: autoresearch-speed
description: Unified Super Mario Bros emulator speed-improvement workflow for this repo. Use when Codex is asked to optimize, profile, benchmark, or coordinate self-improvement research for Super Mario Bros NES throughput, including single-agent optimization tracks, multi-agent branch/worktree campaigns, Modal benchmark tournaments, stale candidate replay, merge adjudication, or cleanup of speed-research worktrees.
---

# Autoresearch Speed

## Operating Contract

Optimize the live repo, not a remembered snapshot. Preserve the externally observed benchmark contract unless the user explicitly changes the goal:

```bash
.venv/bin/python scripts/benchmark_sps.py --num-envs 16 --steps 500 --repeats 3
```

Expected benchmark contract:

- `obs_shape=(16, 4, 84, 84)`
- `obs_dtype=uint8`
- real Super Mario Bros NES reset/step behavior
- correct frame skip, frame stack, grayscale/crop/resize, action mapping, rewards, dones/truncations, reset behavior, and info scalar semantics

Do not fake throughput by skipping required emulator progression, returning stale observations, changing the public command, or weakening the benchmark workload.

An optimization succeeds only when the final arithmetic mean `env_steps_per_sec` is more than 10% higher than the recorded baseline mean for the same track. Treat `<= 10%` gain, noisy evidence, or best-sample-only improvement as insufficient.

Benchmarks for this skill are Modal-only. Use `/modal-benchmark` for baseline, profiling comparisons that produce throughput claims, candidate comparison, and final timing. Do not run local throughput benchmarks as a fallback, do not use local SPS numbers as evidence, and do not continue an optimization track when Modal benchmark execution is unavailable, blocked, or not explicitly authorized. Local commands may be used only for correctness, formatting, compilation, and non-throughput inspection.

Before any single-agent or campaign optimization work that will benchmark throughput, ask the user for explicit Modal pre-authorization by giving them this exact phrase to copy, fill in, and send back:

```text
I approve /autoresearch-speed with track_mode=<single_agent|campaign>; Modal network/auth/upload access is allowed; local repo snapshot upload is allowed; local ROM byte upload at benchmark runtime is allowed; local state byte upload at benchmark runtime is allowed; max_parallel_modal_runs=<N>; max_total_modal_runs=<N>; max_estimated_spend_usd=<USD>.
```

Approval is cleared only when the user's reply preserves the permission grants and fills in concrete values for `track_mode`, `max_parallel_modal_runs`, `max_total_modal_runs`, and `max_estimated_spend_usd`. If any field is missing, vague, or replaced with an open-ended limit, stop and ask for the exact phrase again. Do not treat informal approval, partial approval, or approval for a different skill name as sufficient.

If the user does not grant that envelope, or if the execution environment rejects Modal upload/run approval, stop and report the blocker. Do not switch to local benchmarks.

## Track Selection

Use `track_mode=single_agent` when exactly one worker agent should run a complete optimization loop under the same coordinator/judge process used for campaigns.

Use `track_mode=campaign` when multiple agents will explore separate branches/worktrees and the main agent will judge what merges.

If the user does not specify a mode, default to `single_agent` for ordinary optimization requests and `campaign` for self-improvement, tournament, multiple-agent, branch/worktree, or parallel-research requests.

## Agent Coordination Model

The main agent is always the coordinator and judge, including `single_agent`. Do not implement optimization candidates inline in the coordinator. For `single_agent`, fork exactly one worker agent and run the same work-item, candidate-submission, correctness-check, Modal remeasure, verdict, merge, and cleanup loop used for N-agent campaigns. The only difference is concurrency and search breadth: `single_agent` has one active worker and one active work item unless the user explicitly expands the run.

If subagent/fork tooling is unavailable, stop and report that blocker. Do not silently degrade to an inline coordinator-only optimization loop.

## Single-Agent Track

1. Ask for the required Modal permission envelope before forking the worker or running any throughput benchmark. If Modal or subagent approval is unavailable, stop.

2. Inspect the live hot path before creating the worker assignment:
   - `scripts/benchmark_sps.py`
   - `scripts/modal_benchmark_sps.py` when Modal is involved
   - `python/supermariobrosnes_fastenv/env.py`
   - `src/py_api.rs`
   - `src/vec_env.rs`
   - `src/emulator.rs`
   - `Cargo.toml`, `pyproject.toml`, relevant docs, and `git status --short`

3. Establish a baseline before optimization:
   - Run `/modal-benchmark` with at least 3 repeats on Modal.
   - Record individual samples, arithmetic mean, sample stdev, and best.
   - If load or Modal metadata looks noisy or not comparable, say so and do not treat the result as final truth.

4. Create one problem-shaped work item and fork one worker agent for it:
   - Use the same work-item fields required by campaigns: `id`, `hypothesis`, `scope`, `status`, `owner`, `lease_expires_at`, `base_sha`, and `worktree_path`.
   - Require the same worker submission fields as campaigns.
   - Treat the worker's measurements and patch as advisory until the coordinator replays correctness checks and Modal judging.

5. Profile enough to identify the bottleneck:
   - Use controlled Modal probes such as frame skip, frame stack, resize dimensions, `include_info`, lane count, or targeted counters.
   - Separate Python boundary cost, Rust vector-env scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack movement, and output-buffer copying.
   - Prefer measured evidence over generic optimization lists.

6. Optimize aggressively but honestly:
   - Favor Rust-side changes in `src/emulator.rs`, `src/vec_env.rs`, and `src/py_api.rs`.
   - Mario/NES-specific shortcuts are allowed only when they preserve observed SMB behavior for this repo's supported game.
   - Document important shortcut assumptions in `docs/PERFORMANCE_PLAN.md`.

7. Rebuild before testing Python behavior:
   - Run `.venv/bin/python -m maturin develop --release`.
   - If sandboxing blocks uv/pip/cache writes, rerun with the required approval and explain why.

8. Prove correctness before claiming speedup:
   - Run `.venv/bin/python scripts/check_vec_env_equivalence.py` when present.
   - Run `.venv/bin/python scripts/smoke_smb.py`.
   - Add or update targeted checks when changing observations, rewards, termination flags, reset behavior, noop stepping, uniform-action lanes, or divergent-action lanes.
   - Run `cargo fmt` and `cargo check --release`.

9. Run final timing:
   - Run the same Modal benchmark/profile used for the baseline.
   - Compute gain as `(final_mean / baseline_mean - 1) * 100`.
   - Report baseline samples, final samples, both means, both stdevs, both best values, gain percentage, speedup multiplier, checks run, and changed files.
   - End with paste-ready manual playback commands for both regular and preprocessed views:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```

   - If state files require a non-default location, append `--state-dir <path>` to both commands.

## Campaign Track

The main agent is the coordinator and judge. Worker agents provide evidence and patch proposals; they do not authorize merges.

Before spawning agents or running worker Modal benchmarks, ask the user for the exact `/autoresearch-speed` approval phrase from the Operating Contract. It must authorize:

- Modal network/auth/upload access
- local repo snapshot upload
- local ROM byte upload at benchmark runtime
- local state byte upload at benchmark runtime
- maximum parallel Modal runs
- maximum total Modal runs or estimated spend

Record the permission envelope in `.codex/optimization_campaigns/current.json`:

```json
{
  "campaign_id": "sps-YYYY-MM-DD",
  "authorized_by_user": true,
  "authorized_at": "ISO-8601",
  "allowed_command": "modal run scripts/modal_benchmark_sps.py",
  "allowed_output_root": "artifacts/benchmarks/",
  "max_parallel_modal_runs": 4,
  "max_campaign_runs": 20,
  "integration_sha": "HEAD",
  "current_baseline_artifact": "artifacts/benchmarks/...",
  "work_items": [],
  "candidates": []
}
```

Agents may request escalation only for the exact `/modal-benchmark` command shape:

```bash
modal run scripts/modal_benchmark_sps.py --output-json artifacts/benchmarks/<campaign-or-candidate>.json
```

Any altered deploy script, output outside `artifacts/benchmarks/`, unrelated network access, Modal secret changes, pushes, or broad uploads require separate approval.

## Campaign Work Items

Create problem-shaped work items, not file-shaped assignments. Good examples:

- `ppu-cropped-grayscale-render`
- `cpu-opcode-dispatch`
- `frame-stack-buffering`
- `vec-lane-materialization`
- `resize-inner-loop`
- `python-boundary-and-info-path`

Each work item should include `id`, `hypothesis`, `scope`, `status`, `owner`, `lease_expires_at`, `base_sha`, and `worktree_path`. Preserve active leased worktrees only while the lease is valid.

Worker submissions must include:

- `work_item_id`
- `base_sha`
- `head_sha`
- `hypothesis`
- `changed_files`
- `patch_or_branch`
- `claimed_modal_artifact` when run
- risks, failed attempts, and intentionally unsupported cases

Treat worker Modal artifacts as advisory. The coordinator must replay and remeasure before merge.

## Judge, Merge, And Cleanup

For each candidate:

1. Apply the patch or branch onto a fresh temporary integration worktree from the current integration SHA.
2. If it does not apply cleanly, resolve only obvious mechanical conflicts; otherwise mark `stale_rework_needed` or `conflict_needs_human_review`.
3. Run correctness checks locally.
4. Run `/modal-benchmark` with a candidate-specific artifact name when the campaign uses Modal judging.
5. Compare the candidate JSON against the current integration baseline JSON using mean vs mean, not best sample vs baseline.
6. Emit one verdict: `accepted`, `rejected`, `stale_rework_needed`, or `conflict_needs_human_review`.

Merge one candidate at a time. After every accepted merge:

- rerun `/modal-benchmark` on the new integration state
- promote the new artifact to the baseline
- mark all unmerged candidates stale until re-judged
- preserve patch, artifact paths, verdict, and notes in the manifest
- remove temporary judge worktrees
- remove merged, rejected, expired, or superseded campaign worktrees
- run `git worktree list` and `git worktree prune`
- confirm no stale campaign paths remain
- end the report with the regular and preprocessed manual playback commands from the Single-Agent final timing section

Do not assume independent gains add. Rebase and remeasure remaining candidates against the new baseline.

## Existing Fast-Path Assumptions

Preserve or deliberately replace these with stronger checks:

- Deterministic synced lanes: after reset, identical lanes can share one emulator state while all actions are uniform; the state must materialize into independent lanes before mixed actions.
- Cropped grayscale tile rendering: the RL benchmark path emits SMB/NES background tile-row runs and then applies sprite overlay semantics.

Unsupported or intentionally narrow cases are acceptable when documented:

- only Super Mario Bros mapper 0 / NROM is in scope
- no audio requirement
- no general Gym Retro or arbitrary NES mapper compatibility requirement
- RGB and uncropped renderers are compatibility paths, not the primary optimized RL benchmark path
