---
name: autoresearch-speed
description: Unified Super Mario Bros emulator speed-improvement workflow for this repo. Use when Codex is asked to optimize, profile, benchmark, or coordinate self-improvement research for Super Mario Bros NES throughput, including N-agent proposal and implementation waves, Modal benchmark tournaments, stale candidate replay, merge adjudication, or cleanup of speed-research worktrees.
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

An optimization succeeds only when the final arithmetic mean `env_steps_per_sec` is more than 10% higher than the current recorded baseline mean for the same benchmark configuration. Treat `<= 10%` gain, noisy evidence, or best-sample-only improvement as insufficient.

Benchmarks for this skill are Modal-only and must go through `/modal-benchmark`. Use `/modal-benchmark` for the baseline, every worker final benchmark, every coordinator reproduction benchmark, and final timing. Do not run local throughput benchmarks, raw `modal run` commands, ad hoc Modal scripts, or modified benchmark commands as a fallback. Local commands may be used only for correctness, formatting, compilation, and non-throughput inspection.

Before any optimization work that will benchmark throughput, ask the user for explicit Modal pre-authorization by giving them this exact phrase to copy, fill in, and send back:

```text
I approve /autoresearch-speed with num_agents=<N>; Modal network/auth/upload access is allowed; local repo snapshot upload is allowed; local ROM byte upload at benchmark runtime is allowed; local state byte upload at benchmark runtime is allowed; max_parallel_modal_runs=<N>; max_total_modal_runs=<N>; max_estimated_spend_usd=<USD>.
```

Approval is cleared only when the user's reply preserves the permission grants and fills in concrete values for `max_parallel_modal_runs`, `max_total_modal_runs`, and `max_estimated_spend_usd`. If `num_agents` is missing, default to 3 and record `num_agents: 3` in the manifest. If any required limit is missing, vague, or replaced with an open-ended limit, stop and ask for the exact phrase again. Do not treat informal approval, partial approval, or approval for a different skill name as sufficient.

If the user does not grant that envelope, or if the execution environment rejects Modal upload/run approval, stop and report the blocker. Do not switch to local benchmarks.

## Agent Count

The user chooses only the number of subagents. There is no special single-agent mode and no track selection.

If the user does not specify a number, use `num_agents=3`. If the user specifies a number, use that number for both the proposal wave and the implementation wave unless the coordinator deliberately launches fewer implementation agents after deduplication because fewer worthwhile proposals remain.

## Agent Coordination Model

The main agent is always the orchestrator, deduper, reviewer, and merge judge. Do not implement optimization candidates inline in the orchestrator. Worker agents generate proposals or implementation candidates; they do not authorize merges.

If subagent/fork tooling is unavailable, stop and report that blocker. Do not silently degrade to an inline coordinator-only optimization loop.

Before proposal or implementation workers start, the orchestrator must create a unique autoresearch branch from the local `turbo` branch and continue the whole run from that branch. Use a Git-ref-safe UTC ISO timestamp, for example `codex/autoresearch-2026-06-29T17-42-10Z`. The branch isolation is required so accepted candidates, rejected candidates, benchmark artifacts, and manifest edits can be managed safely without destabilizing `turbo` or colliding with other autoresearch runs. If the local `turbo` branch is missing, or if the worktree has unrelated dirty changes that would be carried into the branch, stop and ask the user how to proceed instead of branching from the current branch by accident.

The orchestration is optimized for two goals:

- actual effectiveness at producing worthwhile optimizations
- percent throughput improvement per wall-clock time

Optimize for `reproduced_gain_pct / elapsed_wall_clock_minutes`, not for the number of agents used, number of patches attempted, or best-looking single benchmark sample.

## Shared-State And Race Rules

The orchestrator is the only writer to `.codex/optimization_campaigns/current.json`. Worker agents must not edit the shared manifest directly. Workers report through their own branch/worktree handoff message or a worker-specific result file path assigned by the orchestrator; the orchestrator then validates and copies the result into the manifest.

The orchestrator owns the Modal run counter and a `max_parallel_modal_runs` semaphore. A worker may run its one final `/modal-benchmark` only after the orchestrator grants a run slot from the remaining budget. If no slot is granted, the worker still returns the patch and local correctness evidence, and the orchestrator decides whether the candidate deserves a reproduction run.

Use short leases and expiry times for proposal and implementation workers. Do not wait indefinitely for a slow worker: when the lease expires, rank and launch from the completed proposals or completed candidates already available. Wall-clock efficiency beats perfect coverage.

## Orchestration Flow

1. Ask for the required Modal permission envelope before forking workers or running any throughput benchmark. If Modal or subagent approval is unavailable, stop.

2. Create the isolated orchestration branch from `turbo` before launching workers:
   - Compute a Git-ref-safe UTC ISO timestamp such as `2026-06-29T17-42-10Z`.
   - Create and switch to `codex/autoresearch-<timestamp>` from local `turbo`.
   - Record `root_branch`, `orchestration_branch`, `created_from_sha`, and `created_at` in the manifest.
   - Do not create proposal or implementation worktrees from `turbo` directly after this point; use the orchestration branch as the integration root for the run.

3. Inspect the live hot path before launching workers:
   - `scripts/benchmark_sps.py`
   - `scripts/modal_benchmark_sps.py` when Modal is involved
   - `python/supermariobrosnes_fastenv/env.py`
   - `src/py_api.rs`
   - `src/vec_env.rs`
   - `src/emulator.rs`
   - `Cargo.toml`, `pyproject.toml`, relevant docs, and `git status --short`

4. Start the baseline and proposal wave with deliberate overlap:
   - Launch the baseline `/modal-benchmark` as soon as the orchestration branch is ready and approval is available.
   - While the baseline runs, fork `num_agents` proposal workers. Proposal workers are read-only and do not need the fresh baseline to identify hot-path ideas.
   - Do not rank or launch implementation workers until the baseline artifact is available, because the baseline mean defines the 10% acceptance threshold.
   - Record individual samples, arithmetic mean, sample stdev, best, output artifact, Modal run metadata, and base SHA.
   - If load or Modal metadata looks noisy or not comparable, say so and do not treat the result as final truth.

5. Record the permission envelope, orchestration branch, and baseline in `.codex/optimization_campaigns/current.json`:

```json
{
  "campaign_id": "sps-YYYY-MM-DD",
  "authorized_by_user": true,
  "authorized_at": "ISO-8601",
  "num_agents": 3,
  "root_branch": "turbo",
  "orchestration_branch": "codex/autoresearch-YYYY-MM-DDTHH-MM-SSZ",
  "created_from_sha": "turbo sha",
  "created_at": "ISO-8601",
  "allowed_benchmark_skill": "/modal-benchmark",
  "allowed_output_root": "artifacts/benchmarks/",
  "max_parallel_modal_runs": 3,
  "max_total_modal_runs": 10,
  "max_estimated_spend_usd": 10,
  "modal_runs_used": 1,
  "modal_runs_reserved_for_reproduction": 1,
  "integration_sha": "HEAD",
  "current_baseline_artifact": "artifacts/benchmarks/...",
  "current_baseline_mean_env_steps_per_sec": 0,
  "proposals": [],
  "rejected_proposals": [],
  "work_items": [],
  "candidates": [],
  "efficiency_metrics": {}
}
```

6. Phase 1: proposal workers. Proposal workers do not edit code and do not run benchmarks. Each proposal worker must inspect the hot path and return:
   - `proposal_id`
   - hypothesis
   - expected bottleneck
   - expected throughput upside and confidence
   - estimated implementation wall-clock time
   - estimated correctness/review cost
   - expected files and ownership boundaries
   - files likely to change
   - correctness risks
   - minimum tests/checks needed
   - suggested `/modal-benchmark` comparison artifact name for the implementation worker
   - known overlap with other likely ideas

7. The orchestrator dedupes and ranks proposals before implementation:
   - Merge duplicate or strongly overlapping ideas into one proposal.
   - Compute a rough priority score: `expected_gain_pct * confidence / (estimated_wall_clock_minutes + review_risk_penalty + modal_wait_penalty)`.
   - Prefer proposals with high expected speedup, high confidence, low correctness risk, low coupling to other proposals, and small implementation surface.
   - Prefer independent proposals that touch different bottleneck surfaces, because they reduce merge conflicts and stale remeasurement.
   - Reject vague proposals, proposals that weaken the benchmark contract, and proposals whose expected gain is unlikely to clear 10%.
   - Launch at most `num_agents` implementation workers, and launch fewer if fewer worthwhile independent proposals remain or Modal budget cannot support worker final runs plus at least one coordinator reproduction.
   - Preserve the rejected proposal reasons in the manifest so future proposal waves do not rediscover the same dead ends.

8. Phase 2: fork implementation workers, one per selected proposal, each in a separate branch/worktree rooted at the orchestration branch. Work items are proposal-shaped, not file-shaped. Good examples:
   - `ppu-cropped-grayscale-render`
   - `cpu-opcode-dispatch`
   - `frame-stack-buffering`
   - `vec-lane-materialization`
   - `resize-inner-loop`
   - `python-boundary-and-info-path`

   Each implementation work item must include `id`, `proposal_id`, `hypothesis`, `scope`, `status`, `owner`, `lease_expires_at`, `base_sha`, `worktree_path`, `expected_gain_pct`, and `risk_notes`.

9. Implementation workers optimize aggressively but honestly:
   - Use local profiling and instrumentation only for diagnosis, never as throughput evidence.
   - Separate Python boundary cost, Rust vector-env scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack movement, and output-buffer copying.
   - Prefer measured evidence over generic optimization lists.
   - Favor Rust-side changes in `src/emulator.rs`, `src/vec_env.rs`, and `src/py_api.rs`.
   - Mario/NES-specific shortcuts are allowed only when they preserve observed SMB behavior for this repo's supported game.
   - Document important shortcut assumptions in `docs/PERFORMANCE_PLAN.md`.
   - Stop early and hand back a negative result if the idea proves impossible, contract-breaking, or too invasive for its expected gain. Do not burn the full lease on a doomed proposal.

10. Rebuild before testing Python behavior:
   - Run `.venv/bin/python -m maturin develop --release`.
   - If sandboxing blocks uv/pip/cache writes, rerun with the required approval and explain why.

11. Prove correctness before claiming speedup:
   - Run `.venv/bin/python scripts/check_vec_env_equivalence.py` when present.
   - Run `.venv/bin/python scripts/smoke_smb.py`.
   - Add or update targeted checks when changing observations, rewards, termination flags, reset behavior, noop stepping, uniform-action lanes, or divergent-action lanes.
   - Run `cargo fmt` and `cargo check --release`.

12. Each implementation worker may run exactly one final `/modal-benchmark` for its own branch/worktree after local correctness checks pass and after the orchestrator grants a Modal run slot. The worker must not spend extra Modal runs trying to tune the result unless the orchestrator explicitly grants another run from the remaining budget.

13. Each implementation worker hands results back to the orchestrator with:
   - `work_item_id`
   - `proposal_id`
   - `base_sha`
   - `head_sha`
   - changed files
   - patch or branch/worktree path
   - local checks run and results
   - claimed Modal artifact, if run
   - `/modal-benchmark` command shape used, if run
   - baseline mean used for comparison
   - candidate mean, stdev, best, individual samples
   - claimed gain percentage
   - elapsed wall-clock time from assignment to handoff
   - whether the candidate should be reproduced, rejected, or reworked
   - risks, failed attempts, and intentionally unsupported cases

14. The orchestrator triages candidate results immediately:
   - If the worker did not run local correctness checks, mark `rejected_needs_checks`.
   - If the worker has no Modal artifact, mark `needs_judge_benchmark` only when the code diff is plausibly high-value; otherwise reject.
   - If the worker's claimed Modal mean is less than or equal to 10% over the current baseline mean, immediately discard the candidate as `rejected_under_threshold`.
   - If the worker's evidence is noisy, malformed, or not comparable to the baseline, mark `rejected_inconclusive` or rerun only if the code diff is exceptionally promising and budget remains.
   - If the claimed gain is more than 10%, inspect the diff for contract violations, shortcut dishonesty, excessive risk, and maintainability before spending a coordinator reproduction run.
   - Prioritize reproduction by `claimed_gain_pct / expected_review_minutes`, not by worker completion order.

15. For promising candidates, the orchestrator applies the branch or patch onto the current orchestration branch integration worktree, not directly onto `turbo`. If it does not apply cleanly, resolve only obvious mechanical conflicts; otherwise mark `stale_rework_needed` or `conflict_needs_human_review`.

16. The orchestrator reruns correctness checks on the integrated candidate and then launches a fresh `/modal-benchmark` reproduction from that integration state. Compare reproduced mean vs current baseline mean, not best sample vs baseline. If multiple promising candidates are ready, reproduce one at a time in priority order so every accepted merge updates the baseline before later candidates are judged.

17. If the reproduced result is more than 10% over the current baseline mean and the diff is acceptable, merge the integration worktree into the orchestration branch, commit the accepted candidate there, promote the reproduced artifact to the current baseline, and continue to the next candidate. If reproduction fails, discard or mark for rework.

18. After every accepted merge:
   - rerun or preserve the reproduced `/modal-benchmark` as the new baseline artifact
   - update `current_baseline_artifact`, `current_baseline_mean_env_steps_per_sec`, `integration_sha`, and `modal_runs_used`
   - mark all unmerged candidates stale until rejudged against the new baseline
   - preserve patch, artifact paths, verdict, and notes in the manifest
   - remove temporary judge worktrees
   - remove merged, rejected, expired, or superseded implementation worktrees
   - run `git worktree list` and `git worktree prune`
   - confirm no stale campaign paths remain

19. End the report with the orchestration branch name, baseline samples, accepted candidate samples, both means, both stdevs, both best values, gain percentage, speedup multiplier, checks run, changed files, and paste-ready manual playback commands for both regular and preprocessed views:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```

   - If state files require a non-default location, append `--state-dir <path>` to both commands.

Agents may request benchmark escalation only by invoking `/modal-benchmark`. The underlying command shape used by that skill is:

```bash
modal run scripts/modal_benchmark_sps.py --output-json artifacts/benchmarks/<campaign-or-candidate>.json
```

Any direct benchmark command outside `/modal-benchmark`, altered deploy script, output outside `artifacts/benchmarks/`, unrelated network access, Modal secret changes, pushes, or broad uploads require separate approval.

Do not assume independent gains add. Rebase and remeasure remaining candidates against the new baseline.

## Modal Budget Strategy

Reserve Modal runs before launching implementation work:

- 1 run for the baseline
- up to 1 final worker run for each selected implementation worker
- at least 1 coordinator reproduction run for the best promising candidate
- optionally 1 final confirmation run after all accepted merges if budget allows

If `max_total_modal_runs` is too small for `num_agents`, reduce the number of implementation workers before forking them. Prefer fewer well-chosen implementation workers with enough reproduction budget over many workers whose results cannot be verified.

Keep Modal concurrency below both `max_parallel_modal_runs` and the number of runs that can still be usefully acted on. Proposal workers should overlap with the baseline; implementation worker final benchmarks may run in parallel only when they have independent worktrees, completed correctness checks, and granted run slots. Coordinator reproductions run serially because each accepted merge changes the baseline for the next candidate.

## Protocol Optimization Loop

At the end of every autoresearch run, the orchestrator must review the orchestration protocol itself before final reporting. This is a review of the process, not another benchmark run.

Record these efficiency metrics in the manifest:

- total wall-clock minutes
- baseline wait minutes
- proposal wave elapsed minutes
- implementation wave elapsed minutes
- coordinator review/reproduction elapsed minutes
- Modal runs used, Modal runs discarded, and Modal runs saved by early rejection
- number of proposals received, deduped, rejected, implemented, benchmarked, reproduced, accepted, and stale after merge
- accepted reproduced gain percentage
- `accepted_reproduced_gain_pct / total_wall_clock_minutes`
- top avoidable wait source
- top avoidable race/conflict source

Use the metrics to tune the next run:

- If proposal quality was weak, make the next proposal prompt narrower and include the rejected-proposal list.
- If workers collided on files, add stricter ownership boundaries before the implementation wave.
- If Modal budget ran out before reproduction, launch fewer implementation workers next time.
- If workers waited on Modal slots, lower implementation concurrency or reserve more budget before launch.
- If coordinator review dominated wall time, prefer smaller diffs and more targeted proposals.
- If most workers were rejected under 10%, raise the proposal expected-gain threshold or require stronger bottleneck evidence.

Do not silently edit this skill at the end of a run. Summarize any protocol improvement that would prevent repeated waste or race conditions, then ask the user before updating the skill.

## Workflow Critique And Efficiency Notes

This two-phase design is stronger than immediately launching implementation agents because the cheap proposal wave increases search diversity before expensive worktree and Modal budget are spent. It also gives the orchestrator a chance to dedupe overlapping ideas, reject low-upside work, and allocate implementation agents to the highest expected gain per wall-clock hour.

The main weakness is added latency: proposal wave plus implementation wave can delay the first patch. Keep proposal agents time-boxed and require concrete file-level hypotheses, expected gain, and correctness risks. If proposal quality is low, the orchestrator should stop, refine the problem statement, and relaunch proposals instead of letting vague ideas consume implementation slots.

A second weakness is benchmark noise. One worker `/modal-benchmark` run is useful for fast screening, but it is not enough to merge. The design handles this by requiring coordinator reproduction before acceptance. To improve wall-clock efficiency, discard under-10% candidates immediately, reproduce high-gain candidates first, and avoid spending reproduction runs on diffs that look contract-breaking or hard to maintain.

A third weakness is stale-work risk. After one candidate is accepted, remaining candidates were measured against the old baseline and may conflict or lose their effect. The orchestrator must rebase/reapply and remeasure remaining work against the new baseline; independent gains must never be added arithmetically.

The default `num_agents=3` is a reasonable balance for this repo: enough diversity to cover renderer, vector-env, and Python-boundary hypotheses, but not so many workers that dedupe/review overhead dominates. Increase N only when the bottleneck space is broad and Modal budget can support the extra worker final runs plus coordinator reproductions.

## Existing Fast-Path Assumptions

Preserve or deliberately replace these with stronger checks:

- Deterministic synced lanes: after reset, identical lanes can share one emulator state while all actions are uniform; the state must materialize into independent lanes before mixed actions.
- Cropped grayscale tile rendering: the RL benchmark path emits SMB/NES background tile-row runs and then applies sprite overlay semantics.

Unsupported or intentionally narrow cases are acceptable when documented:

- only Super Mario Bros mapper 0 / NROM is in scope
- no audio requirement
- no general Gym Retro or arbitrary NES mapper compatibility requirement
- RGB and uncropped renderers are compatibility paths, not the primary optimized RL benchmark path
