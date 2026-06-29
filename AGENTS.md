# supermarioemu Codex Notes

## Optimization Skill

Use `/optimize-sps` for future throughput optimization rounds in this repo, especially work involving `scripts/benchmark_sps.py`, Super Mario Bros NES emulator hot paths, or `env_steps_per_sec` targets. The skill lives at `.codex/skills/optimize-sps/SKILL.md`.

## Modal Benchmark Skill

Use `/modal-benchmark` when the user wants the canonical clean-machine Modal CPU benchmark or a fresh 16-env baseline/comparison run. The skill runs `scripts/modal_benchmark_sps.py`, saves a JSON artifact under `artifacts/benchmarks/`, and reports the same compact throughput summary each time. The skill lives at `.codex/skills/modal-benchmark/SKILL.md`.
