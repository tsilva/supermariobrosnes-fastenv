---
name: modal-benchmark
description: Run the canonical clean-machine Modal CPU benchmark for this supermarioemu repo and report Super Mario Bros NES throughput. Use when the user invokes /modal-benchmark, asks to benchmark on Modal/modal.com, wants a clean CPU-only baseline, wants to compare an optimization on fresh compute, or asks for the current Modal benchmark result format.
---

# Modal Benchmark

## Workflow

Run the repo-local Modal launcher from the repository root:

```bash
modal run scripts/modal_benchmark_sps.py --output-json artifacts/benchmarks/modal-baseline-YYYY-MM-DD.json
```

Use the current date in the artifact name. If that path already exists, add a short time suffix, for example `modal-baseline-YYYY-MM-DD-HHMM.json`.

The command uploads the local repo snapshot to Modal and sends the local ROM bytes at runtime. If the user explicitly invoked this skill or asked for a Modal benchmark, treat that as approval to request escalated execution for Modal network/auth/upload. If the escalation reviewer still blocks the command, report that the benchmark could not run and name the upload risk plainly.

Use the launcher defaults unless the user asks otherwise:

- `num_envs=16`
- `steps=500`
- `repeats=3`
- `frame_skip=4`
- `frame_stack=4`
- grayscale, crop top 32, resize 84x84
- action `noop`

## Reporting

After the Modal run completes, read the saved JSON artifact and report the result in this shape:

- Start with whether it worked and briefly say Modal built the image, uploaded the repo snapshot, built/installed the Rust extension, uploaded ROM bytes at runtime, and ran the 16-env benchmark.
- Link the saved artifact with an absolute file link.
- Include a `Benchmark result` code block:

```text
runs env_steps_per_sec: RUN1, RUN2, RUN3
mean: MEAN
stdev: STDEV
best: BEST
obs_shape: (16, 4, 84, 84)
obs_dtype: uint8
```

- Include a `Modal machine metadata` code block:

```text
cpu_request: CPU_REQUEST
memory_mb: MEMORY_MB
os_cpu_count: OS_CPU_COUNT
affinity_cpu_count: AFFINITY_CPU_COUNT
```

- If the command output includes a Modal run URL, include it. If not, omit the URL rather than inventing one.
- Mention any launcher fixes made during the run. If no code changed, say so briefly.

## JSON Extraction

Use a short local read after the run to avoid hand-copying console output:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/benchmarks/modal-baseline-YYYY-MM-DD.json")
data = json.loads(path.read_text())
print("path", path)
print("mean", data["summary"]["env_steps_per_sec"]["mean"])
print("stdev", data["summary"]["env_steps_per_sec"]["stdev"])
print("best", data["summary"]["env_steps_per_sec"]["max"])
print("runs", [round(r["env_steps_per_sec"], 1) for r in data["runs"]])
print("obs", data["observation"]["shape"], data["observation"]["dtype"])
print(
    "modal",
    data["modal"]["cpu_request"],
    data["modal"]["memory_mb"],
    data["modal"]["os_cpu_count"],
    data["modal"]["affinity_cpu_count"],
)
PY
```

Run `git status --short` before the final answer so changed launcher/docs files are not hidden.
