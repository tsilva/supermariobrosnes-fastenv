from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    UV_PROJECT_DIR = str(REPO_ROOT.relative_to(Path.cwd().resolve())) or "."
except ValueError:
    UV_PROJECT_DIR = str(REPO_ROOT)
REMOTE_REPO = "/root/SuperMarioBros-Nes-turbo"
REMOTE_ROM = "/tmp/SuperMarioBros-Nes-v0.nes"
REMOTE_STATE_DIR = "/tmp/SuperMarioBros-Nes-turbo-states"
DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
DEFAULT_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
CPU_REQUEST = 16.0
MEMORY_MB = 8192
PYTHON_VERSION = "3.12"

IMAGE_IGNORE = [
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "artifacts",
    "target",
    "__pycache__",
    "*.egg-info",
    "*.pyc",
    "*.so",
]

app = modal.App("supermariobros-nes-turbo-cpu-benchmarks")

image = (
    modal.Image.from_registry("rust:1.88-bookworm", add_python=PYTHON_VERSION)
    .uv_sync(UV_PROJECT_DIR, extras=["dev"], frozen=True)
    .add_local_dir(REPO_ROOT, REMOTE_REPO, copy=True, ignore=IMAGE_IGNORE)
    .workdir(REMOTE_REPO)
    .run_commands(
        "python -m maturin build --release --out /tmp/SuperMarioBros-Nes-turbo-wheels",
        "/.uv/uv pip install --python /.uv/.venv/bin/python /tmp/SuperMarioBros-Nes-turbo-wheels/*.whl",
    )
)


def benchmark_args(config: dict[str, Any]) -> list[str]:
    pairs = [
        ("--num-envs", config["num_envs"]),
        ("--steps", config["steps"]),
        ("--repeats", config["repeats"]),
        ("--warmup", config["warmup"]),
        ("--frame-skip", config["frame_skip"]),
        ("--frame-stack", config["frame_stack"]),
        ("--resize-width", config["resize_width"]),
        ("--resize-height", config["resize_height"]),
        ("--crop-top", config["crop_top"]),
        ("--crop-bottom", config["crop_bottom"]),
        ("--action", config["action"]),
        ("--pre-start-steps", config["pre_start_steps"]),
        ("--start-steps", config["start_steps"]),
        ("--post-start-steps", config["post_start_steps"]),
    ]
    result = [item for pair in pairs for item in (pair[0], str(pair[1]))]
    if config["rgb"]:
        result.append("--rgb")
    if config["include_info"]:
        result.append("--include-info")
    if config["terminate_on_flag"]:
        result.append("--terminate-on-flag")
    if config["no_start_game"]:
        result.append("--no-start-game")
    return result


def parse_states(states: str) -> list[str]:
    parsed = [state.strip() for state in states.split(",")]
    if not parsed or not all(parsed):
        raise ValueError("--states must be a comma-separated list without empty entries")
    return parsed


def stable_retro_state_dir() -> Path | None:
    try:
        import stable_retro.data  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        state_path = stable_retro.data.get_file_path(
            "SuperMarioBros-Nes-v0",
            "Level1-1.state",
            stable_retro.data.Integrations.ALL,
        )
    except Exception:
        return None
    if not state_path:
        return None
    return Path(state_path).parent


def sibling_stable_retro_state_dir() -> Path | None:
    candidate = (
        REPO_ROOT.parent
        / "stable-retro-turbo"
        / "stable_retro"
        / "data"
        / "stable"
        / "SuperMarioBros-Nes-v0"
    )
    return candidate if candidate.exists() else None


def candidate_state_dirs(state_dir: str) -> list[Path]:
    candidates: list[Path | None] = []
    if state_dir:
        candidates.append(Path(state_dir).expanduser())
    env_dir = os.environ.get("SUPERMARIOBROSNES_FASTENV_STATE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(stable_retro_state_dir())
    candidates.append(sibling_stable_retro_state_dir())

    dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            dirs.append(resolved)
            seen.add(resolved)
    return dirs


def load_state_files(states: list[str], state_dir: str) -> dict[str, bytes]:
    dirs = candidate_state_dirs(state_dir)
    files: dict[str, bytes] = {}
    for state in states:
        filename = state if state.endswith(".state") else f"{state}.state"
        for directory in dirs:
            path = directory / filename
            if path.exists():
                files[state.removesuffix(".state")] = path.read_bytes()
                break
        else:
            checked = ", ".join(str(path) for path in dirs) or "<none>"
            raise FileNotFoundError(f"could not find {filename}; checked state dirs: {checked}")
    return files


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_text(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def local_metadata(rom_path: Path, rom_bytes: bytes, state_files: dict[str, bytes]) -> dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "git": {
            "commit": git_text("rev-parse", "HEAD"),
            "status_short": git_text("status", "--short"),
        },
        "rom": {
            "local_path": str(rom_path),
            "bytes": len(rom_bytes),
            "sha256": sha256(rom_bytes),
        },
        "states": {
            name: {
                "bytes": len(state_bytes),
                "sha256": sha256(state_bytes),
            }
            for name, state_bytes in state_files.items()
        },
    }


@app.function(image=image, cpu=CPU_REQUEST, memory=MEMORY_MB, timeout=1800)
def run_benchmark(
    rom_bytes: bytes,
    state_files: dict[str, bytes],
    forwarded_args: list[str],
) -> dict[str, Any]:
    import os
    import platform
    import sys

    remote_rom = Path(REMOTE_ROM)
    remote_rom.write_bytes(rom_bytes)
    remote_state_dir = Path(REMOTE_STATE_DIR)
    remote_state_dir.mkdir(parents=True, exist_ok=True)
    for state, state_bytes in state_files.items():
        (remote_state_dir / f"{state}.state").write_bytes(state_bytes)

    modal_info = {
        "cpu_request": CPU_REQUEST,
        "memory_mb": MEMORY_MB,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "commands": {},
        "remote_repo": REMOTE_REPO,
        "remote_rom_path": str(remote_rom),
        "remote_state_dir": str(remote_state_dir),
    }

    levels: dict[str, Any] = {}
    for state in state_files:
        command = [
            "python",
            "scripts/benchmark_sps.py",
            "--rom-path",
            str(remote_rom),
            "--json",
            "--state",
            state,
            "--state-dir",
            str(remote_state_dir),
            *forwarded_args,
        ]
        proc = subprocess.run(
            command,
            cwd=REMOTE_REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "remote benchmark failed\n"
                f"state={state}\n"
                f"command={command!r}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )
        levels[state] = json.loads(proc.stdout)
        modal_info["commands"][state] = command

    return {
        "states": list(state_files),
        "levels": levels,
        "summary": aggregate_levels(levels),
        "modal": modal_info,
    }


def stat_summary(values: list[float]) -> dict[str, float]:
    import statistics

    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_levels(levels: dict[str, Any]) -> dict[str, Any]:
    per_level_means = [
        result["summary"]["env_steps_per_sec"]["mean"] for result in levels.values()
    ]
    all_runs = [
        run["env_steps_per_sec"]
        for result in levels.values()
        for run in result["runs"]
    ]
    return {
        "level_mean_env_steps_per_sec": stat_summary(per_level_means),
        "all_runs_env_steps_per_sec": stat_summary(all_runs),
        "level_count": len(levels),
        "run_count": len(all_runs),
    }


def print_summary(result: dict[str, Any]) -> None:
    modal_info = result["modal"]
    print(
        "modal="
        f"cpu_request={modal_info['cpu_request']} memory_mb={modal_info['memory_mb']} "
        f"os_cpu_count={modal_info['os_cpu_count']} "
        f"affinity_cpu_count={modal_info['affinity_cpu_count']}"
    )
    for state, level in result["levels"].items():
        config = level["config"]
        obs = level["observation"]
        summary = level["summary"]["env_steps_per_sec"]
        print(
            "level="
            f"{state} num_envs={config['num_envs']} steps={config['steps']} "
            f"repeats={config['repeats']} frame_skip={config['frame_skip']} "
            f"frame_stack={config['frame_stack']} resize={config['resize_width']}x{config['resize_height']} "
            f"action={config['action']}"
        )
        print(f"obs_shape={tuple(obs['shape'])} obs_dtype={obs['dtype']} obs_mib={obs['mib']:.2f}")
        for idx, run in enumerate(level["runs"], start=1):
            print(
                f"  run={idx} elapsed_s={run['elapsed_s']:.6f} "
                f"env_steps_per_sec={run['env_steps_per_sec']:.1f} "
                f"emulated_frames_per_sec={run['emulated_frames_per_sec']:.1f}"
            )
        print(
            "  summary="
            f"mean={summary['mean']:.1f} stdev={summary['stdev']:.1f} "
            f"best={summary['max']:.1f}"
        )
    level_mean = result["summary"]["level_mean_env_steps_per_sec"]
    all_runs = result["summary"]["all_runs_env_steps_per_sec"]
    print(
        "average="
        f"level_mean_env_steps_per_sec={level_mean['mean']:.1f} "
        f"level_mean_stdev={level_mean['stdev']:.1f} "
        f"all_runs_env_steps_per_sec={all_runs['mean']:.1f} "
        f"all_runs_stdev={all_runs['stdev']:.1f} "
        f"best_run={all_runs['max']:.1f}"
    )


@app.local_entrypoint()
def main(
    rom_path: str = str(DEFAULT_ROM),
    output_json: str = "",
    print_json: bool = False,
    num_envs: int = 16,
    steps: int = 500,
    repeats: int = 3,
    warmup: int = 100,
    frame_skip: int = 4,
    frame_stack: int = 4,
    resize_width: int = 84,
    resize_height: int = 84,
    crop_top: int = 32,
    crop_bottom: int = 0,
    action: str = "noop",
    rgb: bool = False,
    include_info: bool = False,
    terminate_on_flag: bool = False,
    no_start_game: bool = False,
    pre_start_steps: int = 30,
    start_steps: int = 8,
    post_start_steps: int = 30,
    states: str = ",".join(DEFAULT_STATES),
    state_dir: str = "",
) -> None:
    state_names = parse_states(states)
    config = {
        "num_envs": num_envs,
        "steps": steps,
        "repeats": repeats,
        "warmup": warmup,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "resize_width": resize_width,
        "resize_height": resize_height,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "action": action,
        "rgb": rgb,
        "include_info": include_info,
        "terminate_on_flag": terminate_on_flag,
        "no_start_game": no_start_game,
        "pre_start_steps": pre_start_steps,
        "start_steps": start_steps,
        "post_start_steps": post_start_steps,
    }
    local_rom_path = Path(rom_path).expanduser().resolve()
    if not local_rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {local_rom_path}")

    rom_bytes = local_rom_path.read_bytes()
    state_files = load_state_files(state_names, state_dir)
    result = run_benchmark.remote(rom_bytes, state_files, benchmark_args(config))
    result["local"] = local_metadata(local_rom_path, rom_bytes, state_files)

    output_path = Path(output_json).expanduser() if output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    if print_json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
        if output_path is not None:
            print(f"wrote_json={output_path}")
