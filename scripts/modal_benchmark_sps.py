from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    UV_PROJECT_DIR = str(REPO_ROOT.relative_to(Path.cwd().resolve())) or "."
except ValueError:
    UV_PROJECT_DIR = str(REPO_ROOT)
REMOTE_REPO = "/root/supermarioemu"
REMOTE_ROM = "/tmp/SuperMarioBros-Nes-v0.nes"
DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
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

app = modal.App("supermarioemu-cpu-benchmarks")

image = (
    modal.Image.from_registry("rust:1.88-bookworm", add_python=PYTHON_VERSION)
    .uv_sync(UV_PROJECT_DIR, extras=["dev"], frozen=True)
    .add_local_dir(REPO_ROOT, REMOTE_REPO, copy=True, ignore=IMAGE_IGNORE)
    .workdir(REMOTE_REPO)
    .run_commands(
        "python -m maturin build --release --out /tmp/supermarioemu-wheels",
        "/.uv/uv pip install --python /.uv/.venv/bin/python /tmp/supermarioemu-wheels/*.whl",
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


def local_metadata(rom_path: Path, rom_bytes: bytes) -> dict[str, Any]:
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
    }


@app.function(image=image, cpu=CPU_REQUEST, memory=MEMORY_MB, timeout=1800)
def run_benchmark(rom_bytes: bytes, forwarded_args: list[str]) -> dict[str, Any]:
    import os
    import platform
    import sys

    remote_rom = Path(REMOTE_ROM)
    remote_rom.write_bytes(rom_bytes)
    command = [
        "python",
        "scripts/benchmark_sps.py",
        "--rom-path",
        str(remote_rom),
        "--json",
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
            f"command={command!r}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    result = json.loads(proc.stdout)
    result["modal"] = {
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
        "command": command,
        "remote_repo": REMOTE_REPO,
        "remote_rom_path": str(remote_rom),
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    config = result["config"]
    obs = result["observation"]
    modal_info = result["modal"]
    summary = result["summary"]["env_steps_per_sec"]
    print(
        "modal="
        f"cpu_request={modal_info['cpu_request']} memory_mb={modal_info['memory_mb']} "
        f"os_cpu_count={modal_info['os_cpu_count']} "
        f"affinity_cpu_count={modal_info['affinity_cpu_count']}"
    )
    print(
        "config="
        f"num_envs={config['num_envs']} steps={config['steps']} repeats={config['repeats']} "
        f"frame_skip={config['frame_skip']} frame_stack={config['frame_stack']} "
        f"resize={config['resize_width']}x{config['resize_height']} action={config['action']}"
    )
    print(f"obs_shape={tuple(obs['shape'])} obs_dtype={obs['dtype']} obs_mib={obs['mib']:.2f}")
    for idx, run in enumerate(result["runs"], start=1):
        print(
            f"run={idx} elapsed_s={run['elapsed_s']:.6f} "
            f"env_steps_per_sec={run['env_steps_per_sec']:.1f} "
            f"emulated_frames_per_sec={run['emulated_frames_per_sec']:.1f}"
        )
    print(
        "summary="
        f"env_steps_per_sec_mean={summary['mean']:.1f} "
        f"env_steps_per_sec_stdev={summary['stdev']:.1f} "
        f"best_env_steps_per_sec={summary['max']:.1f}"
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
) -> None:
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
    result = run_benchmark.remote(rom_bytes, benchmark_args(config))
    result["local"] = local_metadata(local_rom_path, rom_bytes)

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
