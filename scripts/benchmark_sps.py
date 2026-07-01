from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from supermariobrosnes_turbo import ACTION_SETS, CORE_ACTION_MEANINGS, SuperMarioBrosVecEnv


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
DEFAULT_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark no-GUI Super Mario Bros vector-env steps per second."
    )
    parser.add_argument("--rom-path", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="simple")
    parser.add_argument("--action", choices=CORE_ACTION_MEANINGS, default="noop")
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--states",
        default=None,
        help=(
            "Comma-separated stable-retro states assigned round-robin across lanes, "
            f"default: {','.join(DEFAULT_STATES)} unless --state is provided."
        ),
    )
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--include-info", action="store_true")
    parser.add_argument("--terminate-on-flag", action="store_true")
    parser.add_argument("--no-start-game", action="store_true")
    parser.add_argument("--pre-start-steps", type=int, default=30)
    parser.add_argument("--start-steps", type=int, default=8)
    parser.add_argument("--post-start-steps", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="Enable local Rust hot-path profiling and write benchmark+profile JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "num_envs",
        "steps",
        "repeats",
        "frame_skip",
        "frame_stack",
        "resize_width",
        "resize_height",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    non_negative_fields = (
        "warmup",
        "crop_top",
        "crop_bottom",
        "pre_start_steps",
        "start_steps",
        "post_start_steps",
    )
    for field in non_negative_fields:
        if getattr(args, field) < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be non-negative")
    action_meanings = ACTION_SETS[args.action_set]
    if args.action not in action_meanings:
        raise ValueError(
            f"--action {args.action!r} is not in action_set={args.action_set!r}; "
            f"valid actions: {', '.join(action_meanings)}"
        )
    if args.state is not None and args.states is not None:
        raise ValueError("--state and --states are mutually exclusive")


def parse_states(states: str | None) -> tuple[str, ...] | None:
    if states is None:
        return None
    parsed = tuple(state.strip() for state in states.split(","))
    if not parsed or not all(parsed):
        raise ValueError("--states must be a comma-separated list without empty entries")
    return parsed


def initial_states_for_args(args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.state is not None:
        return None
    return parse_states(args.states) if args.states is not None else DEFAULT_STATES


def lane_states(num_envs: int, states: tuple[str, ...] | None) -> list[str] | None:
    if states is None:
        return None
    return [states[index % len(states)] for index in range(num_envs)]


def benchmark_state(args: argparse.Namespace) -> str | list[str] | None:
    if args.parsed_states is None:
        return args.state
    return lane_states(args.num_envs, args.parsed_states)


def has_initial_state(args: argparse.Namespace) -> bool:
    return args.state is not None or args.parsed_states is not None


def fill_action(num_envs: int, action_name: str, action_meanings: tuple[str, ...]) -> np.ndarray:
    return np.full((num_envs,), action_meanings.index(action_name), dtype=np.uint8)


def step_env(env: SuperMarioBrosVecEnv, actions: np.ndarray, include_info: bool) -> None:
    if include_info:
        env.step(actions)
    else:
        env.step_fast(actions)


def step_repeated(
    env: SuperMarioBrosVecEnv,
    actions: np.ndarray,
    count: int,
    include_info: bool,
) -> None:
    for _ in range(count):
        step_env(env, actions, include_info)


def prepare_game(
    env: SuperMarioBrosVecEnv,
    args: argparse.Namespace,
    action_meanings: tuple[str, ...],
) -> None:
    env.reset()
    if args.no_start_game or has_initial_state(args) or "start" not in action_meanings:
        return
    noop = fill_action(args.num_envs, "noop", action_meanings)
    start = fill_action(args.num_envs, "start", action_meanings)
    step_repeated(env, noop, args.pre_start_steps, args.include_info)
    step_repeated(env, start, args.start_steps, args.include_info)
    step_repeated(env, noop, args.post_start_steps, args.include_info)


def run_once(env: SuperMarioBrosVecEnv, actions: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    start = time.perf_counter()
    step_repeated(env, actions, args.steps, args.include_info)
    elapsed = time.perf_counter() - start
    batch_sps = args.steps / elapsed
    env_sps = batch_sps * args.num_envs
    frame_sps = env_sps * args.frame_skip
    return {
        "elapsed_s": elapsed,
        "batch_steps_per_sec": batch_sps,
        "env_steps_per_sec": env_sps,
        "emulated_frames_per_sec": frame_sps,
    }


def summarize(values: list[float]) -> dict[str, float]:
    result = {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }
    result["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def build_result(
    args: argparse.Namespace,
    obs: np.ndarray,
    runs: list[dict[str, float]],
    active_states: tuple[str | None, ...],
) -> dict[str, Any]:
    batch_sps = [run["batch_steps_per_sec"] for run in runs]
    env_sps = [run["env_steps_per_sec"] for run in runs]
    frame_sps = [run["emulated_frames_per_sec"] for run in runs]
    elapsed = [run["elapsed_s"] for run in runs]
    mean_batch_sps = statistics.fmean(batch_sps)
    return {
        "config": {
            "num_envs": args.num_envs,
            "steps": args.steps,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack,
            "grayscale": not args.rgb,
            "crop_top": args.crop_top,
            "crop_bottom": args.crop_bottom,
            "resize_width": args.resize_width,
            "resize_height": args.resize_height,
            "action_set": args.action_set,
            "action": args.action,
            "state": args.state,
            "states": list(args.parsed_states) if args.parsed_states is not None else None,
            "lane_states": list(active_states) if has_initial_state(args) else None,
            "state_dir": str(args.state_dir) if args.state_dir is not None else None,
            "include_info": args.include_info,
            "terminate_on_flag": args.terminate_on_flag,
            "start_game": (
                not args.no_start_game
                and not has_initial_state(args)
                and "start" in ACTION_SETS[args.action_set]
            ),
        },
        "observation": {
            "shape": list(obs.shape),
            "dtype": str(obs.dtype),
            "bytes": int(obs.nbytes),
            "mib": obs.nbytes / (1024**2),
        },
        "runs": runs,
        "summary": {
            "elapsed_s": summarize(elapsed),
            "batch_steps_per_sec": summarize(batch_sps),
            "env_steps_per_sec": summarize(env_sps),
            "emulated_frames_per_sec": summarize(frame_sps),
            "obs_buffer_gib_per_sec": (obs.nbytes * mean_batch_sps) / (1024**3),
        },
    }


def print_human(result: dict[str, Any]) -> None:
    config = result["config"]
    obs = result["observation"]
    summary = result["summary"]
    print(
        "config="
        f"num_envs={config['num_envs']} steps={config['steps']} repeats={config['repeats']} "
        f"frame_skip={config['frame_skip']} frame_stack={config['frame_stack']} "
        f"grayscale={config['grayscale']} crop=({config['crop_top']},{config['crop_bottom']}) "
        f"resize={config['resize_width']}x{config['resize_height']} "
        f"action_set={config['action_set']} action={config['action']} "
        f"state={config['state']} states={config['states']} "
        f"include_info={config['include_info']}"
    )
    if config["lane_states"] is not None:
        print(f"lane_states={config['lane_states']}")
    print(
        f"obs_shape={tuple(obs['shape'])} obs_dtype={obs['dtype']} "
        f"obs_mib={obs['mib']:.2f}"
    )
    for idx, run in enumerate(result["runs"], start=1):
        print(
            f"run={idx} elapsed_s={run['elapsed_s']:.6f} "
            f"batch_steps_per_sec={run['batch_steps_per_sec']:.1f} "
            f"env_steps_per_sec={run['env_steps_per_sec']:.1f} "
            f"emulated_frames_per_sec={run['emulated_frames_per_sec']:.1f}"
        )
    print(
        "summary="
        f"env_steps_per_sec_mean={summary['env_steps_per_sec']['mean']:.1f} "
        f"env_steps_per_sec_stdev={summary['env_steps_per_sec']['stdev']:.1f} "
        f"best_env_steps_per_sec={summary['env_steps_per_sec']['max']:.1f} "
        f"emulated_frames_per_sec_mean={summary['emulated_frames_per_sec']['mean']:.1f} "
        f"obs_buffer_gib_per_sec={summary['obs_buffer_gib_per_sec']:.2f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.parsed_states = initial_states_for_args(args)
    action_set = args.action_set
    action_meanings = ACTION_SETS[action_set]
    env = SuperMarioBrosVecEnv(
        rom_path=args.rom_path.expanduser(),
        num_envs=args.num_envs,
        frame_skip=args.frame_skip,
        grayscale=not args.rgb,
        frame_stack=args.frame_stack,
        terminate_on_flag=args.terminate_on_flag,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        state=benchmark_state(args),
        state_dir=args.state_dir,
        action_set=action_set,
    )
    if args.profile_output is not None:
        env.enable_profiler()
    obs = env.reset()
    active_states = env.active_states()
    actions = fill_action(args.num_envs, args.action, action_meanings)
    prepare_game(env, args, action_meanings)
    step_repeated(env, actions, args.warmup, args.include_info)
    if args.profile_output is not None:
        env.reset_profiler()
    runs = [run_once(env, actions, args) for _ in range(args.repeats)]
    result = build_result(args, obs, runs, active_states)
    if args.profile_output is not None:
        result["profiler"] = env.profiler_snapshot()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.profile_output is not None:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
