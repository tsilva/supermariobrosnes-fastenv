from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from supermariobrosnes_turbo import ACTION_SETS, SuperMarioBrosVecEnv
from supermariobrosnes_turbo.env import DEFAULT_STABLE_RETRO_GAME


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
EXPECTED_STABLE_RETRO_VERSION = "1.0.0.post22"
INFO_KEY_MAP = {
    "coins": "coins",
    "levelHi": "level_hi",
    "levelLo": "level_lo",
    "lives": "lives",
    "score": "score",
    "scrolling": "scrolling",
    "time": "time",
    "xscrollHi": "xscroll_hi",
    "xscrollLo": "xscroll_lo",
}
ACTION_BUTTONS = {
    "noop": (),
    "right": ("RIGHT",),
    "right_b": ("RIGHT", "B"),
    "right_a": ("RIGHT", "A"),
    "right_a_b": ("RIGHT", "A", "B"),
    "a": ("A",),
    "left": ("LEFT",),
    "start": ("START",),
}


@dataclass(frozen=True)
class ComparisonConfig:
    rom_path: Path
    stable_retro_path: Path | None
    game: str
    state: str
    num_envs: int
    steps: int
    seed: int
    frame_skip: int
    frame_stack: int
    grayscale: bool
    crop_top: int
    crop_bottom: int
    resize_width: int
    resize_height: int
    action_set: str
    maxpool_last_two: bool
    terminate_on_flag: bool
    include_obs: bool
    include_rewards: bool
    include_dones: bool
    include_infos: bool
    stop_on_done: bool
    output_json: Path | None
    allow_version_mismatch: bool


class ComparisonFailure(AssertionError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(json.dumps(payload, indent=2, sort_keys=True))
        self.payload = payload


def parse_args() -> ComparisonConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compare supermariobrosnes-turbo against stable-retro-turbo "
            "RetroVecEnv on the same seeded action trace."
        ),
    )
    parser.add_argument("--rom-path", type=Path, default=DEFAULT_ROM)
    parser.add_argument(
        "--stable-retro-path",
        type=Path,
        default=None,
        help="Optional checkout/wheel-unpack path to prepend before importing stable_retro.",
    )
    parser.add_argument("--game", default=DEFAULT_STABLE_RETRO_GAME)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="simple")
    parser.add_argument(
        "--maxpool-last-two",
        action="store_true",
        help="Enable RetroVecEnv maxpooling. The fast env has no matching option yet.",
    )
    parser.add_argument(
        "--terminate-on-flag",
        action="store_true",
        help="Enable fast-env flag termination. RetroVecEnv still uses its scenario done rules.",
    )
    parser.add_argument("--skip-obs", action="store_true")
    parser.add_argument("--skip-rewards", action="store_true")
    parser.add_argument("--skip-dones", action="store_true")
    parser.add_argument("--skip-infos", action="store_true")
    parser.add_argument(
        "--no-stop-on-done",
        action="store_true",
        help="Continue after a done lane. This will usually diverge because RetroVecEnv autoresets.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help=f"Do not require stable-retro-turbo=={EXPECTED_STABLE_RETRO_VERSION}.",
    )
    args = parser.parse_args()

    positive = {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "frame_skip": args.frame_skip,
        "frame_stack": args.frame_stack,
        "resize_width": args.resize_width,
        "resize_height": args.resize_height,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.crop_top < 0 or args.crop_bottom < 0:
        parser.error("--crop-top and --crop-bottom must be non-negative")

    return ComparisonConfig(
        rom_path=args.rom_path.expanduser(),
        stable_retro_path=args.stable_retro_path.expanduser()
        if args.stable_retro_path is not None
        else None,
        game=args.game,
        state=args.state,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        grayscale=not args.rgb,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        action_set=args.action_set,
        maxpool_last_two=args.maxpool_last_two,
        terminate_on_flag=args.terminate_on_flag,
        include_obs=not args.skip_obs,
        include_rewards=not args.skip_rewards,
        include_dones=not args.skip_dones,
        include_infos=not args.skip_infos,
        stop_on_done=not args.no_stop_on_done,
        output_json=args.output_json,
        allow_version_mismatch=args.allow_version_mismatch,
    )


def maybe_prepend_stable_retro_path(path: Path | None) -> None:
    if path is None:
        return
    sys.path.insert(0, str(path))


def check_stable_retro_version(path: Path | None, allow_mismatch: bool) -> str:
    try:
        version = importlib.metadata.version("stable-retro-turbo")
    except importlib.metadata.PackageNotFoundError:
        version_path = path / "stable_retro" / "VERSION.txt" if path is not None else None
        if version_path is not None and version_path.exists():
            version = version_path.read_text(encoding="utf-8").strip()
        else:
            version = "<not installed as a distribution>"
    if version != EXPECTED_STABLE_RETRO_VERSION and not allow_mismatch:
        raise SystemExit(
            "Expected stable-retro-turbo=="
            f"{EXPECTED_STABLE_RETRO_VERSION}, found {version}. "
            "Install post22 or pass --allow-version-mismatch for checkout diagnostics."
        )
    return version


def make_fast_env(config: ComparisonConfig) -> SuperMarioBrosVecEnv:
    return SuperMarioBrosVecEnv(
        rom_path=config.rom_path,
        num_envs=config.num_envs,
        frame_skip=config.frame_skip,
        grayscale=config.grayscale,
        frame_stack=config.frame_stack,
        terminate_on_flag=config.terminate_on_flag,
        crop_top=config.crop_top,
        crop_bottom=config.crop_bottom,
        resize_width=config.resize_width,
        resize_height=config.resize_height,
        state=config.state,
        action_set=config.action_set,
    )


def make_retro_env(config: ComparisonConfig):
    import stable_retro as retro

    env = retro.RetroVecEnv(
        config.game,
        state=config.state,
        num_envs=config.num_envs,
        num_threads=config.num_envs,
        rom_path=str(config.rom_path),
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.ALL,
        obs_crop=(config.crop_top, config.crop_bottom, 0, 0),
        obs_resize=(config.resize_height, config.resize_width),
        obs_grayscale=config.grayscale,
        obs_resize_algorithm="area",
        frame_skip=config.frame_skip,
        frame_stack=config.frame_stack,
        maxpool_last_two=config.maxpool_last_two,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_mode="all",
        obs_layout="chw",
        copy_observations=True,
    )
    if hasattr(env, "seed"):
        env.seed(config.seed)
    return env


def retro_button_names(retro, rom_path: Path) -> tuple[str | None, ...]:
    system = retro.get_romfile_system(str(rom_path))
    core = retro.get_system_info(system)
    return tuple(None if name is None else str(name).upper() for name in core["buttons"])


def stable_action_masks(action_names: tuple[str, ...], buttons: tuple[str | None, ...]) -> np.ndarray:
    button_to_index = {name: index for index, name in enumerate(buttons) if name is not None}
    masks = np.zeros((len(action_names), len(buttons)), dtype=np.uint8)
    for action_index, action_name in enumerate(action_names):
        for button in ACTION_BUTTONS[action_name]:
            try:
                masks[action_index, button_to_index[button]] = 1
            except KeyError as exc:
                raise ValueError(
                    f"Retro core buttons {buttons!r} do not include required {button!r}",
                ) from exc
    return masks


def generate_action_trace(config: ComparisonConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    return rng.integers(
        0,
        len(ACTION_SETS[config.action_set]),
        size=(config.steps, config.num_envs),
        dtype=np.uint8,
    )


def array_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sum": int(np.asarray(value, dtype=np.uint64).sum()),
    }


def mismatch_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "fast": array_summary(left),
            "retro": array_summary(right),
        }
    diff = left != right
    first = tuple(int(index) for index in np.argwhere(diff)[0]) if diff.any() else None
    payload: dict[str, Any] = {
        "fast": array_summary(left),
        "retro": array_summary(right),
        "mismatch_count": int(diff.sum()),
    }
    if np.issubdtype(left.dtype, np.number):
        delta = np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)
        payload["max_abs_delta"] = int(np.abs(delta).max(initial=0))
    if first is not None:
        payload["first_mismatch_index"] = list(first)
        payload["fast_value"] = np.asarray(left)[first].item()
        payload["retro_value"] = np.asarray(right)[first].item()
    return payload


def require_array_equal(
    *,
    phase: str,
    step: int | None,
    field: str,
    fast: np.ndarray,
    retro: np.ndarray,
    action_names: list[str] | None = None,
) -> None:
    if fast.shape == retro.shape and fast.dtype == retro.dtype and np.array_equal(fast, retro):
        return
    payload: dict[str, Any] = {
        "phase": phase,
        "step": step,
        "field": field,
        "mismatch": mismatch_summary(fast, retro),
    }
    if action_names is not None:
        payload["actions"] = action_names
    raise ComparisonFailure(payload)


def fast_info_snapshot(env: SuperMarioBrosVecEnv) -> dict[str, np.ndarray]:
    return {
        retro_key: np.asarray(getattr(env, fast_attr)).copy()
        for retro_key, fast_attr in INFO_KEY_MAP.items()
    }


def retro_info_snapshot(infos: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    snapshot = {}
    for key in INFO_KEY_MAP:
        snapshot[key] = np.asarray([info.get(key) for info in infos])
    return snapshot


def compare_infos(
    *,
    phase: str,
    step: int | None,
    fast_env: SuperMarioBrosVecEnv,
    retro_infos: list[dict[str, Any]],
    action_names: list[str] | None = None,
) -> None:
    fast = fast_info_snapshot(fast_env)
    retro = retro_info_snapshot(retro_infos)
    for key, fast_values in fast.items():
        retro_values = retro[key].astype(fast_values.dtype, copy=False)
        require_array_equal(
            phase=phase,
            step=step,
            field=f"info.{key}",
            fast=fast_values,
            retro=retro_values,
            action_names=action_names,
        )


def run_comparison(config: ComparisonConfig) -> dict[str, Any]:
    maybe_prepend_stable_retro_path(config.stable_retro_path)
    stable_retro_version = check_stable_retro_version(
        config.stable_retro_path,
        config.allow_version_mismatch,
    )

    import stable_retro as retro

    buttons = retro_button_names(retro, config.rom_path)
    action_meanings = ACTION_SETS[config.action_set]
    retro_masks_by_action = stable_action_masks(action_meanings, buttons)
    action_trace = generate_action_trace(config)

    fast_env = make_fast_env(config)
    retro_env = make_retro_env(config)
    try:
        fast_obs = fast_env.reset()
        retro_obs = retro_env.reset()
        if config.include_obs:
            require_array_equal(
                phase="reset",
                step=None,
                field="obs",
                fast=fast_obs,
                retro=retro_obs,
            )
        if config.include_infos:
            compare_infos(
                phase="reset",
                step=None,
                fast_env=fast_env,
                retro_infos=list(getattr(retro_env, "reset_infos", [{}] * config.num_envs)),
            )

        compared_steps = 0
        for step, fast_actions in enumerate(action_trace):
            action_names = [action_meanings[int(action)] for action in fast_actions]
            retro_actions = retro_masks_by_action[fast_actions]

            fast_obs, fast_rewards, fast_terminated, fast_truncated, fast_infos = fast_env.step(
                fast_actions,
            )
            retro_obs, retro_rewards, retro_dones, retro_infos = retro_env.step(retro_actions)
            compared_steps += 1

            if config.include_obs:
                require_array_equal(
                    phase="step",
                    step=step,
                    field="obs",
                    fast=fast_obs,
                    retro=retro_obs,
                    action_names=action_names,
                )
            if config.include_rewards:
                require_array_equal(
                    phase="step",
                    step=step,
                    field="rewards",
                    fast=np.asarray(fast_rewards, dtype=np.float32),
                    retro=np.asarray(retro_rewards, dtype=np.float32),
                    action_names=action_names,
                )
            if config.include_dones:
                require_array_equal(
                    phase="step",
                    step=step,
                    field="dones",
                    fast=np.asarray(fast_terminated | fast_truncated, dtype=np.bool_),
                    retro=np.asarray(retro_dones, dtype=np.bool_),
                    action_names=action_names,
                )
            if config.include_infos:
                compare_infos(
                    phase="step",
                    step=step,
                    fast_env=fast_env,
                    retro_infos=retro_infos,
                    action_names=action_names,
                )
            if config.stop_on_done and (np.any(fast_terminated | fast_truncated) or np.any(retro_dones)):
                break

        return {
            "status": "ok",
            "stable_retro_version": stable_retro_version,
            "config": config_json(config),
            "retro_buttons": list(buttons),
            "action_meanings": list(action_meanings),
            "compared_steps": compared_steps,
            "final_fast_obs": array_summary(np.asarray(fast_obs)),
            "final_retro_obs": array_summary(np.asarray(retro_obs)),
        }
    finally:
        fast_env.close()
        retro_env.close()


def config_json(config: ComparisonConfig) -> dict[str, Any]:
    return {
        "rom_path": str(config.rom_path),
        "stable_retro_path": str(config.stable_retro_path)
        if config.stable_retro_path is not None
        else None,
        "game": config.game,
        "state": config.state,
        "num_envs": config.num_envs,
        "steps": config.steps,
        "seed": config.seed,
        "frame_skip": config.frame_skip,
        "frame_stack": config.frame_stack,
        "grayscale": config.grayscale,
        "crop_top": config.crop_top,
        "crop_bottom": config.crop_bottom,
        "resize_width": config.resize_width,
        "resize_height": config.resize_height,
        "action_set": config.action_set,
        "maxpool_last_two": config.maxpool_last_two,
        "terminate_on_flag": config.terminate_on_flag,
        "include_obs": config.include_obs,
        "include_rewards": config.include_rewards,
        "include_dones": config.include_dones,
        "include_infos": config.include_infos,
        "stop_on_done": config.stop_on_done,
        "allow_version_mismatch": config.allow_version_mismatch,
    }


def emit_result(result: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    if result["status"] == "ok":
        print(
            "comparison=ok "
            f"steps={result['compared_steps']} "
            f"seed={result['config']['seed']} "
            f"stable_retro_turbo={result['stable_retro_version']}",
        )
    else:
        print(text)


def main() -> None:
    config = parse_args()
    try:
        result = run_comparison(config)
    except ComparisonFailure as exc:
        result = {
            "status": "mismatch",
            "config": config_json(config),
            "failure": exc.payload,
        }
        emit_result(result, config.output_json)
        raise SystemExit(1) from None
    emit_result(result, config.output_json)


if __name__ == "__main__":
    main()
