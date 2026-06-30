from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from supermariobrosnes_turbo import ACTION_SETS, SuperMarioBrosVecEnv
from supermariobrosnes_turbo.env import DEFAULT_STABLE_RETRO_GAME


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
EXPECTED_STABLE_RETRO_VERSION = "1.0.0.post23"
STABLE_VISIBLE_WIDTH = 240
STABLE_VISIBLE_HEIGHT = 224
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
SANDBOX_SB3_LEVEL1_1_ENVIRONMENT_HASH = (
    "sha256:f5000bb13abcc81d000892b6b0d5ebb7fb101f729859af9ec8ca524a5b2b02f8"
)
SANDBOX_SB3_LEVEL1_1_DONE_ON = {
    "life_loss": ("lives", "decrease"),
    "level_change": (("levelHi", "levelLo"), "change"),
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
    env_threads: int
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
    frame_maxpool: bool
    obs_copy: str
    terminate_on_flag: bool
    terminate_on_life_loss: bool
    terminate_on_level_change: bool
    include_obs: bool
    include_rewards: bool
    include_dones: bool
    include_infos: bool
    stop_on_done: bool
    output_json: Path | None
    allow_version_mismatch: bool
    preprocessing_matrix: bool


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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Vector env count. Default matches sandbox-sb3 Level1-1 training.",
    )
    parser.add_argument(
        "--env-threads",
        type=int,
        default=4,
        help="RetroVecEnv worker threads. Default matches sandbox-sb3 Level1-1 training.",
    )
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
        "--frame-maxpool",
        action="store_true",
        help=(
            "Enable RetroVecEnv maxpooling. sandbox-sb3 Level1-1 training leaves this disabled."
        ),
    )
    parser.add_argument(
        "--obs-copy",
        choices=("copy", "safe_view", "unsafe_view"),
        default="safe_view",
        help="RetroVecEnv observation ownership mode. sandbox-sb3 Level1-1 training uses safe_view.",
    )
    parser.add_argument(
        "--terminate-on-flag",
        action="store_true",
        help="Enable fast-env flag termination. RetroVecEnv still uses its scenario done rules.",
    )
    parser.add_argument(
        "--no-terminate-on-life-loss",
        dest="terminate_on_life_loss",
        action="store_false",
        default=True,
        help="Disable the fast-env equivalent of sandbox-sb3's life_loss done_on rule.",
    )
    parser.add_argument(
        "--no-terminate-on-level-change",
        dest="terminate_on_level_change",
        action="store_false",
        default=True,
        help="Disable the fast-env equivalent of sandbox-sb3's level_change done_on rule.",
    )
    parser.add_argument("--skip-obs", action="store_true")
    parser.add_argument("--skip-rewards", action="store_true")
    parser.add_argument("--skip-dones", action="store_true")
    parser.add_argument("--skip-infos", action="store_true")
    parser.set_defaults(stop_on_done=False)
    parser.add_argument(
        "--stop-on-done",
        dest="stop_on_done",
        action="store_true",
        help="Stop after the first done lane instead of comparing all requested steps.",
    )
    parser.add_argument(
        "--no-stop-on-done",
        dest="stop_on_done",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help=f"Do not require stable-retro-turbo=={EXPECTED_STABLE_RETRO_VERSION}.",
    )
    parser.add_argument(
        "--preprocessing-matrix",
        action="store_true",
        help="Run obs-only comparisons for raw RGB, grayscale, cropped, and resized obs.",
    )
    args = parser.parse_args()

    positive = {
        "num_envs": args.num_envs,
        "env_threads": args.env_threads,
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
        env_threads=args.env_threads,
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
        frame_maxpool=args.frame_maxpool,
        obs_copy=args.obs_copy,
        terminate_on_flag=args.terminate_on_flag,
        terminate_on_life_loss=args.terminate_on_life_loss,
        terminate_on_level_change=args.terminate_on_level_change,
        include_obs=not args.skip_obs,
        include_rewards=not args.skip_rewards,
        include_dones=not args.skip_dones,
        include_infos=not args.skip_infos,
        stop_on_done=args.stop_on_done,
        output_json=args.output_json,
        allow_version_mismatch=args.allow_version_mismatch,
        preprocessing_matrix=args.preprocessing_matrix,
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
            "Install post23 or pass --allow-version-mismatch for checkout diagnostics."
        )
    return version


def install_sb3_vecenv_shim_if_needed() -> None:
    if "stable_baselines3.common.vec_env" in sys.modules:
        return
    try:
        has_vec_env = importlib.util.find_spec("stable_baselines3.common.vec_env") is not None
    except (ModuleNotFoundError, ValueError):
        has_vec_env = False
    if has_vec_env:
        return

    stable_baselines3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")

    class VecEnv:
        def __init__(self, num_envs: int, observation_space: Any, action_space: Any) -> None:
            self.num_envs = int(num_envs)
            self.observation_space = observation_space
            self.action_space = action_space
            self._seeds = [None for _ in range(self.num_envs)]
            self._options = [{} for _ in range(self.num_envs)]
            self.reset_infos = [{} for _ in range(self.num_envs)]

        def seed(self, seed: int | None = None) -> list[int | None]:
            self._seeds = (
                [None for _ in range(self.num_envs)]
                if seed is None
                else [int(seed) + index for index in range(self.num_envs)]
            )
            return list(self._seeds)

        def step(self, actions: Any):
            self.step_async(actions)
            return self.step_wait()

        def _reset_seeds(self) -> None:
            self._seeds = [None for _ in range(self.num_envs)]

        def _reset_options(self) -> None:
            self._options = [{} for _ in range(self.num_envs)]

        def _get_indices(self, indices: Any = None) -> list[int]:
            if indices is None:
                return list(range(self.num_envs))
            if isinstance(indices, int):
                return [indices]
            return [int(index) for index in indices]

    vec_env.VecEnv = VecEnv
    common.vec_env = vec_env
    stable_baselines3.common = common
    sys.modules["stable_baselines3"] = stable_baselines3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.vec_env"] = vec_env


def make_fast_env(config: ComparisonConfig) -> SuperMarioBrosVecEnv:
    return SuperMarioBrosVecEnv(
        rom_path=config.rom_path,
        num_envs=config.num_envs,
        frame_skip=config.frame_skip,
        grayscale=config.grayscale,
        frame_stack=config.frame_stack,
        terminate_on_flag=config.terminate_on_flag,
        terminate_on_life_loss=config.terminate_on_life_loss,
        terminate_on_level_change=config.terminate_on_level_change,
        crop_top=config.crop_top,
        crop_bottom=config.crop_bottom,
        resize_width=config.resize_width,
        resize_height=config.resize_height,
        state=config.state,
        action_set=config.action_set,
        seed=config.seed,
    )


def make_retro_env(config: ComparisonConfig):
    import stable_retro as retro

    source_height = STABLE_VISIBLE_HEIGHT - config.crop_top - config.crop_bottom
    obs_crop = None
    if config.crop_top != 0 or config.crop_bottom != 0:
        obs_crop = (config.crop_top, config.crop_bottom, 0, 0)
    obs_resize = None
    if config.resize_width != STABLE_VISIBLE_WIDTH or config.resize_height != source_height:
        obs_resize = (config.resize_height, config.resize_width)
    kwargs = {
        "state": config.state,
        "num_envs": config.num_envs,
        "num_threads": config.env_threads,
        "rom_path": str(config.rom_path),
        "render_mode": "rgb_array",
        "use_restricted_actions": retro.Actions.ALL,
        "obs_crop": obs_crop,
        "obs_resize": obs_resize,
        "obs_grayscale": config.grayscale,
        "obs_resize_algorithm": "area",
        "frame_skip": config.frame_skip,
        "frame_stack": config.frame_stack,
        "frame_maxpool": config.frame_maxpool,
        "reset_noops": 0,
        "action_sticky_prob": 0.0,
        "reward_clip": False,
        "info_filter": "all",
        "obs_layout": "chw",
        "obs_copy": config.obs_copy,
    }
    kwargs["done_on"] = SANDBOX_SB3_LEVEL1_1_DONE_ON
    env = retro.RetroVecEnv(config.game, **kwargs)
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


def jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    payload: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(array.dtype),
    }
    try:
        payload["sum"] = int(np.asarray(value, dtype=np.uint64).sum())
    except (TypeError, ValueError):
        payload["values_sample"] = [
            jsonable_scalar(item) for item in array.reshape(-1)[: min(array.size, 8)]
        ]
    return payload


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


def fast_infos_from_env(env: SuperMarioBrosVecEnv) -> list[dict[str, Any]]:
    snapshot = fast_info_snapshot(env)
    return [
        {key: values[index].item() for key, values in snapshot.items()}
        for index in range(env.num_envs)
    ]


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
        try:
            retro_values = retro[key].astype(fast_values.dtype, copy=False)
        except (TypeError, ValueError):
            retro_values = retro[key]
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
    install_sb3_vecenv_shim_if_needed()

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

        compared_steps = 0
        for step, fast_actions in enumerate(action_trace):
            action_names = [action_meanings[int(action)] for action in fast_actions]
            retro_actions = retro_masks_by_action[fast_actions]

            fast_obs, fast_rewards, fast_terminated, fast_truncated, fast_infos = fast_env.step(
                fast_actions,
            )
            retro_obs, retro_rewards, retro_dones, retro_infos = retro_env.step(retro_actions)
            fast_native_dones = np.asarray(fast_terminated | fast_truncated, dtype=np.bool_)
            retro_dones = np.asarray(retro_dones, dtype=np.bool_)
            fast_compare_rewards = np.asarray(fast_rewards, dtype=np.float32)
            retro_compare_rewards = np.asarray(retro_rewards, dtype=np.float32)
            fast_compare_dones = fast_native_dones
            retro_compare_dones = retro_dones
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
                    fast=fast_compare_rewards,
                    retro=retro_compare_rewards,
                    action_names=action_names,
                )
            if config.include_dones:
                require_array_equal(
                    phase="step",
                    step=step,
                    field="dones",
                    fast=fast_compare_dones,
                    retro=retro_compare_dones,
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
            if config.stop_on_done and (np.any(fast_compare_dones) or np.any(retro_compare_dones)):
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


def preprocessing_matrix_configs(config: ComparisonConfig) -> list[tuple[str, ComparisonConfig]]:
    common = {
        "include_obs": True,
        "include_rewards": False,
        "include_dones": False,
        "include_infos": False,
    }
    return [
        (
            "rgb_visible_no_crop_no_resize",
            replace(
                config,
                grayscale=False,
                crop_top=0,
                crop_bottom=0,
                resize_width=STABLE_VISIBLE_WIDTH,
                resize_height=STABLE_VISIBLE_HEIGHT,
                **common,
            ),
        ),
        (
            "gray_visible_no_crop_no_resize",
            replace(
                config,
                grayscale=True,
                crop_top=0,
                crop_bottom=0,
                resize_width=STABLE_VISIBLE_WIDTH,
                resize_height=STABLE_VISIBLE_HEIGHT,
                **common,
            ),
        ),
        (
            "gray_crop_no_resize",
            replace(
                config,
                grayscale=True,
                resize_width=STABLE_VISIBLE_WIDTH,
                resize_height=STABLE_VISIBLE_HEIGHT - config.crop_top - config.crop_bottom,
                **common,
            ),
        ),
        (
            "gray_crop_resize",
            replace(
                config,
                grayscale=True,
                **common,
            ),
        ),
    ]


def run_preprocessing_matrix(config: ComparisonConfig) -> dict[str, Any]:
    results = []
    status = "ok"
    for name, matrix_config in preprocessing_matrix_configs(config):
        try:
            result = run_comparison(matrix_config)
            results.append(
                {
                    "name": name,
                    "status": "ok",
                    "compared_steps": result["compared_steps"],
                    "config": result["config"],
                    "final_fast_obs": result["final_fast_obs"],
                    "final_retro_obs": result["final_retro_obs"],
                },
            )
        except ComparisonFailure as exc:
            status = "mismatch"
            results.append(
                {
                    "name": name,
                    "status": "mismatch",
                    "config": config_json(matrix_config),
                    "failure": exc.payload,
                },
            )
    return {
        "status": status,
        "mode": "preprocessing_matrix",
        "config": config_json(config),
        "results": results,
    }


def config_json(config: ComparisonConfig) -> dict[str, Any]:
    return {
        "rom_path": str(config.rom_path),
        "stable_retro_path": str(config.stable_retro_path)
        if config.stable_retro_path is not None
        else None,
        "game": config.game,
        "state": config.state,
        "num_envs": config.num_envs,
        "env_threads": config.env_threads,
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
        "frame_maxpool": config.frame_maxpool,
        "obs_copy": config.obs_copy,
        "terminate_on_flag": config.terminate_on_flag,
        "terminate_on_life_loss": config.terminate_on_life_loss,
        "terminate_on_level_change": config.terminate_on_level_change,
        "sandbox_sb3_level1_1_native_retro_vec_env_profile": {
            "environment_hash": SANDBOX_SB3_LEVEL1_1_ENVIRONMENT_HASH,
            "game": DEFAULT_STABLE_RETRO_GAME,
            "state": "Level1-1",
            "num_envs": 16,
            "env_threads": 4,
            "frame_skip": 4,
            "frame_stack": 4,
            "frame_maxpool": False,
            "obs_copy": "safe_view",
            "obs_crop": [32, 0, 0, 0],
            "obs_grayscale": True,
            "obs_resize": [84, 84],
            "obs_resize_algorithm": "area",
            "obs_layout": "chw",
            "reset_noops": 0,
            "action_sticky_prob": 0.0,
            "action_set": "simple",
            "info_filter": "all",
            "done_on": {
                "life_loss": ["lives", "decrease"],
                "level_change": [["levelHi", "levelLo"], "change"],
            },
        },
        "include_obs": config.include_obs,
        "include_rewards": config.include_rewards,
        "include_dones": config.include_dones,
        "include_infos": config.include_infos,
        "stop_on_done": config.stop_on_done,
        "allow_version_mismatch": config.allow_version_mismatch,
        "preprocessing_matrix": config.preprocessing_matrix,
    }


def emit_result(result: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    if result["status"] == "ok":
        if result.get("mode") == "preprocessing_matrix":
            steps = ",".join(
                f"{item['name']}:{item['compared_steps']}" for item in result["results"]
            )
            print(
                "comparison_matrix=ok "
                f"steps={steps} "
                f"seed={result['config']['seed']}",
            )
        else:
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
    if config.preprocessing_matrix:
        result = run_preprocessing_matrix(config)
        emit_result(result, config.output_json)
        if result["status"] != "ok":
            raise SystemExit(1)
        return
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
