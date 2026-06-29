from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from ._supermariobrosnes_fastenv import FastMarioVecEnv

ACTION_MEANINGS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left", "start")
DEFAULT_STABLE_RETRO_GAME = "SuperMarioBros-Nes-v0"
GZIP_MAGIC = b"\x1f\x8b"


def _expand_rom_path(path: str | Path) -> str:
    return str(Path(path).expanduser())


def _stable_retro_state_dir() -> Path | None:
    try:
        import stable_retro.data  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        state_path = stable_retro.data.get_file_path(
            DEFAULT_STABLE_RETRO_GAME,
            "Level1-1.state",
            stable_retro.data.Integrations.ALL,
        )
    except Exception:
        return None
    if not state_path:
        return None
    return Path(state_path).parent


def _sibling_stable_retro_state_dir() -> Path | None:
    game_path = Path("stable_retro/data/stable") / DEFAULT_STABLE_RETRO_GAME
    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "stable-retro-turbo" / game_path
        if candidate.exists():
            return candidate
    return None


def _candidate_state_dirs(state_dir: str | Path | None = None) -> list[Path]:
    candidates: list[Path | None] = []
    if state_dir is not None:
        candidates.append(Path(state_dir).expanduser())
    env_dir = os.environ.get("SUPERMARIOBROSNES_FASTENV_STATE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(_stable_retro_state_dir())
    candidates.append(_sibling_stable_retro_state_dir())

    dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved not in seen and resolved.exists():
            dirs.append(resolved)
            seen.add(resolved)
    return dirs


def list_available_states(state_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return available stable-retro Super Mario Bros state names."""
    states: set[str] = set()
    for directory in _candidate_state_dirs(state_dir):
        states.update(
            path.stem for path in directory.glob("*.state") if not path.name.startswith("_")
        )
    return tuple(sorted(states))


def _resolve_state_path(state: str | Path, state_dir: str | Path | None = None) -> Path:
    raw_path = Path(state).expanduser()
    if raw_path.exists():
        return raw_path

    state_name = str(state)
    state_file = state_name if state_name.endswith(".state") else f"{state_name}.state"
    for directory in _candidate_state_dirs(state_dir):
        candidate = directory / state_file
        if candidate.exists():
            return candidate

    dirs = ", ".join(str(path) for path in _candidate_state_dirs(state_dir)) or "<none>"
    raise FileNotFoundError(
        f"could not resolve state {state_name!r}; checked direct path and state dirs: {dirs}"
    )


def _load_initial_state(
    state: str | Path | bytes | bytearray | memoryview | None,
    state_dir: str | Path | None = None,
) -> bytes | None:
    if state is None:
        return None
    if isinstance(state, (bytes, bytearray, memoryview)):
        raw = bytes(state)
    else:
        raw = _resolve_state_path(state, state_dir).read_bytes()
    if raw.startswith(GZIP_MAGIC):
        return gzip.decompress(raw)
    return raw


class SuperMarioBrosVecEnv:
    """Vectorized Mario environment with the hot loop in Rust.

    The important API is `step_wait()`: it performs one Python/Rust crossing for
    the whole batch, with frame skip, grayscale, and frame stacking already done
    before the observation buffer reaches Python.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rom_path: str | Path = "~/Desktop/roms/SuperMarioBros.nes",
        num_envs: int = 1,
        frame_skip: int = 4,
        grayscale: bool = True,
        frame_stack: int = 4,
        terminate_on_flag: bool = True,
        crop_top: int = 0,
        crop_bottom: int = 0,
        resize_width: int = 84,
        resize_height: int = 84,
        state: str | Path | bytes | bytearray | memoryview | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        initial_state = _load_initial_state(state, state_dir)
        self._has_initial_state = initial_state is not None
        self._core = FastMarioVecEnv(
            _expand_rom_path(rom_path),
            num_envs,
            frame_skip,
            grayscale,
            frame_stack,
            terminate_on_flag,
            crop_top,
            crop_bottom,
            resize_width,
            resize_height,
            initial_state,
        )
        self.num_envs = self._core.num_envs
        self.frame_skip = self._core.frame_skip
        self.grayscale = self._core.grayscale
        self.frame_stack = self._core.frame_stack
        self.terminate_on_flag = terminate_on_flag
        self.crop_top = self._core.crop_top
        self.crop_bottom = self._core.crop_bottom
        self.resize_width = self._core.resize_width
        self.resize_height = self._core.resize_height
        self.single_action_space = spaces.Discrete(len(ACTION_MEANINGS))
        self.action_space = spaces.MultiDiscrete([len(ACTION_MEANINGS)] * self.num_envs)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self._core.obs_shape()[1:],
            dtype=np.uint8,
        )

        self._actions = np.zeros((self.num_envs,), dtype=np.uint8)
        self._obs = np.empty(self._core.obs_shape(), dtype=np.uint8)
        self._rewards = np.empty((self.num_envs,), dtype=np.float32)
        self._terminated = np.empty((self.num_envs,), dtype=np.bool_)
        self._truncated = np.empty((self.num_envs,), dtype=np.bool_)
        self._x_pos = np.empty((self.num_envs,), dtype=np.uint16)
        self._lives = np.empty((self.num_envs,), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        self._core.reset_into(self._obs)
        self._rewards.fill(0)
        self._terminated.fill(False)
        self._truncated.fill(False)
        if self._has_initial_state:
            self._core.info_into(self._x_pos, self._lives)
        else:
            self._x_pos.fill(0)
            self._lives.fill(3)
        return self._obs

    def step_async(self, actions: np.ndarray) -> None:
        actions_arr = np.asarray(actions, dtype=np.uint8)
        if actions_arr.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape {(self.num_envs,)}, got {actions_arr.shape}")
        np.copyto(self._actions, actions_arr)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, terminated, truncated = self.step_wait_fast()
        infos = [{"x_pos": int(x), "lives": int(l)} for x, l in zip(self._x_pos, self._lives)]
        return obs, rewards, terminated, truncated, infos

    def step_wait_fast(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step the whole batch without allocating per-env info dictionaries."""
        self._core.step_into(
            self._actions,
            self._obs,
            self._rewards,
            self._terminated,
            self._truncated,
            self._x_pos,
            self._lives,
        )
        return self._obs, self._rewards, self._terminated, self._truncated

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait()

    def step_fast(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.step_async(actions)
        return self.step_wait_fast()

    @property
    def x_pos(self) -> np.ndarray:
        return self._x_pos

    @property
    def lives(self) -> np.ndarray:
        return self._lives

    def close(self) -> None:
        pass
