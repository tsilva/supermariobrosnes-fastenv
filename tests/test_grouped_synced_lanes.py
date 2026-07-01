from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from supermariobrosnes_turbo import ACTION_MEANINGS, SuperMarioBrosVecEnv


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
GROUP_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")


def require_rom() -> Path:
    rom_path = DEFAULT_ROM.expanduser()
    if not rom_path.exists():
        pytest.skip(f"local SuperMarioBros-Nes ROM is missing: {rom_path}")
    return rom_path


def make_env(rom_path: Path, state: str | list[str], num_envs: int) -> SuperMarioBrosVecEnv:
    return SuperMarioBrosVecEnv(
        rom_path=rom_path,
        num_envs=num_envs,
        frame_skip=4,
        frame_stack=4,
        grayscale=True,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
        state=state,
        terminate_on_flag=False,
    )


def assert_fast_step_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_repeated_state_groups_match_independent_lane_references() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    grouped = make_env(rom_path, lane_states, num_envs=16)
    refs = [make_env(rom_path, state, num_envs=1) for state in GROUP_STATES]

    grouped_obs = grouped.reset()
    ref_obs = [ref.reset() for ref in refs]
    for lane, state in enumerate(lane_states):
        ref_index = GROUP_STATES.index(state)
        np.testing.assert_array_equal(grouped_obs[lane], ref_obs[ref_index][0])

    noop = ACTION_MEANINGS.index("noop")
    right = ACTION_MEANINGS.index("right")
    for action_id in (noop, noop, right, noop):
        grouped_result = grouped.step_fast(np.full((16,), action_id, dtype=np.uint8))
        ref_results = [
            ref.step_fast(np.asarray([action_id], dtype=np.uint8)) for ref in refs
        ]
        for lane, state in enumerate(lane_states):
            ref_index = GROUP_STATES.index(state)
            for actual_array, expected_array in zip(grouped_result, ref_results[ref_index], strict=True):
                np.testing.assert_array_equal(actual_array[lane], expected_array[0])


def test_grouped_lanes_materialize_before_divergent_actions() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    grouped = make_env(rom_path, lane_states, num_envs=16)
    independent = make_env(rom_path, lane_states, num_envs=16)
    grouped.reset()
    independent.reset()

    noop = ACTION_MEANINGS.index("noop")
    right = ACTION_MEANINGS.index("right")
    actions = np.full((16,), noop, dtype=np.uint8)
    actions[4] = right

    assert_fast_step_equal(grouped.step_fast(actions), independent.step_fast(actions))
    assert_fast_step_equal(grouped.step_fast(actions), independent.step_fast(actions))
