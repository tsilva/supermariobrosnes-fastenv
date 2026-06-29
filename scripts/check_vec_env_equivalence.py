from __future__ import annotations

from pathlib import Path

import numpy as np

from supermarioemu import ACTION_MEANINGS, SuperMarioBrosVecEnv


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")


def make_env(rom_path: Path, num_envs: int) -> SuperMarioBrosVecEnv:
    return SuperMarioBrosVecEnv(
        rom_path=rom_path.expanduser(),
        num_envs=num_envs,
        frame_skip=4,
        grayscale=True,
        frame_stack=4,
        terminate_on_flag=False,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
    )


def step_uniform(
    env: SuperMarioBrosVecEnv, action: int, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = np.full((env.num_envs,), action, dtype=np.uint8)
    result = None
    for _ in range(count):
        result = env.step_fast(actions)
    assert result is not None
    return result


def check_uniform_sync(rom_path: Path) -> None:
    noop = ACTION_MEANINGS.index("noop")
    start = ACTION_MEANINGS.index("start")
    right = ACTION_MEANINGS.index("right")

    vec = make_env(rom_path, 16)
    one = make_env(rom_path, 1)
    obs_vec = vec.reset()
    obs_one = one.reset()
    assert obs_vec.shape == (16, 4, 84, 84)
    assert obs_vec.dtype == np.uint8
    np.testing.assert_array_equal(obs_vec[0], obs_one[0])

    for action, count in (
        (noop, 30),
        (start, 8),
        (noop, 30),
        (noop, 20),
        (right, 5),
        (noop, 10),
    ):
        vec_obs, vec_rewards, vec_terminated, vec_truncated = step_uniform(vec, action, count)
        one_obs, one_rewards, one_terminated, one_truncated = step_uniform(one, action, count)
        np.testing.assert_array_equal(vec_obs[0], one_obs[0])
        np.testing.assert_array_equal(vec_rewards, np.full((16,), one_rewards[0], dtype=np.float32))
        np.testing.assert_array_equal(
            vec_terminated, np.full((16,), one_terminated[0], dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            vec_truncated, np.full((16,), one_truncated[0], dtype=np.bool_)
        )
        np.testing.assert_array_equal(vec.x_pos, np.full((16,), one.x_pos[0], dtype=np.uint16))
        np.testing.assert_array_equal(vec.lives, np.full((16,), one.lives[0], dtype=np.uint8))
        for lane in range(1, 16):
            np.testing.assert_array_equal(vec_obs[0], vec_obs[lane])


def check_divergence_materializes_independent_lanes(rom_path: Path) -> None:
    noop = ACTION_MEANINGS.index("noop")
    start = ACTION_MEANINGS.index("start")
    right = ACTION_MEANINGS.index("right")
    a_button = ACTION_MEANINGS.index("a")

    vec = make_env(rom_path, 8)
    refs = [make_env(rom_path, 1) for _ in range(8)]
    vec.reset()
    for ref in refs:
        ref.reset()

    for action, count in ((noop, 30), (start, 8), (noop, 30), (noop, 5)):
        step_uniform(vec, action, count)
        for ref in refs:
            step_uniform(ref, action, count)

    actions = np.array([noop, right, a_button, start, noop, right, a_button, start], dtype=np.uint8)
    obs, rewards, terminated, truncated = vec.step_fast(actions)
    for lane, ref in enumerate(refs):
        ref_obs, ref_rewards, ref_terminated, ref_truncated = ref.step_fast(
            np.asarray([int(actions[lane])], dtype=np.uint8)
        )
        np.testing.assert_array_equal(obs[lane], ref_obs[0])
        assert rewards[lane] == ref_rewards[0]
        assert terminated[lane] == ref_terminated[0]
        assert truncated[lane] == ref_truncated[0]
        assert vec.x_pos[lane] == ref.x_pos[0]
        assert vec.lives[lane] == ref.lives[0]


def main() -> None:
    rom_path = DEFAULT_ROM.expanduser()
    check_uniform_sync(rom_path)
    check_divergence_materializes_independent_lanes(rom_path)
    print("equivalence=ok obs_shape=(16, 4, 84, 84) obs_dtype=uint8")


if __name__ == "__main__":
    main()
