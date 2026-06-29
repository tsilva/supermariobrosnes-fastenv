from __future__ import annotations

from pathlib import Path

import numpy as np

from supermariobrosnes_fastenv import ACTION_MEANINGS, SuperMarioBrosVecEnv


ROM_PATH = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")


def main() -> None:
    env = SuperMarioBrosVecEnv(
        rom_path=ROM_PATH.expanduser(),
        num_envs=4,
        frame_skip=1,
        grayscale=True,
        frame_stack=1,
    )
    obs = env.reset()
    print(f"actions={ACTION_MEANINGS}")
    print(f"reset_sum={int(obs.sum())} x_pos={env.x_pos.tolist()} lives={env.lives.tolist()}")

    for _ in range(20):
        env.step_fast(np.zeros((env.num_envs,), dtype=np.uint8))
    print(
        f"after_noop_sum={int(obs.sum())} "
        f"x_pos={env.x_pos.tolist()} lives={env.lives.tolist()} "
        f"unique_pixels={len(np.unique(obs[0, 0]))}"
    )

    for _ in range(10):
        env.step_fast(np.full((env.num_envs,), ACTION_MEANINGS.index("start"), dtype=np.uint8))
    for _ in range(60):
        env.step_fast(np.zeros((env.num_envs,), dtype=np.uint8))
    print(
        f"after_start_sum={int(obs.sum())} "
        f"x_pos={env.x_pos.tolist()} lives={env.lives.tolist()} "
        f"unique_pixels={len(np.unique(obs[0, 0]))}"
    )


if __name__ == "__main__":
    main()
