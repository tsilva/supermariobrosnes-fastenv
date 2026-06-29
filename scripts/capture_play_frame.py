from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from play import action_id, latest_frame, png_from_frame
from supermariobrosnes_fastenv import SuperMarioBrosVecEnv


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom-path", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--output", type=Path, default=Path("artifacts/play-frame.png"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--pre-start-frames", type=int, default=120)
    parser.add_argument("--start-frames", type=int, default=30)
    parser.add_argument("--right-frames", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SuperMarioBrosVecEnv(
        rom_path=args.rom_path.expanduser(),
        num_envs=1,
        frame_skip=1,
        grayscale=False,
        frame_stack=1,
        resize_width=256,
        resize_height=240,
    )
    obs = env.reset()[0]

    def step_one(action_name: str) -> np.ndarray:
        actions = np.asarray([action_id(action_name)], dtype=np.uint8)
        return env.step_fast(actions)[0][0]

    for _ in range(args.pre_start_frames):
        obs = step_one("noop")
    for _ in range(args.start_frames):
        obs = step_one("start")
    for _ in range(60):
        obs = step_one("noop")
    for _ in range(args.right_frames):
        obs = step_one("right_b")

    frame = latest_frame(obs)
    if args.scale > 1:
        frame = np.repeat(np.repeat(frame, args.scale, axis=0), args.scale, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png_from_frame(np.ascontiguousarray(frame)))
    print(args.output)


if __name__ == "__main__":
    main()
