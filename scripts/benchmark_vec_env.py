from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from supermariobrosnes_fastenv import SuperMarioBrosVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes"),
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--state", default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--action", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SuperMarioBrosVecEnv(
        rom_path=args.rom_path.expanduser(),
        num_envs=args.num_envs,
        frame_skip=args.frame_skip,
        grayscale=not args.rgb,
        frame_stack=args.frame_stack,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        state=args.state,
        state_dir=args.state_dir,
    )
    obs = env.reset()
    actions = np.full((args.num_envs,), args.action, dtype=np.uint8)

    for _ in range(args.warmup):
        env.step_fast(actions)

    start = time.perf_counter()
    for _ in range(args.steps):
        env.step_fast(actions)
    elapsed = time.perf_counter() - start

    batch_sps = args.steps / elapsed
    env_sps = batch_sps * args.num_envs
    frame_sps = env_sps * args.frame_skip
    obs_gib_per_s = (obs.nbytes * batch_sps) / (1024**3)

    print(f"obs_shape={obs.shape} obs_dtype={obs.dtype} obs_mib={obs.nbytes / (1024**2):.2f}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"batch_steps_per_sec={batch_sps:.1f}")
    print(f"env_steps_per_sec={env_sps:.1f}")
    print(f"emulated_frames_per_sec={frame_sps:.1f}")
    print(f"obs_buffer_gib_per_sec={obs_gib_per_s:.2f}")


if __name__ == "__main__":
    main()
