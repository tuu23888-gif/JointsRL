"""把 EvoGym 里一个随机机器人的运动渲染成 GIF（离屏渲染，不需要开窗口）。

用法：
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_render_gif.py
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_render_gif.py --task Jumper-v0 --steps 150

产物：demos/out/<task>_random.gif
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import gymnasium as gym

import evogym.envs  # noqa: F401
from evogym import sample_robot

OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Walker-v0")
    parser.add_argument("--shape", nargs=2, type=int, default=[5, 5], metavar=("H", "W"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    body, _ = sample_robot((args.shape[0], args.shape[1]))

    env = gym.make(args.task, body=body, render_mode="rgb_array")
    env.reset(seed=args.seed)

    frames = [env.render()]
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.task}_random.gif"
    imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
    print(f"已保存 {out_path}（{len(frames)} 帧，{frames[0].shape[1]}x{frames[0].shape[0]}）")


if __name__ == "__main__":
    main()
