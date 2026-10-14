"""EvoGym 最小可运行示例：随机策略跑完若干任务的一个 episode。

用法（必须用 Python 3.10 环境，EvoGym 只支持 3.7~3.10）：
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_quickstart.py
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_quickstart.py --tasks Walker-v0 Jumper-v0

说明：
  * body 是体素矩阵 (h, w)，取值 {0:空, 1:刚性, 2:软体, 3:水平驱动器, 4:垂直驱动器}
  * 动作空间 = 每个驱动器体素的目标形变比例，范围 [0.6, 1.6]
  * 观测 = 体素角点相对质心位置 + 质心速度/姿态 + 地形窗口 + 任务相关量
"""

from __future__ import annotations

import argparse
import random
from importlib.metadata import version

import numpy as np
import gymnasium as gym

import evogym.envs  # noqa: F401  必须导入才会注册所有 env id
from evogym import sample_robot

DEFAULT_TASKS = ["Walker-v0", "Jumper-v0", "Balancer-v1", "Carrier-v1", "Climber-v2"]

VOXEL_NAME = {0: "空", 1: "刚性", 2: "软体", 3: "水平驱动器", 4: "垂直驱动器"}


def run_task(task: str, shape: tuple[int, int], seed: int, greedy_steps: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    body, connections = sample_robot(shape)
    env = gym.make(task, body=body)
    obs, _ = env.reset(seed=seed)

    n_act = int(np.prod(env.action_space.shape))
    n_obs = int(np.prod(obs.shape))
    max_steps = env.spec.max_episode_steps if env.spec is not None else greedy_steps

    total_reward = 0.0
    steps = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    env.close()

    counts = {VOXEL_NAME[k]: int(np.sum(body == k)) for k in sorted(VOXEL_NAME)}
    return {
        "task": task,
        "obs_dim": n_obs,
        "act_dim": n_act,
        "voxels": counts,
        "steps": steps,
        "return": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--shape", nargs=2, type=int, default=[5, 5], metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    shape = (args.shape[0], args.shape[1])

    print(f"evogym {version('evogym')} | numpy {np.__version__} | gymnasium {gym.__version__}")
    print(f"机器人尺寸 {shape}，随机策略，seed={args.seed}\n")
    print(f"{'任务':<22}{'obs':>6}{'act':>6}{'步数':>6}{'累计奖励':>12}  体素构成")
    print("-" * 96)

    for task in args.tasks:
        r = run_task(task, shape, args.seed, 500)
        voxels = " ".join(f"{k}{v}" for k, v in r["voxels"].items())
        print(
            f"{r['task']:<22}{r['obs_dim']:>6}{r['act_dim']:>6}"
            f"{r['steps']:>6}{r['return']:>12.2f}  {voxels}"
        )

    print("\n提示：随机策略只是跑通管线；想看到会走路的机器人需要训练控制器（examples/run_ppo.py）。")


if __name__ == "__main__":
    main()
