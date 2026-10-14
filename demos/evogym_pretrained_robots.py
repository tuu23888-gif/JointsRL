"""使用 EvoGym 官方开源的机器人结构数据集（HuggingFace: EvoGym/robots）。

数据集内容：90563 个在论文中被演化出来、并带有实测奖励标注的机器人结构。
字段：uid / body(体素矩阵) / connections(连接表) / reward(该设计训练出的策略所得奖励)
      / env_name(任务) / generated_by(GA | Bayesian Optimization | CPPN-NEAT)
注意：数据集里**只有"身体"（形态 + 连接），没有控制器权重**。要复现 reward 需要自己用 PPO 重新训练控制器。

用法：
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_pretrained_robots.py
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_pretrained_robots.py --env Walker-v0 --top 5 --simulate

首次运行会从 HuggingFace 下载约 7 MB 的 parquet 到 data/evogym_robots.parquet。
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HF_URL = (
    "https://huggingface.co/datasets/EvoGym/robots/resolve/main/data/"
    "train-00000-of-00001.parquet"
)
CACHE = Path(__file__).resolve().parent.parent / "data" / "evogym_robots.parquet"


def to_matrix(x) -> np.ndarray:
    """parquet 里 body / connections 是 object ndarray（每个元素是一行），转成二维整数矩阵。"""
    if isinstance(x, np.ndarray) and x.dtype == object:
        return np.vstack([np.asarray(row, dtype=np.int64) for row in x])
    return np.asarray(x, dtype=np.int64)


def load_dataset() -> pd.DataFrame:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载数据集 -> {CACHE}")
        urllib.request.urlretrieve(HF_URL, CACHE)
    return pd.read_parquet(CACHE)


def simulate(body: np.ndarray, task: str, steps: int = 200, seed: int = 0) -> float:
    """把数据集里的一个结构丢进仿真器，用随机动作跑一段（只为验证结构可用）。"""
    import gymnasium as gym
    import evogym.envs  # noqa: F401

    env = gym.make(task, body=body)
    env.reset(seed=seed)
    total = 0.0
    for _ in range(steps):
        _, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total += float(r)
        if terminated or truncated:
            break
    env.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker-v0", help="按任务名筛选，如 Walker-v0 / Carrier-v0")
    parser.add_argument("--top", type=int, default=5, help="展示奖励最高的前 N 个设计")
    parser.add_argument("--simulate", action="store_true", help="把最优结构放进仿真器跑一遍")
    args = parser.parse_args()

    df = load_dataset()
    print(f"数据集共 {len(df)} 个机器人结构，覆盖 {df['env_name'].nunique()} 个任务\n")

    print("按任务统计（前 10）：")
    print(df["env_name"].value_counts().head(10).to_string(), "\n")

    print("按生成算法统计：")
    print(df["generated_by"].value_counts().to_string(), "\n")

    sub = df[df["env_name"] == args.env].sort_values("reward", ascending=False)
    if sub.empty:
        print(f"没有任务 {args.env} 的数据，可用任务示例：{', '.join(df['env_name'].unique()[:8])}")
        return

    print(f"{args.env} 共 {len(sub)} 个设计，奖励最高的 {args.top} 个：")
    for i, row in enumerate(sub.head(args.top).itertuples(), 1):
        body = to_matrix(row.body)
        n_act = int(np.sum((body == 3) | (body == 4)))
        print(
            f"  {i}. uid={row.uid:<10} reward={row.reward:>9.3f}  "
            f"算法={row.generated_by:<22} 尺寸={body.shape} 驱动器={n_act}"
        )

    best = sub.iloc[0]
    body = to_matrix(best["body"])
    print(f"\n最优结构的体素矩阵（0 空 / 1 刚性 / 2 软体 / 3 水平驱动器 / 4 垂直驱动器）：")
    print(body)

    if args.simulate:
        print("\n用随机策略在仿真器里跑 200 步（结构可加载性验证）……")
        total = simulate(body, args.env)
        print(f"随机策略累计奖励 = {total:.2f}（数据集里记录的 {best['reward']:.3f} 是训练过的策略所得）")


if __name__ == "__main__":
    main()
