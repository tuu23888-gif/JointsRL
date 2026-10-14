"""加载 EvoGym 官方发布的『结构 + 已训练好的策略』数据集，直接跑出会走路的机器人。

数据集：HuggingFace `EvoGym/robots-with-policies`
        —— 2553 个机器人，每个都带序列化的 torch 策略（policy_blob），总计约 201 MB。

依赖：torch + stable-baselines3（只为反序列化 RunningMeanStd 类）
      + `third_party/evogym-datasets` 里官方提供的 `a2c_ppo_acktr` 策略类定义。

用法：
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_run_pretrained_policy.py --env Walker-v0
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_run_pretrained_policy.py --env Walker-v0 --gif
    D:\\miniconda3\\envs\\evogym\\python.exe demos\\evogym_run_pretrained_policy.py --env Walker-v0 --top 3 --gif
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS_REPO = ROOT / "third_party" / "evogym-datasets"
PARQUET = ROOT / "data" / "evogym_robots_with_policies.parquet"
HF_URL = (
    "https://huggingface.co/datasets/EvoGym/robots-with-policies/resolve/main/data/"
    "train-00000-of-00001.parquet"
)
OUT_DIR = Path(__file__).resolve().parent / "out"

# 官方 evogym-datasets 仓库里带有策略类的定义，反序列化 pickle 时必须能 import 到
sys.path.insert(0, str(DATASETS_REPO))

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import evogym.envs  # noqa: E402  (注册所有 env id)
from a2c_ppo_acktr.model import Policy  # noqa: E402,F401  反序列化 pickle 需要


def to_matrix(x) -> np.ndarray:
    """parquet 里 body / connections 是 object ndarray（每行一个数组），转成二维整数矩阵。"""
    if isinstance(x, np.ndarray) and x.dtype == object:
        return np.vstack([np.asarray(row, dtype=np.int64) for row in x])
    return np.asarray(x, dtype=np.int64)


def ensure_dataset() -> Path:
    if not PARQUET.exists():
        import urllib.request

        PARQUET.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载数据集（约 201 MB）-> {PARQUET}")
        urllib.request.urlretrieve(HF_URL, PARQUET)
    return PARQUET


def deserialize_policy(blob: bytes):
    """policy_blob -> (actor_critic, obs_rms)，官方 robots_with_policies.py 的做法。"""
    return torch.load(io.BytesIO(blob), weights_only=False, map_location="cpu")


def normalize_obs(obs: np.ndarray, obs_rms, epsilon: float = 1e-8, clip_obs: float = 10.0) -> np.ndarray:
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)


def rollout(row, record_gif: bool = False, seed: int = 0):
    actor_critic, obs_rms = deserialize_policy(row["policy_blob"])
    body, connections = to_matrix(row["body"]), to_matrix(row["connections"])

    env = gym.make(
        row["env_name"],
        body=body,
        connections=connections,
        render_mode="rgb_array" if record_gif else None,
    )
    obs, _ = env.reset(seed=seed)

    hidden = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    frames = [env.render()] if record_gif else []

    total, steps = 0.0, 0
    while True:
        with torch.no_grad():
            o = torch.tensor(
                normalize_obs(np.asarray(obs, dtype=np.float32), obs_rms), dtype=torch.float32
            ).unsqueeze(0)
            _, action, _, hidden = actor_critic.act(o, hidden, masks, deterministic=True)
        # +1.0 是动作范围从旧版 [-1,1] 迁到 [0.6,1.6] 的向后兼容偏移
        obs, reward, terminated, truncated, _ = env.step(action.numpy().reshape(-1) + 1.0)
        masks.fill_(0.0 if (terminated or truncated) else 1.0)
        total += float(reward)
        steps += 1
        if record_gif:
            frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return total, steps, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker-v0", help="任务名，例如 Walker-v0 / Carrier-v0 / Lifter-v0")
    parser.add_argument("--top", type=int, default=1, help="跑奖励最高的前 N 个策略")
    parser.add_argument("--gif", action="store_true", help="导出运动 GIF 到 demos/out/")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(ensure_dataset())
    sub = df[df["env_name"] == args.env].sort_values("reward", ascending=False)
    if sub.empty:
        print(f"没有 {args.env} 的策略；可用任务示例：{', '.join(sorted(df['env_name'].unique())[:10])}")
        return

    print(f"数据集共 {len(df)} 个带策略的机器人，其中 {args.env} 有 {len(sub)} 个")
    print(f"{'#':<3}{'reward(数据集)':>15}{'reward(实测)':>14}{'步数':>7}  uid")
    print("-" * 78)

    for i, (_, row) in enumerate(sub.head(args.top).iterrows(), 1):
        total, steps, frames = rollout(row, record_gif=args.gif, seed=args.seed)
        print(f"{i:<3}{row['reward']:>15.3f}{total:>14.3f}{steps:>7}  {row['uid']}")

        if args.gif and frames:
            import imageio.v2 as imageio

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUT_DIR / f"{args.env}_policy_{i}.gif"
            imageio.mimsave(out, frames, fps=args.fps, loop=0)
            print(f"    -> {out}（{len(frames)} 帧）")

    print("\n说明：实测 reward 与数据集记录的会有差异（论文里也提到，库版本/架构变化会导致漂移）。")


if __name__ == "__main__":
    main()
