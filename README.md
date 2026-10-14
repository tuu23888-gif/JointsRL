# JointsRL

关节 / 形态-控制协同设计（co-design）相关的实验代码库。

当前已接入的第三方基线平台：**[Evolution Gym (EvoGym)](https://github.com/EvolutionGym/evogym)**（MIT CSAIL，NeurIPS 2021），
以 git submodule 形式放在 `third_party/evogym`。

```
JointsRL/
├── demos/                       # EvoGym 上手 demo（本仓库自己写的）
│   ├── evogym_quickstart.py     # 随机策略跑若干任务，验证仿真管线
│   ├── evogym_render_gif.py     # 离屏渲染，导出机器人运动 GIF 到 demos/out/
│   └── evogym_pretrained_robots.py  # 使用官方 9 万+ 机器人结构数据集
├── third_party/evogym/          # EvoGym 源码（submodule，含其自身的 5 个 submodule）
├── data/                        # 数据集缓存（gitignore，运行脚本时自动下载）
└── .gitignore
```

## 1. 环境准备（必须 Python 3.7~3.10）

EvoGym 只发布到 cp310 的 wheel，Python 3.11+ 装不上。本机用 conda 建了一个 3.10 环境：

```shell
conda create -y -n evogym python=3.10

# 关键：EvoGym 2.0.0 仍然 import pkg_resources，而 setuptools>=81 已经删掉了它
D:\miniconda3\envs\evogym\python.exe -m pip install "setuptools==80.9.0"
D:\miniconda3\envs\evogym\python.exe -m pip install evogym

# demos 需要的额外依赖
D:\miniconda3\envs\evogym\python.exe -m pip install imageio pandas pyarrow
```

验证：

```shell
D:\miniconda3\envs\evogym\python.exe -c "import evogym, gymnasium; print(evogym.__file__, gymnasium.__version__)"
```

## 2. 拉取代码

```shell
git clone --recurse-submodules https://github.com/tuu23888-gif/JointsRL.git
# 已经 clone 过的：
git submodule update --init --recursive
```

> 注意：`third_party/evogym` 自己还有 5 个 submodule（pybind11 / glfw / glew / PyTorch-NEAT / ppo 实现），
> 只有从源码编译仿真器或跑 `examples/` 里的 PPO/NEAT 例子时才需要，`--recursive` 会一并拉下来。

## 3. 运行 demo

```shell
cd D:\SharedProjects\JointsRL

# 随机策略跑 5 个任务，打印 obs/act 维度与累计奖励
D:\miniconda3\envs\evogym\python.exe demos\evogym_quickstart.py

# 渲染成 GIF（离屏 rgb_array，不需要开窗口）-> demos/out/Walker-v0_random.gif
D:\miniconda3\envs\evogym\python.exe demos\evogym_render_gif.py --task Walker-v0 --steps 200

# 官方数据集：90563 个带奖励标注的机器人结构
D:\miniconda3\envs\evogym\python.exe demos\evogym_pretrained_robots.py --env Walker-v0 --top 5 --simulate
```

## 4. 关于 EvoGym

* 机器人 = 体素矩阵，取值 `0 空 / 1 刚性 / 2 软体 / 3 水平驱动器 / 4 垂直驱动器`；
  动作是每个驱动器的目标形变比例（`[0.6, 1.6]`）。
* 后端是纯 C++ 的 2D 质量-弹簧仿真器（本机加载的是 `Evolution Gym Simulator v2.2.5`），
  Python 接口对齐 gymnasium：`gym.make(task, body=body)` → `reset()` / `step()` 五元组。
* 共 32 个任务：`Walker-v0`、`Jumper-v0`、`Balancer-v1`、`Carrier-v1`、`Climber-v2`、
  `BeamSlider-v0`、`Lifter-v0`、`Traverser-v0`、`Catcher-v0` ……
  （`Walker-v0` 这类无后缀的是旧版本，gymnasium 会给 deprecation 警告）。
* 官方基线：外层设计优化 GA / 贝叶斯优化 / CPPN-NEAT，内层控制 PPO，脚本在
  `third_party/evogym/examples/`（跑这些需要额外装 `requirements.txt`，包含 torch）。

### 现成的模型？

官方在 HuggingFace 开源了 **[EvoGym/robots](https://huggingface.co/datasets/EvoGym/robots)**：
**90563 个**在论文中被演化出来、带实测奖励标注的机器人结构，
字段为 `uid / body / connections / reward / env_name / generated_by`。

需要明确的点：**数据集里只有"身体"（形态 + 连接），没有控制器权重**。
`reward` 是该形态当年训练出的策略所取得的成绩，想复现分数必须自己用 PPO 重新训练控制器
（`third_party/evogym/examples/run_ppo.py`）。`demos/evogym_pretrained_robots.py --simulate`
只是把结构加载进仿真器验证可用性，随机动作拿不到那个分数。

数据集许可证：CC-BY-NC-4.0（仅限非商业用途）。

## 5. 已踩过的坑

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools>=81 移除了 `pkg_resources`，EvoGym 2.0.0 还在用它 → 降到 `setuptools==80.9.0` |
| `pip install evogym` 报找不到匹配版本 | 当前解释器是 3.12+，wheel 只到 cp310 → 换 Python 3.10 |
| `git clone github.com` 超时，但浏览器能打开 | 系统代理只写在 WinINET 里，git 不读；`git config --global http.proxy http://127.0.0.1:7897` |
| clone 到一半 `RPC failed; curl 56 schannel` | `git config --global http.version HTTP/1.1` 后重试 |

## 6. 引用

```bibtex
@article{bhatia2021evolution,
  title={Evolution gym: A large-scale benchmark for evolving soft robots},
  author={Bhatia, Jagdeep and Jackson, Holly and Tian, Yunsheng and Xu, Jie and Matusik, Wojciech},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
