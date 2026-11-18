# ntuple2048

纯 CPU、无 GUI 的 2048 AI：自研模拟环境 + n-tuple 网络 + TD(afterstate) 训练 + 贪心推理。

默认特征集为 **12 个矩形 6-tuple + 8 个条形 4-tuple**，学习目标是 **afterstate 的价值**
（移动/合并完成、尚未生成新数字时的棋盘），更新规则为类 SARSA 的 on-policy TD(0)。

> **▶ 在线演示：[huaji-xiaobing.github.io/2048ai](https://huaji-xiaobing.github.io/2048ai/)**
> 纯静态页面，价值表在浏览器里直接推理，没有后端、没有搜索。

## 安装

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy pytest
# 或者
pip install -e .
```

不安装也能跑，把 `src` 放进 `PYTHONPATH` 即可。

## 快速开始

```bash
# 训练（8 进程 × 128 棋盘，500 万环境步，约 8 分钟）
python -m ntuple2048 train --workers 8 --boards 128 --steps 5000000 \
    --chunk 2000 --epsilon-decay-steps 2000000 --eval-episodes 100 \
    --save checkpoints/model.npz

# 用训好的价值表评估（纯贪心）
python -m ntuple2048 eval --model checkpoints/model.npz --episodes 100

# 评估时对 8 个对称变换取平均（慢 8 倍，效果取决于模型对称程度）
python -m ntuple2048 eval --model checkpoints/model.npz --episodes 100 --d4-average

# 打印一局完整对局
python -m ntuple2048 eval --model checkpoints/model.npz --play

# 单进程吞吐基准
python -m ntuple2048 bench --boards 256 --steps 2000

# 查看 n-tuple 模式定义与 D4 轨道
python -m ntuple2048 patterns
```

限时训练 + 定时评估（按挂钟切分，ε 退火跨切片连续）：

```bash
python scripts/long_train.py --minutes 60 --eval-every 5 --eval-episodes 100 \
    --workers 8 --boards 128 --out checkpoints/longrun_60min
# 产物：model.npz（每轮评估后刷新）、eval_log.jsonl（逐次指标）、train.log
```

## 项目结构

```
src/ntuple2048/
├── lookup.py      65536 项的行级查找表（滑动 / 合并 / 得分）
├── board.py       单棋盘引擎（不可变 Board，方向常量，spawn）
├── env.py         批量引擎：一次处理 (N, 16) 的棋盘矩阵
├── tuples.py      12 个 6-tuple + 8 个 4-tuple 的模式定义
├── symmetry.py    D4 的 8 个置换、轨道、棋盘变换
├── network.py     n-tuple 价值表：编码、前向、稀疏 TD 更新、存取
├── policies.py    策略接口 + ε-greedy / softmax / 贪心，可注册扩展
├── trainer.py     TD(afterstate) 训练核心（单进程、多棋盘）
├── parallel.py    多进程驱动（fork + 共享内存权重）
├── inference.py   贪心推理、D4 平均、批量评估
├── teacher.py     调用网页版 WASM 专家生成轨迹（Node 子进程 + 二进制协议）
├── distill.py     afterstate 分类的行为克隆
└── cli.py         命令行入口
scripts/
├── long_train.py           限时训练 + 定时评估驱动
├── distill_from_teacher.py 生成专家轨迹并蒸馏
└── export_web_model.py     导出 int8 量化 + gzip 的浏览器模型
docs/             静态演示页（GitHub Pages 从 main/`docs` 发布）
├── index.html / style.css / app.js   2048 UI 与 AI 自动循环
├── ai.js         浏览器端推理：LUT + n-tuple 编码 + 查表
└── model.bin.gz  6.2 MB，base=12 量化模型
2048-web/         上游网页版原样拷贝（含 ai_core.wasm 与 teacher_server.mjs）
tests/            无 GUI 的单元测试（42 个）
```

## 设计要点

### 1. 棋盘表示

一个格子存 `log2(tile)`（`0` 为空，`1` 为 2，`2` 为 4 …），16 个格子打包成
`(16,)` 的 `uint8`。单棋盘是 `tuple`，批量是 `(N, 16)` 矩阵。

一条 4 格的线编码为 16 bit 整数索引查找表，表里有三样东西：滑动后的新线、
合并产生的 `log2` 得分、真实游戏得分。因此一次移动 = 4 次查表，四个方向一次算完。

```python
from ntuple2048 import Board
b = Board.from_tiles([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
b2, gain_log, gain_real = b.move(0)   # 0 = LEFT
```

### 2. n-tuple 特征

| 组 | 数量 | 形状 | 说明 |
|----|------|------|------|
| 6-tuple | 12 | 2×3 / 3×2 滑窗 | 4×4 上全部 6 个 2×3 与 6 个 3×2 窗口 |
| 4-tuple | 8 | 4 行 + 4 列 | 整行整列 |

模式内的单元格顺序为窗口内行优先，它决定编码中 `base**t` 的幂次：

```
code(P, s) = Σ_t s[P[t]] · base**t
V(s) = Σ_P W_P[ code(P, s) ]
```

默认 `base = 15`（格子值裁剪到 0..14，即可表示到 16384 的方块）。
6-tuple 表 `12 × 15^6 × 4B ≈ 547 MB`，4-tuple 表只有 1.6 MB。
`--base` 可调：`13` 约 190 MB（上到 4096），`14` 约 362 MB（上到 8192）。

编码用一次 `float32` 矩阵乘法批量算出（所有 code 都 < 2^24，浮点整数运算精确），
前向是两次 fancy-index gather，反向是两次 scatter add。

### 3. TD 更新：afterstate + 类 SARSA

状态定义为 **afterstate** `s'`（已合并、未生成新数字）。转移是

```
afterstate s'  --spawn-->  board s  --a ~ π-->  afterstate s''   奖励 r
```

于是 on-policy 的 TD(0) 更新为

```
V(s') ← V(s') + α · [ r + γ · V(s'') − V(s') ]
```

其中 `a` 由**当前正在训练的策略**采样（这就是"类 SARSA"：用同一策略的后续价值做目标），
`r` 是在 `s` 上执行 `a` 得到的奖励。当 `spawn` 之后棋盘无路可走时目标退化为 `0`
（终止态价值为 0）。

奖励默认用 `log` 模式：一次移动的奖励 = 本次合并出的方块的 `log2` 之和，
`--reward score` 可切换为真实 2048 计分。

`γ = 0.999` 是默认值：2048 的对局可达上千步，`0.99` 的 100 步有效视野明显不足
（实测 500 万步后平均分 17657 vs 23855）。

### 4. 可插拔策略

训练循环只依赖一个窄接口：

```python
class Policy:
    def select(self, q, valid, rng) -> np.ndarray:  # (N,4) -> (N,)
    def on_step(self, step: int) -> None:           # 可选的调度钩子
```

内置 `epsilon_greedy`（带线性退火）、`softmax`、`greedy`。
新增策略只要实现 `select` 并用 `@register("name")` 注册，`--policy name` 即可切换：

```python
from ntuple2048.policies import Policy, register

@register("my-policy")
class MyPolicy(Policy):
    def select(self, q, valid, rng):
        ...
```

### 5. 并行

* **多棋盘**：`TrainConfig.boards` 个棋盘在同一个 numpy batch 里同步推进，
  每步四方向 afterstate、编码、前向、采样、更新全部向量化。
* **多核**：`parallel.py` 用 `fork` 起多个进程，各自跑一个 `Trainer`，
  权重放在一块 `multiprocessing.RawArray` 里共享，更新是稀疏 scatter add
  （Hogwild! 风格；极少数索引冲突会丢掉一次更新，对 TD 无实质影响）。
  每个 chunk 之间用 `pool.map` 做一次屏障，主进程可安全地保存 checkpoint。

```bash
python -m ntuple2048 train --workers 8 --boards 128 --steps 5000000
# => 8 核实测约 1.3M board-steps/s，500 万环境步 ≈ 480 s
```

单进程 `--workers 1 --boards 256` 约 300k board-steps/s。

### 6. D4 对称性

8 个对称变换定义在 `symmetry.py`，棋盘变换是 `T_g(b)[p] = b[g(p)]`。

**训练增强（默认开启）**：每个 rollout chunk 开始时随机对整批棋盘施加一次 D4 变换，
chunk 内保持同一朝向。12 个矩形 + 8 个条形作为一个*位置集合*在 D4 下封闭，
所以这相当于让 20 张表在训练中轮换着看到各个朝向的数据。实测（150 万步、60 局）：

| 设置 | 平均分 | p10 | 2048 达成 |
|------|--------|-----|-----------|
| 无增强 | 19080 | 12226 | 24/60 |
| 每步都变换 | 16344 | 12047 | 11/60 |
| **每 chunk 变换一次** | **20631** | **14250** | **31/60** |

"每步都变换"更差，因为 `V(s')` 和 `V(s'')` 会落在不同朝向下，给 TD 目标注入噪声。

**权重投影（刻意不做）**：`W_P ← W_{g(P)}` 要求**有序元组**集合在 D4 下封闭，
但 2×3 窗口按行优先读取、转置后变成 3×2 的列优先读取，顺序对不上。
要做严格投影就得为每个窗口保存 4 种遍历顺序，表数量翻 4 倍而无精度收益。
`network.check_d4_closed()` 会明确返回 `False` 并说明原因。

**推理平均（`--d4-average`，可选）**：`V(s) = 1/8 Σ_g V(T_g s)`。
它只在网络本身接近对称时有益，否则会把 8 个朝向上的系统性偏差平均进决策。
实测（同为 100 局、同一 seed）：

| 模型 | 贪心 | D4 平均 |
|------|------|---------|
| 160 万步（早期，对称偏差小） | 17657 | **18612** |
| 500 万步 | 27892 | 21917 |
| 3827 万步（60 分钟） | 39548 | 26230 |

也就是说，训练越久、位置特化越强，平均反而越糟。**默认关闭，用前先跑小样本对比。**

**诊断**：训练结束会打印

```
D4 consistency: max |V(b) - V(g b)| = 65.750 (0.05% of mean |V|)
```

探针棋盘是 6~13 个随机方块的"中局"棋盘 —— 刚开局的棋盘价值太小，
会让这个指标看起来虚假地好。

## 蒸馏 WASM 专家（2048EndgameTablebase）

上游项目 [game-difficulty/2048EndgameTablebase](https://github.com/game-difficulty/2048EndgameTablebase)
只发布浏览器版本，仓库里没有可直接读取的表文件。网页版由三部分组成：

| 文件 | 作用 |
|------|------|
| `ai_core.wasm` (2.8 MB) | C++ 编译的搜索 + 残局表基，**98.8% 是 data 段**（内嵌表数据） |
| `ai_core.js` | Emscripten/embind 胶水 |
| `worker.js` | `L3Manager`（何时把大块 mask 成 15 去查表）+ `CoreAILogic`（调度、剪枝） |

### 为什么不"用静态语言重写"

`ai_core.wasm` 本身就是 C++（`-O3`）的产物，JS 只是不到 1000 行的调度胶水。
实测各接口成本：

| 接口 | 耗时 |
|------|------|
| `move_board` | 152 ns |
| `find_best_egtb_move`（残局查表） | 586 ns |
| `evaluate(board)`（单局面启发式） | 1.2 μs |
| `start_search(depth=4)`（开局 / 中盘） | 4.0 ms / 0.8 ms |

瓶颈是**搜索深度**而不是宿主语言，换 Rust/C 重写不会更快。真正的成本在
Python↔JS 边界的往返，所以这里把边界做得足够粗：**一个 Node 进程跑完整局**，
Python 只读一个二进制文件。

### 原版的搜索深度

**不是固定深度**，而是 `CoreAILogic.calculate_step` 按局面分档的迭代加深：

| 分支 | 条件 | initial_depth | max_depth | 时间预算 |
|------|------|---------------|-----------|----------|
| 乱局 / 5-tiler | `is_mess \|\| is_5tiler` | 5 | 24 | `1.2 · bigNums^0.25` s |
| 开局 | 空位>4 且 bigNums<2 且不合并 | 3 | 3 | 0.1 s |
| 中局 | bigNums ≤ 3 | 4 / 5 | 同左 | 0.1 s |
| 常规 | 其他 | 4 (+prune) | 24 | `0.16 · bigNums^0.25` s |
| 尾盘 | `board_sum` 接近 65536 | 20–33 | 60 | 0.8–1.0 s |

时间预算再乘 `time_limit_ratio`（网页版速度滑块 `10^((100-slider)/100)`，默认 1.0）。

实测每层成本（同一局不同阶段）：

| 局面 | depth 4 | depth 6 | depth 8 | 迭代加深可达 |
|------|---------|---------|---------|--------------|
| 10 个 tile（稀疏） | 2.4 ms | 2.1 ms | 3.0 ms | **depth 24** / 52 ms |
| 11 个 tile | 1.0 ms | 1.6 ms | 2.7 ms | **depth 24** / 58 ms |
| 14 个 tile（密集） | 4.3 ms | 82.8 ms | 4739 ms | depth 5–7（时间到） |

也就是说：稀疏局面深度能拉到 24；密集局面被时间预算卡在 5–7 层。
**早期写死 `depth=4` 等于把后期的 20 层全砍掉了**，所以只到 8192。

另外原版还有一层保险：`L3Manager` 在 `bigNums` 达到阈值时把大块 mask 成 15
去查 EGTB（`find_best_egtb_move`），这一步是纯查表（586 ns），和搜索互补。

`ai_core.wasm` 的 data 段里确实内嵌了残局表，但索引逻辑（`L3Manager` 的
threshold / table_type 组合）要反汇编才能安全复现，所以这里直接把
`worker.js` 的 `L3Manager` + `CoreAILogic` 原样抽到 `expert_core.mjs` 复用，
而不是重新实现。

修正后的教师（完整原版策略，`--speed` 对应网页版滑块）：

| speed | 结果 | 步数 | 耗时/局 |
|-------|------|------|---------|
| 0.2（快 5×） | **32768** | 22419 | ≈5 min |
| 0.5 | **16384** | 14819 | 9.8 min |
| 1.0（网页版默认） | — | — | 20 min+ |

### 管线

```
Node (×8)  教师自对弈 depth=4  ──►  traj_XXXX.bin  (17 B/状态)
                                          │
Python     numpy 零拷贝解析 ──► 行为克隆（afterstate 分类）
                                          │
                               n-tuple 价值表 (.npz)
```

蒸馏目标把网络当成四选一的分类器：专家动作对应的 afterstate 应得分最高，

```
p_i = softmax(V(afterstate_i) / T)      L = -log p_expert
dL/dV_i = p_i - 1[i == expert]
```

n-tuple 的"梯度"就是查表加法，直接复用 `update_from_codes`；
`V` 先按样本做 z-score 再 softmax，避免已训练表的巨大数值把梯度压死。

### 用法

```bash
# 生成 8 局专家轨迹并蒸馏（8 进程，speed=0.2 约 5 分钟/局）
python -m ntuple2048 distill --games 8 --speed 0.2 --epochs 5 --lr 0.01 \
    --eval-episodes 100 --save checkpoints/distilled.npz

# 复用已生成的轨迹
python -m ntuple2048 distill --reuse --epochs 10 --lr 0.005

# 在已有 TD 模型上继续蒸馏
python -m ntuple2048 distill --reuse --resume checkpoints/longrun_60min/model.npz
```

`--speed` 对应网页版的速度滑块：`<1` 更快更弱，`1.0` 是网页默认（每局 20 分钟以上），
`>1` 更慢更强。

### 实测

纯蒸馏（从零开始，没有任何 TD）：

| 专家配置 | 数据量 | 蒸馏耗时 | 动作一致率 | 贪心评估（100 局） |
|----------|--------|----------|------------|--------------------|
| 固定 depth=4 | 27.1 万 | 5.8 s | 71.9% | 5176 分 / 1024 |
| 固定 depth=4 | 98.6 万 | 16.3 s | 68.4% | 6498 分 / 1024 |
| **原版策略 speed=0.2** | 16.5 万 | 4.3 s | **73.3%** | 4862 分 / 1024 |

原版策略的单条样本质量更高（一致率 73.3%），但每局 2 万步、5 分钟，
样本量增长慢；固定 depth=4 便宜 5 倍，适合先堆量。

蒸馏当作初始化，再接 TD（300 万步，ε 0.3→0.03，用的是固定 depth=4 的 98.6 万样本）：

| 方案 | 环境步 | 平均分 | 中位数 | 4096 达成 |
|------|--------|--------|--------|-----------|
| 纯 TD 从零 | 500 万 | 27892 | 27836 | 7% |
| **蒸馏 + TD** | **300 万** | **32238** | **34010** | **14%** |

也就是说，用 17 分钟的专家数据换掉了约一半的 TD 训练量，还高出 15%。

## 浏览器演示（`docs/`）

**线上地址：<https://huaji-xiaobing.github.io/2048ai/>**（GitHub Pages，`main` 分支 `/docs` 目录）

`docs/` 是一个纯静态页面，把训练好的 n-tuple 表直接跑在浏览器里——没有后端，也没有搜索：

* 移动/合并：65536 项行查找表，和训练端同一套逻辑；
* 决策：对四个 afterstate 各做一次前向取最大，每步约 **0.03 ms**；
* 模型：`base=12` 的 12×6-tuple + 8×4-tuple，int8 每表独立 scale 量化，gzip 后 **6.2 MB**。

### 为什么单独训一个 base=12

训练用的 `base=15` 表有 547 MB，不可能塞进网页。`base=12` 把 6-tuple 表从
`15^6` 降到 `12^6`（136.7M → 35.8M 项），代价是编码时把大于 2048 的块裁剪到 2048。
这只影响 4096 以上的局面（仍能玩出来，只是那部分信息被截断）。

### 实测（在 Node 里直接跑 `docs/ai.js`，每档 20 局）

| 量化 | 文件体积 | 平均分 | 中位数 | 最大方块 |
|------|----------|--------|--------|----------|
| int16 | 30.5 MB | 29615 | 32296 | 4096 |
| **int8** | **6.2 MB** | 31530 | 32212 | 4096 |

int8 的量化噪声（每表 scale ≈ 1.0）远小于相邻动作的价值差，所以 argmax 基本不变。

### 生成与本地预览

```bash
python -m ntuple2048 train --workers 8 --boards 128 --base 12 \
    --steps 6000000 --epsilon-decay-steps 2500000 --save checkpoints/web_model.npz
python scripts/export_web_model.py --model checkpoints/web_model.npz \
    --out docs/model.bin.gz --bits 8
python -m http.server -d docs 8000   # 打开 http://localhost:8000
```

GitHub Pages：Settings → Pages → Source 选 `Deploy from a branch`，分支 `main`、目录 `/docs`。

## 实测

### 60 分钟限时训练（`scripts/long_train.py`）

8 进程 × 128 棋盘、`γ=0.999`、chunk 级 D4 增强，每 5 分钟做一次 100 局贪心评估，
最终 **3827 万环境步 / 49 亿 board-steps**（约 3600 s，10.6k env-step/s）。
下表是最终模型在 200 局上的复核（贪心）：

| 指标 | 数值 |
|------|------|
| 平均分 | 39978 |
| 中位数 | 36026 |
| 最低 / 最高 | 7064 / 76832 |
| p10 / p90 / p99 | 27104 / 59408 / 71298 |
| 平均步数 | 1960 |
| 方块分布 | 512 ×1、1024 ×14、**2048 ×125（62%）**、**4096 ×60（30%）** |

训练过程中的 12 个评估点（各 100 局）：

| t (min) | 环境步 | ε | 平均分 | 中位数 | p10 | 最大方块 |
|---------|--------|---|--------|--------|-----|----------|
| 5 | 3.0M | 0.821 | 12090 | 12298 | 7105 | 2048 |
| 10 | 6.0M | 0.641 | 12817 | 12202 | 8345 | 2048 |
| 15 | 9.1M | 0.461 | 14083 | 13044 | 9860 | 2048 |
| 20 | 12.1M | 0.283 | 17402 | 16238 | 12011 | 2048 |
| 25 | 15.2M | 0.100 | 20820 | 23210 | 15437 | 2048 |
| 30 | 18.5M | 0.050 | 31091 | 32274 | 16156 | 4096 |
| 35 | 21.9M | 0.050 | 34068 | 34976 | 16074 | 4096 |
| 40 | 25.4M | 0.050 | 36994 | 35914 | 26811 | 4096 |
| 45 | 28.8M | 0.050 | 35654 | 35692 | 25038 | 4096 |
| 50 | 32.0M | 0.050 | 39423 | 36168 | 27506 | 4096 |
| 55 | 35.2M | 0.050 | 40332 | 35942 | 27240 | 4096 |
| 60 | 38.3M | 0.050 | 40666 | 36120 | 31926 | 4096 |

ε 触及下限（t≈30min）时平均分有一次明显跃升（20.8k → 31.1k），
此后主要是稳定性改善（最低分 4.6k → 15k）。

### 5 分钟短跑

同样配置、500 万环境步（6.4 亿 board-steps，480 s），100 局：

| 指标 | 数值 |
|------|------|
| 平均分 | 27892 |
| 中位数 | 27836 |
| p10 / p90 | 15183 / 36081 |
| 最大方块 | 4096（7%）、2048（66%） |

随机策略基线约 1100 分、256 方块。

## 参数速查（`train`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--workers` | 4 | 进程数 |
| `--boards` | 256 | 每个进程并行的棋盘数 |
| `--steps` | 1e6 | 所有进程**合计**的环境步数 |
| `--chunk` | 2000 | 每个进程在两轮屏障之间执行的步数 |
| `--base` | 15 | 编码基数（决定内存与可表示的最大方块） |
| `--lr` | 0.0025 | TD 学习率 |
| `--gamma` | 0.999 | 折扣因子 |
| `--reward` | log | `log` / `score` |
| `--policy` | epsilon_greedy | 见 `available_policies()` |
| `--epsilon-start/end/decay-steps` | 1.0 / 0.05 / 5e6 | ε 线性退火；建议 `decay ≈ 0.4 × steps` |
| `--no-d4-augment` | 关 | 关闭 chunk 级随机 D4 变换 |
| `--save-every` | 0 | 每 N 个 chunk 存一次 checkpoint |
| `--resume` | – | 从已有 `.npz` 继续训练 |

`--steps × --boards` 就是总的 board-step 数。单机建议 `--workers = 物理核数`、
`--boards` 取 64～256（越大越快，但每步的更新粒度更粗）。

想继续提升：把 `--steps` 加到 2e7 以上、`--boards 256`，并把 `--epsilon-end` 降到 0.01。

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

覆盖：四向滑动与合并、得分、非法移动、终止判定、批量与标量一致性、
模式编码、前向与手算一致、稀疏更新、存取往返、D4 群性质与轨道、
策略合法性、TD 更新、推理与批量评估、多进程 smoke test。

## 参考

* Szubert & Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the Game 2048* (2014)
* nneonneo, `2048-ai`（n-tuple TD 的经典工程实现）
