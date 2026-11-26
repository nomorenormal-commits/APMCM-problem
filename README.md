# APMCM 2026 Problem C — African Penguin Acoustic Communication

**Signal Fidelity, Behavioural Response and Effective Communication Range in the African Penguin**

本仓库包含 APMCM 2026 C 题的完整论文、分析代码、数据、图表与结果，全部结果可由固定随机种子复现。

- 论文 PDF：[`C题_论文/main.pdf`](C题_论文/main.pdf)
- LaTeX 源码：[`C题_论文/main.tex`](C题_论文/main.tex)
- 分析代码：[`C题_论文/code/`](C题_论文/code)
- 原始数据：[`C题_论文/data/`](C题_论文/data)

## 研究概要

论文把“叫声能否被听到、能否被识别、是否会被回应”放在同一条概率链条上，使用 108 条海面叫声、139 组回放实验和 18,900 条 4–256 m 传播相关性，完成四个递进问题：

1. **声学结构与分类**：三类叫声主要由谐波—时间微结构区分，而非总体频谱包络。全特征随机森林交叉验证准确率 93.4%，但其中包含双音叫声的定义性识别；仅用普适频谱特征时准确率降至 54.2%。
2. **传播衰减与有效距离**：Fisher-z 复合保真度从 4 m 的 0.86 衰减至 256 m 的 0.18–0.28，符合平台—陡降型 Hill 衰减。贝叶斯分层 Hill 模型与六源蒙特卡洛给出中位有效半径：flat 61 m、modulated 110 m、two-voice 134 m；达到 120 m 的概率分别为 0%、30%、88%。
3. **回放行为与因果审计**：自然叫声显著提高响应几率（OR 14.0–20.9）；E-value、逆倾向加权与阴性对照表明 call-type 效应稳健，而“群体越大响应越少”的关联不能解释为因果。
4. **管理优化**：以通信有效性指数与约束优化给出分区缓冲方案；在示意性分区下，自适应缓冲相对统一 150 m 方案最多提升 10.8% 加权覆盖率，且最优排序对权重 ±20%、噪声 ±0.03–0.05 扰动保持不变。

## 仓库结构

```text
.
├── README.md
├── 2026 APMCM Problems B&C&D/                  # 官方赛题与数据（部分超大文件未上传）
├── 2026 APMCM 论文提交规范、论文模板、承诺书/    # 官方规范与模板
├── APMCM2026_示例论文/                         # 官方示例论文
├── 2026 APMCM 论文模板_中文对照.docx
└── C题_论文/
    ├── main.tex                                # 论文源码
    ├── main.pdf                                # 最终论文（35 页）
    ├── apmcmthesis.cls / .cfg                  # LaTeX 模板
    ├── code/                                   # 分析脚本
    │   ├── common.py                           # 公共路径、绘图与表格工具
    │   ├── p1.py                               # 声学描述统计、检验、分类
    │   ├── p2.py                               # 复合保真度、Hill 拟合、阈值半径
    │   ├── p3.py                               # 回放响应、组合、删失时延模型
    │   ├── p4.py                               # 有效性指数、半径、噪声情景
    │   ├── p5_bayes.py                         # 贝叶斯分层 Hill + 风险蒙特卡洛
    │   ├── p5_threshold.py                     # 阈值条件概率热图
    │   ├── p6_causal.py                        # E-value、IPW、SMD、响应分解
    │   └── p7_buffer.py                        # 约束缓冲优化与输入敏感性
    ├── data/                                   # 原始 CSV 与 WAV
    ├── figures/                                # 论文引用的 PDF 图
    ├── tables/                                 # 论文引用的 LaTeX 表格
    └── results/                                # JSON / NPZ 结果
```

## 复现方式

### 1. 环境

- Python 3.10+
- 依赖：

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib emcee patsy
```

- LaTeX：XeLaTeX（含 `ctex`、`booktabs`、`float` 等常用宏包）

### 2. 运行分析

在 `C题_论文/code/` 下依次执行：

```bash
python p1.py
python p2.py
python p3.py
python p4.py
python p5_bayes.py        # 生成贝叶斯链 results/p5_chain.npz
python p5_threshold.py    # 依赖上一步的链文件
python p6_causal.py
python p7_buffer.py       # 依赖 p5_chain.npz
```

脚本会把图表写入 `figures/`、`tables/`，把关键数值写入 `results/`。

### 3. 编译论文

在 `C题_论文/` 下执行：

```bash
xelatex main.tex
xelatex main.tex
xelatex main.tex
```

或使用：

```bash
latexmk -xelatex main.tex
```

## 数据与复现说明

- 分析使用官方 APMCM-C 数据：`Data01`–`Data11`、三份 1 秒 12 kHz 录音，以及回放实验数据。
- 所有随机种子固定（42、7、20261126）；主要中间结果以 JSON/NPZ 保存，可从脚本输出追溯到论文数字。
- 论文正文（含参考文献）25 页，附录第 26–33 页，共 35 页。
- 为遵守 GitHub 单文件限制，以下 3 个大文件未上传，但在本地工作区保留：
  - `2026 APMCM Problem C.zip`（约 59 MB）
  - `2026 APMCM Problem B/Data.7z`（约 102 MB）
  - `Supplementary2_Video_recordings_of_a_playback_experiment.mp4`（约 60 MB）

## 说明

仓库中包含官方赛题、提交规范与示例论文，仅用于竞赛复现与学习；相关版权归原作者/主办方所有。论文与代码如用于其他用途，请先确认数据使用许可。
