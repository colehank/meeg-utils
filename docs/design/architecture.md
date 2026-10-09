# meeg-utils 架构设计（v0.2 草案）

> 状态：讨论稿。本文档是 0.2.0 重构的依据，接口定稿前可随时修改。

## 1. 定位

**用 Python API 把 MNE 生态的成熟方法串成"开箱即用、处处可调、步步留痕"的 M/EEG 处理工具**，覆盖质量检查（QC）、预处理和数据分析。

- **封装，不重写。** 算法交给 MNE、mne-bids、mne-denoise、pyprep、mne-icalabel、autoreject、specparam、mne-connectivity 等成熟库；本库负责编排、默认值、质量检查和处理记录。
- **数据对象用 MNE 原生类型。** `Raw` / `Epochs` / `Evoked` 不再包一层，用户随时可以退回原生 MNE。
- **Python API 优先。** notebook 里交互调参，同一套配置再批量跑完整个数据集。YAML 只作为导出和复现格式；另提供命令行 `meu`，用来运行 YAML 配置和数据集级任务。
- **命名约定。** 统一用缩写 `meu`：`import meeg_utils as meu`，命令行为 `meu`，文档示例一律写 `meu.xxx`。
- **BIDS 优先，兼容任意文件。** 文件命名、衍生数据、元信息（工频、通道类型等）优先从 BIDS 读取；非 BIDS 文件也能处理，元信息由用户提供。
- **支持 MNE 能读的所有系统。** 各系统特有的处理由系统识别和分系统预设负责（见 §5）。
- **与 MNE-BIDS-Pipeline 的区别：** 后者以配置文件驱动、整体较重；本库以 API 为主，可以组合、交互使用。

### 非目标

- 不实现新的信号处理算法（需要的算法优先推动到上游，如 mne-denoise）。
- 不做 GUI。
- 不做预处理参数的自动寻优（GridSearch）。预处理没有可优化的评分；以后可以考虑多宇宙分析（multiverse）。

## 2. 设计原则（硬约束）

1. **数据完整性。** 任何 Step 都不得隐式丢弃通道，不得改变 `first_samp`、`meas_date`、标注的时间对齐。只处理某类通道时，结果必须写回原对象。
2. **默认直接报错（fail loud）。** 步骤失败默认抛出异常。容错要显式开启（`on_error="warn"`），并且必须记录进 provenance 和报告。
3. **不做隐式的科学决策。** 能从数据读到的就读（如 `info["line_freq"]`），读不到时报错或明确警告，不悄悄套用默认值（如 50 Hz）。
4. **全程可追溯。** 每一步的参数、软件版本、QC 指标和中间产物都自动记录。
5. **库不配置日志。** import 时不添加、也不移除日志输出（handler），由用户调用 `setup_logging()` 自己配置。
6. **可选依赖分组安装**（§8），核心依赖保持精简。

## 3. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│ L3  数据集层   Dataset / BatchRunner                          │  遍历 BIDS、并行、断点续跑、失败汇总
├──────────────────────────────────────────────────────────────┤
│ L2  流程层     Pipeline([(name, Step), ...]) + 预设            │  组合、改参、切片、类型转换、YAML
├──────────────────────────────────────────────────────────────┤
│ L1  步骤层     Step(BaseEstimator): fit / transform           │  学到的状态和 qc_ 以 _ 结尾挂在对象上
├──────────────────────────────────────────────────────────────┤
│ L0  函数层     纯函数 f(inst, **params) -> inst | result       │  对上游库的薄封装，可单独调用
└──────────────────────────────────────────────────────────────┘
 横向模块：io · qc · provenance · report · logging
```

每层都可以单独使用：只想调一个函数用 L0，想要可组合、有记录的流程用 L1/L2，批处理整个数据集用 L3。

## 4. 核心抽象：借鉴 scikit-learn 的约定

### 4.1 借鉴什么

| sklearn 约定 | 本库对应做法 |
|---|---|
| 参数只在 `__init__` 中声明，`__init__` 不做计算 | Step 的参数完整可序列化（YAML 往返） |
| `get_params` / `set_params`、嵌套参数 `a__b` | `pipe.set_params(ica__n_components=0.99)` |
| `fit` / `transform` 分离，学到的状态以 `_` 结尾 | ICA 在 1 Hz 高通副本上拟合、应用到原数据；在一个 run 上拟合、应用到同 session 其他 run；拟合结果可复核 |
| 命名步骤、`named_steps`、切片 | `pipe[:3].fit_transform(raw)` 用来调试中间结果 |
| `clone` | 批处理时每个被试拿到一份干净的流程副本 |
| `joblib.Memory` | 缓存耗时步骤（ICA、PREP） |
| `check_estimator` 契约测试 | 对每个 Step 自动检查 §2 的硬约束 |
| `ColumnTransformer` | `ByChannelType({"meg": [...], "eeg": [...]})`：混合采集时按通道类型分支 |

实现上，Step 继承 `sklearn.base.BaseEstimator`，直接获得 `get_params`、`set_params`、`clone`、`repr`（scikit-learn 本来就是 MNE 和 mne-denoise 的依赖）。**Pipeline 自己实现，不用 sklearn 的 Pipeline。**

### 4.2 不照搬什么（以及为什么）

- **入口只有 `fit` / `transform` / `fit_transform`，不另设 `run()`。** 与 sklearn 完全一致：
  - `fit(inst)`：学习状态（坏道、ICA 解混矩阵、ZapLine 空间滤波器等），返回 `self`；
  - `transform(inst)`：用已学到的状态处理数据，**不重新拟合**，返回数据；
  - `fit_transform(inst)`：最常见的用法，等价于对同一份数据先 fit 再 transform，返回数据。对 Pipeline 来说，每一步都先 `fit_transform`，再把结果交给下一步。
- **数据类型会变。** 流程会从 Raw 走到 Epochs 再到 Evoked，每个 Step 要声明接受和输出的类型，Pipeline 在构造时就检查能否衔接。
- **旁路输出挂在对象上，不随返回值传递。** 按 sklearn 惯例，fit 之后的产物都是以 `_` 结尾的属性：
  - 每个 Step 有自己的状态（如 `ica_`、`bads_`）和 `qc_`（该步的 QC 指标）；
  - Pipeline 汇总得到 `qc_`（`{步骤名: 指标}`）、`provenance_`（参数、版本、耗时、警告）、`system_`（识别出的采集系统）。
  - 因此不需要单独的 Context 对象。系统识别在 `Pipeline.fit` 开始时做一次，作为参数传给各 Step 的 `fit`。
- **输入就是 MNE 对象。** `fit` 接受 `Raw` / `Epochs`；用 `io.read(bids_path)` 读入时，工频、通道类型等元信息已由 mne-bids 写进 `info`。为方便起见，`Pipeline.fit*` 也接受 `BIDSPath` 或文件路径，内部调用 `io.read`。
- **复制还是原地修改。** 默认先复制，`copy=False` 时原地修改，用于节省 MEG 大数据的内存。

### 4.3 接口草图

```python
class Step(BaseEstimator):
    accepts: tuple[type, ...] = (BaseRaw,)      # 允许的输入类型
    returns: type | None = None                 # None 表示与输入同类型
    systems: set[str] | None = None             # 适用的采集系统，None 表示全部

    def fit(self, inst, y=None, *, system: str | None = None) -> Self: ...
    def transform(self, inst): ...
    def fit_transform(self, inst, y=None, *, system=None): ...  # 默认 fit 后 transform，可覆盖
    # 拟合后：<state>_（如 ica_、bads_）、qc_: dict


class Pipeline(BaseEstimator):
    def __init__(self, steps: list[tuple[str, Step]], memory=None, copy=True): ...
    def fit(self, inst_or_path, y=None) -> Self: ...
    def transform(self, inst_or_path): ...
    def fit_transform(self, inst_or_path, y=None): ...
    # get_params / set_params（嵌套 a__b）/ clone 继承自 BaseEstimator
    def __getitem__(self, idx) -> Pipeline                   # 切片
    named_steps: dict[str, Step]
    def insert_after(self, name, new_name, step); def replace(self, name, step)
    def to_yaml(self, path); @classmethod def from_yaml(cls, path)
    @classmethod def preset(cls, name, system="auto") -> Pipeline
    # 拟合后：system_, qc_, provenance_


# 输出与报告是独立的函数，不是 Pipeline 的方法
io.save_derivative(inst, bids_path, root, pipeline=pipe, desc="preproc")  # 数据 + provenance + 中间产物
qc.report(pipe, inst) -> mne.Report
```

### 4.4 使用示例

```python
import meeg_utils as meu
from meeg_utils import Pipeline, steps as S

pipe = Pipeline([
    ("filter", S.Filter(l_freq=0.1, h_freq=100)),
    ("line",   S.LineNoise(method="zapline-plus")),   # 底层：mne-denoise
    ("bads",   S.BadChannels(method="auto")),         # EEG→PREP，Neuromag→Maxwell
    ("ref",    S.Reference("auto")),                  # EEG→平均参考，CTF→grade 3，其余系统→不处理
    ("ica",    S.ICA(n_components=0.99, fit_l_freq=1.0,
                     labeler="iclabel", threshold=0.8)),
])

raw = meu.io.read(bids_path)
pipe.set_params(ica__threshold=0.9)

tmp   = pipe[:3].fit_transform(raw)          # 只跑前三步调试
clean = pipe.fit_transform(raw)

pipe.named_steps["ica"].ica_                 # 可检查、可保存
pipe.qc_["line"], pipe.provenance_
meu.qc.report(pipe, clean).save("sub-01_report.html")
meu.io.save_derivative(clean, bids_path, deriv_root, pipeline=pipe)
pipe.to_yaml("config.yaml")

# 预设 + 局部修改
pipe = Pipeline.preset("eeg-erp").set_params(filter__h_freq=40)

# 复现已发表的流程
pipe = Pipeline.preset("had-meeg", datatype="meg")
```

命令行：

```bash
meu qc     /data/bids                      # 数据集 QC 汇总
meu run    config.yaml /data/bids -j 8     # 按配置批量处理
meu preset had-meeg --datatype meg > config.yaml
```

## 4.5 多 run 的处理策略

**默认以单个 run 为处理单位**：坏道、工频、ICA 都在每个 run 上独立拟合。原因是 run 之间的头位置、噪声环境、电极阻抗变化都较大，跨 run 共享这些状态反而有害。

但 run 之间有两件事必须统一，放在分段（`epochs`）和组水平之前，由专门的 Step 负责，不混进单 run 流程：

1. **通道集合一致。** 合并各 run 的 epochs 前，坏道取各 run 的并集，或者全部插值后对齐通道顺序。
2. **MEG 头位置对齐。** 把所有 run 变换到同一个目标头位置。`S.HeadAlign(destination=...)` 用最小范数场映射（Knösche, 2002）计算传感器在目标头位置下应测到的信号，适用于有 `dev_head_t` 的所有 MEG 系统（OPM 除外）；Neuromag 也可以用 `maxwell_filter(destination=...)`（P2 的 `Maxwell` Step）。目标位置通常取各 run 的平均：

   ```python
   dest = meu.io.average_dev_head_t(runs)          # 按各 run 非 BAD 时长加权平均
   pipe = Pipeline([..., ("align", S.HeadAlign(dest)), ...])
   ```

   **仿真验证**（`tests/test_steps/test_head.py`）：球模型中的偶极子分别在参考头位置和移动后的头位置正向投影，用前者作为真值。头移动 10 mm / 5° 时，不对齐的相对误差为 MAG 32%、GRAD 49%；正确对齐后降到 1.1% 和 3.2%；映射方向反过来（HAD-MEEG 的写法）则是 64% 和 101%，比不对齐还差一倍。头相对传感器移动超过 20 mm 时会发出警告。

`fit(run1)` 之后 `transform(run2)` 在技术上仍然可用（例如 run 很短、ICA 数据量不足时），但不进入任何预设。

## 5. 采集系统识别

MNE 能读几乎所有格式，但预处理随系统不同：

| 系统 | 特有处理 |
|---|---|
| Neuromag / MEGIN | SSS/tSSS、头动校正、cross-talk 和 fine-cal 文件、Maxwell 坏道检测 |
| CTF | 梯度补偿等级（grade 0/3）；参考通道（`ref_meg`）必须保留 |
| KIT / Ricoh | 参考通道回归降噪 |
| OPM | 均匀场校正（`compute_proj_hfc`）；没有固定的头坐标 |
| EEG | montage（电极位置）、参考方式、EOG/ECG 通道类型 |

`io.detect_system(raw) -> str` 读数据时识别系统（依据 `info` 中的设备信息、`compensation_grade`、通道类型等），结果存为 `pipe.system_` 并传给各 Step 的 `fit`。

- 声明了 `systems` 的 Step，用在不适用的系统上**直接报错**；`method="auto"` 的 Step 按系统选择后端。
- MEG 和 EEG 混合采集时，用 `ByChannelType` 分支处理，所有通道都保留。
- **数据类型由 BIDS 的 `datatype` 或用户参数决定**，不再根据"有没有 EEG 通道"来推断。

## 5.5 预设

预设分两类：

| 类型 | 目的 | 规则 |
|---|---|---|
| **数据集型**（如 `had-meeg`、`nod-meeg`） | 以已发表数据集的预处理为蓝本 | 沿用原流程的步骤和参数，但**已知问题一律修正**；每处与原代码不同的地方都在预设文档中逐条列出（原做法、问题、修正、依据） |
| **推荐型**（如 `eeg-erp`、`meg-erp`、`eeg-rest`、`meg-rest`） | 本库推荐的默认流程 | 每个参数都要能追溯到文献或权威流程的默认值，在文档中逐条注明出处 |

推荐型预设的参考来源（逐条核对后再采用）：MNE-BIDS-Pipeline 的默认配置、Jas et al. 2018（*Frontiers in Neuroscience*，MNE 组分析可复现示例）、FLUX（Ferrante et al. 2022，MEG 流程）、PREP（Bigdely-Shamlo et al. 2015）、ICLabel（Pion-Tonachini et al. 2019）的输入要求、ZapLine-plus（Klug & Kloosterman 2022）。

## 6. 模块与后端

| 模块 | Step / 功能 | 后端 |
|---|---|---|
| `io` | 读入（BIDSPath/路径/Raw）、`detect_system`、BIDS derivatives 写出 | mne, mne-bids |
| `qc` | 原始数据 QC、处理后 QC、数据集汇总表（§7） | mne, mne-denoise.qa, pandas |
| `preprocessing` | `Filter`、`Resample` | mne |
| | `LineNoise(method=zapline \| zapline-plus \| notch \| spectrum-interpolation)` | **mne-denoise**（ZapLine、ZapLine-plus、SpectrumInterpolation），mne（notch） |
| | `BadChannels(method=auto \| prep \| maxwell \| ...)` | pyprep, mne |
| | `Maxwell(st_duration=..., head_pos=...)` | mne |
| | `HeadAlign(destination=...)`（跨 run 头位置对齐） | mne（场映射） |
| | `HFC`（OPM）、`RefRegression`（KIT） | mne |
| | `BadSegments(method=amplitude \| muscle \| asr)` | mne，**mne-denoise**（ASR） |
| | `ASR`（作为数据修复使用） | **mne-denoise** |
| | `SNS`（传感器噪声抑制） | **mne-denoise** |
| | `Reference` | mne |
| | `ICA(labeler=iclabel \| megnet \| manual, threshold=...)` | mne, mne-icalabel |
| | `Interpolate` | mne |
| `epochs` | `Events`、`Epoch`、`Baseline`、`AutoReject` | mne, autoreject |
| `analysis` | ERP/ERF、PSD + 非周期成分（specparam）、时频、连接性、解码 | mne, specparam, mne-connectivity, mne.decoding |
| | `DSS`（ERP/SSVEP 增强等分析用途） | **mne-denoise** |
| `group` | 总平均、组水平统计（基于聚类的置换检验、TFCE）、组水平解码和 RSA 统计 | mne.stats, scipy |
| （后期）`source` | 正向模型、逆解、ROI | mne |
| （专用，可选） | iCanClean（移动 EEG）、SOUND、SSP-SIR（TMS-EEG） | **mne-denoise** |

### 6.1 引入 mne-denoise 的方针

- 用 mne-denoise 的 ZapLine 系列**替换 meegkit**。它只取同一类型的通道处理，再写回原对象，符合 §2.1 的数据完整性要求；接口是 sklearn 风格，与 §4 的 Step 自然对应。
- 它目前是 0.0.x，接口可能变化，因此：
  - 只在 `meeg_utils/_backends/denoise.py` 一个文件里调用它，其他代码都不直接引用；
  - 固定版本范围，并在 CI 里加一个针对它最新版本的测试；
  - 标为实验性的功能（如 ASR 的 riemannian 方法）不进入预设。
- 它的 `qa`（工频压制、频谱失真）和 `overcorrection` 指标直接纳入本库的 QC。

## 7. 质量检查（QC）、出图与报告

### 7.1 统一出图接口

**每个 Step、每个 QC 检查项都能出图**，接口一致：

```python
step.plot_kinds                      # {"psd": False, "properties": True, ...}；值表示是否需要传入数据
figs = step.plot()                   # 拟合后：所有不需要数据的图 -> {kind: Figure}
figs = step.plot(inst=raw)           # 再加上需要数据的图
fig  = step.plot("components")["components"]
pipe.plot(inst=raw)                  # {步骤名: {kind: Figure}}
check.plot()                         # QC 检查项同样如此
```

- 未拟合时调用 `plot()` 抛 `NotFittedError`；需要数据却没传 `inst` 时报错并说明。
- **画图所需的信息在拟合或变换时以紧凑形式保存**为 `_` 结尾的属性（例如 PSD 摘要、Maxwell 评分、PREP 各判据的 z 分数），不保留整份数据。所以没有原始数据也能复现大部分图。
- **优先使用 MNE 自带的绘图函数**（如 `plot_bridged_electrodes`、`ICA.plot_components`、`plot_filter`、`plot_head_positions`），在此基础上标注阈值和判定结果。
- 返回 matplotlib Figure，不自动显示（`show=False`），方便在 notebook、报告和批处理里统一使用。
- 契约测试 `check_step` 也检查出图：每个 `plot_kind` 都必须产出 Figure。

### 7.2 采集质量 QC（`meu.qc`）

回答"这份数据采得好不好、要给采集者什么反馈"，**不修改数据**，所以不是 Step。每个检查项是一个小类：参数在 `__init__` 中声明，默认阈值注明出处，声明适用的系统和模态，`compute(raw)` 后得到指标、判定（`ok` / `warn` / `fail`）和依据，并可 `plot()`。

| 模态 | 检查项（第一批加粗） | 主要依据 / 函数 |
|---|---|---|
| EEG | **电极桥接** | `mne.preprocessing.compute_bridged_electrodes`，图：`mne.viz.plot_bridged_electrodes` |
| EEG | 阻抗 | 读取采集软件或 BIDS 记录的阻抗 |
| EEG/MEG | **平坦、削顶、饱和** | `annotate_amplitude`，加上检测"信号长时间停在最大值" |
| EEG/MEG | 肌电时间占比 | `annotate_muscle_zscore` |
| EEG | 眨眼率、EOG 质量 | EOG 通道 |
| MEG | **头动** | Neuromag：`compute_chpi_amplitudes` + `compute_head_pos`；CTF：`extract_chpi_locs_ctf`；KIT：`extract_chpi_locs_kit`；图：`plot_head_positions` |
| MEG | 头与传感器的距离 | `dev_head_t` |
| MEG | 坏传感器、SQUID 跳变 | Maxwell 评分、跳变检测 |
| MEG | 环境噪声 | 与当天空房间记录（mne-bids `find_empty_room`）比较 PSD |
| 通用 | **工频及谐波、其他窄带峰** | 未清理数据的 PSD |
| 通用 | **事件与触发** | 事件数与预期是否一致、触发与光电二极管之间的抖动 |
| 通用 | 采集中断、采集参数 | `BAD_ACQ_SKIP` 标注；采样率、在线滤波、时长与 BIDS 描述是否一致 |

```python
report = meu.qc.inspect(raw)          # 按系统和模态自动选择适用的检查
report.flags; report.to_frame(); report.plot()
meu.qc.inspect_dataset(bids_root)     # 数据集汇总表，标出离群的被试和 run
```

QC 结果可以接到预处理里，例如 `S.BridgedElectrodes`（`interpolate_bridged_electrodes`）。阈值默认取 MNE 默认值和文献标准值并注明出处；实验室自己的阈值通过参数或预设覆盖。

### 7.3 报告

`meu.report.build(pipe, inst=None, qc=None) -> mne.Report`：每个步骤一节，包含该步的 `qc_` 指标表和全部图；有 QC 结果时放在最前面。`meu.qc.inspect_dataset` 汇总所有被试的指标并标出离群者。

## 8. 输出与可追溯性

- 遵循 BIDS derivatives：`derivatives/meeg-utils/sub-XX/[ses-YY/]<datatype>/..._desc-preproc_<datatype>.fif`，附带 `dataset_description.json`（含 `GeneratedBy`）。
- 每个输出旁边写一份 `_provenance.json`，内容包括流程配置（YAML 等价物）、各步骤参数、软件版本、耗时、警告和 QC 指标。
- 中间产物按需保存：`_ica.fif`、`_desc-badchs_channels.tsv`（符合 BIDS channels.tsv 规范）。

## 9. 依赖策略

```
核心:      mne>=1.13, mne-bids, mne-denoise[mne], pyprep, mne-icalabel, onnxruntime,
           numpy, scipy, scikit-learn, pandas, joblib, pyyaml, loguru
[epochs]   autoreject
[analysis] specparam, mne-connectivity
[all]      以上全部
```

预处理的基本步骤（工频、坏道、ICA 自动标注）依赖的库放进核心依赖，保证 `pip install meeg-utils` 后预设流程就能直接运行。mne-denoise 仍是 0.0.x，固定 `<0.1`。meegkit 在旧的 `PreprocessingPipeline` 被预设取代后移除。

用到可选依赖的 Step 在构造时检查依赖是否已安装，缺失时给出明确的安装提示（`pip install meeg-utils[denoise]`）。

## 10. 测试策略

- **Step 契约测试**（类似 sklearn 的 `check_estimator`）：对每个 Step 自动参数化检查：
  - 通道名和数量保持不变（`picks` 之外的数据逐位不变）；
  - `first_samp`、`meas_date`、标注保持不变；
  - `get_params` / `set_params` / `clone` / YAML 往返；
  - 不适用的系统明确报错。
- **科学行为测试**：注入 50 Hz 后衰减超过 X dB，频带外的失真低于阈值；人工坏道能被检出；人工眨眼成分能被剔除。
- **端到端测试**：用 `mne.datasets.testing` 里的 Neuromag 和 CTF 数据，加一份 EEG 数据，各跑一次预设流程。
- 不再屏蔽 ICLabel、MEGnet 关于输入不合规的警告；需要时只在单个测试里显式处理。

## 11. 路线图

| 阶段 | 内容 |
|---|---|
| **P0 打地基** | §2 硬约束；`core`（Step、Pipeline、YAML、统一出图接口 §7.1）；`io`（读入、`detect_system`、derivatives）；现有预处理迁移为 Step（Filter、LineNoise→mne-denoise、BadChannels、Reference、ICA 修正）；`had-meeg` 数据集型预设作为第一个端到端用例；日志与错误处理；契约测试；版本 0.2.0 |
| **P1 质量检查** | 采集质量 QC（§7.2，第一批：电极桥接、平坦/削顶/饱和、工频与窄带峰、头动、事件）、`meu.report`、数据集 QC 汇总表；Neuromag 的 cross-talk / fine-cal 文件从 BIDS 自动查找 |
| **P2 预处理补全** | Maxwell/SSS、HFC、RefRegression、BadSegments/ASR、SNS、`ByChannelType`、预设（eeg-erp / eeg-rest / meg-erp / meg-rest，按系统自动选择）、`epochs` 模块、L3 `Dataset` / `BatchRunner` |
| **P3 分析** | ERP/ERF、PSD + specparam、时频、解码、连接性、DSS；`group` 模块；`meu` 命令行 |
| **P4 源分析** | 正向模型、逆解、ROI（可选） |

## 12. 待定问题

- [ ] 推荐型预设的具体参数：逐条对照 §5.5 所列来源确定并注明出处。
- [x] 多 run：默认按单个 run 处理，跨 run 只统一通道集合和 MEG 头位置（§4.5）。
- [x] 组水平：支持，新增 `group` 模块。
- [x] 命令行：`meu`；缩写风格统一为 `meu`。
- [x] 数据集型预设：只提供修正后的版本，与原代码的差异逐条记录在预设文档中。
