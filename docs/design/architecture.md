# meeg-utils 架构设计（v0.2 草案）

> 状态：讨论稿。本文档是 0.2.0 重构的依据，接口定稿前可随时修改。

## 1. 定位

**用 Python API 把 MNE 生态的成熟方法串成"开箱即用、处处可调、步步留痕"的 M/EEG 处理工具**，覆盖质量检查（QC）、预处理和数据分析。

- **封装，不重写。** 算法交给 MNE、mne-bids、mne-denoise、pyprep、mne-icalabel、autoreject、specparam、mne-connectivity 等成熟库；本库负责编排、默认值、质量检查和处理记录。
- **数据对象用 MNE 原生类型。** `Raw` / `Epochs` / `Evoked` 不再包一层，用户随时可以退回原生 MNE。
- **Python API 优先。** notebook 里交互调参，同一套配置再批量跑完整个数据集。YAML 只作为导出和复现格式。
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
│ L1  步骤层     Step(BaseEstimator): fit / transform           │  学到的状态以 _ 结尾，向 ctx 写 QC 和产物
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
| 命名步骤、`named_steps`、切片 | `pipe[:3].run(raw)` 用来调试中间结果 |
| `clone` | 批处理时每个被试拿到一份干净的流程副本 |
| `joblib.Memory` | 缓存耗时步骤（ICA、PREP） |
| `check_estimator` 契约测试 | 对每个 Step 自动检查 §2 的硬约束 |
| `ColumnTransformer` | `ByChannelType({"meg": [...], "eeg": [...]})`：混合采集时按通道类型分支 |

实现上，Step 继承 `sklearn.base.BaseEstimator`，直接获得 `get_params`、`set_params`、`clone`、`repr`（scikit-learn 本来就是 MNE 和 mne-denoise 的依赖）。**Pipeline 自己实现，不用 sklearn 的 Pipeline。**

### 4.2 不照搬什么（以及为什么）

- **数据类型会变。** 流程会从 Raw 走到 Epochs 再到 Evoked，每个 Step 要声明自己接受什么类型、输出什么类型，Pipeline 在组装时就检查类型能否衔接。
- **需要旁路输出。** QC 指标、坏道、ICA 对象、报告图都写入 `Context`，与数据一起传递。sklearn 没有这样的通道。
- **大多数步骤是对同一份数据拟合再变换。** `pipe.run(x)` 就等于 `fit_transform`，不强迫用户分两次调用。
- **复制还是原地修改。** 默认先复制，`copy=False` 时原地修改，用于节省 MEG 大数据的内存。

### 4.3 接口草图

```python
class Step(BaseEstimator):
    accepts: tuple[type, ...] = (BaseRaw,)      # 允许的输入类型
    returns: type | None = None                 # None 表示与输入同类型
    systems: set[str] | None = None             # 适用的采集系统，None 表示全部

    def fit(self, inst, ctx: Context | None = None) -> Self: ...
    def transform(self, inst, ctx: Context | None = None): ...
    def fit_transform(self, inst, ctx=None): ...   # 默认 fit 后 transform，可覆盖


@dataclass
class Context:
    system: str                       # neuromag / ctf / kit / opm / eeg / ...
    bids_path: BIDSPath | None
    qc: dict[str, dict]               # step_name -> 指标
    artifacts: dict[str, Any]         # step_name -> ICA 对象、坏道列表、图等
    provenance: list[StepRecord]      # 参数、版本、耗时、警告


class Pipeline:
    def __init__(self, steps: list[tuple[str, Step]], memory=None): ...
    def run(self, inst_or_path, ctx=None) -> Result: ...     # = fit_transform
    def fit(self, ...); def transform(self, ...)
    def set_params(self, **kw); def get_params(self, deep=True)
    def __getitem__(self, idx) -> Pipeline                   # 切片
    named_steps: dict[str, Step]
    def insert_after(self, name, new_name, step); def replace(self, name, step)
    def to_yaml(self, path); @classmethod def from_yaml(cls, path)
    @classmethod def preset(cls, name, system="auto") -> Pipeline


@dataclass
class Result:
    data: BaseRaw | BaseEpochs | Evoked
    ctx: Context
    def save(self, root, desc="preproc"): ...   # BIDS derivatives
    def report(self) -> mne.Report: ...
```

### 4.4 使用示例

```python
from meeg_utils import Pipeline, steps as S

pipe = Pipeline([
    ("filter", S.Filter(l_freq=0.1, h_freq=100)),
    ("line",   S.LineNoise(method="zapline-plus")),   # 底层：mne-denoise
    ("bads",   S.BadChannels(method="auto")),         # EEG→PREP，Neuromag→Maxwell
    ("ref",    S.Reference("auto")),                  # EEG→平均参考，CTF→grade 3，其余系统→不处理
    ("ica",    S.ICA(n_components=0.99, fit_l_freq=1.0,
                     labeler="iclabel", threshold=0.8)),
])

pipe.set_params(ica__threshold=0.9)
res = pipe[:3].run(bids_path)        # 只跑前三步调试
res = pipe.run(bids_path)
pipe.named_steps["ica"].ica_         # 可检查、可保存
res.save(deriv_root)
pipe.to_yaml("config.yaml")

# 预设 + 局部修改
pipe = Pipeline.preset("eeg-erp").set_params(filter__h_freq=40)
```

## 5. 采集系统识别

MNE 能读几乎所有格式，但预处理随系统不同：

| 系统 | 特有处理 |
|---|---|
| Neuromag / MEGIN | SSS/tSSS、头动校正、cross-talk 和 fine-cal 文件、Maxwell 坏道检测 |
| CTF | 梯度补偿等级（grade 0/3）；参考通道（`ref_meg`）必须保留 |
| KIT / Ricoh | 参考通道回归降噪 |
| OPM | 均匀场校正（`compute_proj_hfc`）；没有固定的头坐标 |
| EEG | montage（电极位置）、参考方式、EOG/ECG 通道类型 |

`io.detect_system(raw) -> str` 读数据时识别系统（依据 `info` 中的设备信息、`compensation_grade`、通道类型等），结果写入 `Context.system`。

- 声明了 `systems` 的 Step，用在不适用的系统上**直接报错**；`method="auto"` 的 Step 按系统选择后端。
- MEG 和 EEG 混合采集时，用 `ByChannelType` 分支处理，所有通道都保留。
- **数据类型由 BIDS 的 `datatype` 或用户参数决定**，不再根据"有没有 EEG 通道"来推断。

## 6. 模块与后端

| 模块 | Step / 功能 | 后端 |
|---|---|---|
| `io` | 读入（BIDSPath/路径/Raw）、`detect_system`、BIDS derivatives 写出 | mne, mne-bids |
| `qc` | 原始数据 QC、处理后 QC、数据集汇总表（§7） | mne, mne-denoise.qa, pandas |
| `preprocessing` | `Filter`、`Resample` | mne |
| | `LineNoise(method=zapline \| zapline-plus \| notch \| spectrum-interpolation)` | **mne-denoise**（ZapLine、ZapLine-plus、SpectrumInterpolation），mne（notch） |
| | `BadChannels(method=auto \| prep \| maxwell \| ...)` | pyprep, mne |
| | `Maxwell(st_duration=..., head_pos=...)` | mne |
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
| （后期）`source` | 正向模型、逆解、ROI | mne |
| （专用，可选） | iCanClean（移动 EEG）、SOUND、SSP-SIR（TMS-EEG） | **mne-denoise** |

### 6.1 引入 mne-denoise 的方针

- 用 mne-denoise 的 ZapLine 系列**替换 meegkit**。它只取同一类型的通道处理，再写回原对象，符合 §2.1 的数据完整性要求；接口是 sklearn 风格，与 §4 的 Step 自然对应。
- 它目前是 0.0.x，接口可能变化，因此：
  - 只在 `meeg_utils/_backends/denoise.py` 一个文件里调用它，其他代码都不直接引用；
  - 固定版本范围，并在 CI 里加一个针对它最新版本的测试；
  - 标为实验性的功能（如 ASR 的 riemannian 方法）不进入预设。
- 它的 `qa`（工频压制、频谱失真）和 `overcorrection` 指标直接纳入本库的 QC。

## 7. 质量检查（QC）与报告

- **原始数据 QC**（`qc.raw`）：各通道方差和平坦度、噪声通道候选、PSD、工频强度、事件和标注检查、MEG 头动、时长和采样率核对。
- **步骤 QC**：每个 Step 向 `ctx.qc[name]` 写入指标，例如坏道数及比例、ICA 剔除的成分及其解释方差、工频衰减 dB、ASR 修复比例、epoch 拒绝率。
- **报告**：`Result.report()` 根据 ctx 生成 `mne.Report`（每个被试一份 HTML）。`Dataset.qc_table()` 汇总所有被试的指标，并按阈值标出离群被试。
- QC 可以单独运行：`qc.run(bids_root)`，先看数据、再定参数。

## 8. 输出与可追溯性

- 遵循 BIDS derivatives：`derivatives/meeg-utils/sub-XX/[ses-YY/]<datatype>/..._desc-preproc_<datatype>.fif`，附带 `dataset_description.json`（含 `GeneratedBy`）。
- 每个输出旁边写一份 `_provenance.json`，内容包括流程配置（YAML 等价物）、各步骤参数、软件版本、耗时、警告和 QC 指标。
- 中间产物按需保存：`_ica.fif`、`_desc-badchs_channels.tsv`（符合 BIDS channels.tsv 规范）。

## 9. 依赖策略

```
核心:      mne, mne-bids, numpy, scipy, scikit-learn, pandas, joblib, loguru
[denoise]  mne-denoise
[prep]     pyprep
[icalabel] mne-icalabel, onnxruntime
[epochs]   autoreject
[analysis] specparam, mne-connectivity
[all]      以上全部
```

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
| **P0 打地基** | §2 硬约束；`core`（Step、Pipeline、Context、Result、YAML）；`io`（读入、`detect_system`、derivatives）；现有预处理迁移为 Step（Filter、LineNoise→mne-denoise、BadChannels、Reference、ICA 修正）；日志与错误处理；契约测试；版本 0.2.0 |
| **P1 质量检查** | `qc` 模块、mne.Report、数据集 QC 汇总表 |
| **P2 预处理补全** | Maxwell/SSS、HFC、RefRegression、BadSegments/ASR、SNS、`ByChannelType`、预设（eeg-erp / eeg-rest / meg-erp / meg-rest，按系统自动选择）、`epochs` 模块、L3 `Dataset` / `BatchRunner` |
| **P3 分析** | ERP/ERF、PSD + specparam、时频、解码、连接性、DSS |
| **P4 源分析** | 正向模型、逆解、ROI（可选） |

## 12. 待定问题

- [ ] 预设的具体参数（各预设分别包含哪些步骤、默认值是多少），需要对照文献逐条确定。
- [ ] 多 run 场景：ICA 和坏道检测按 run 做，还是按 session 合并后做（是否提供 `fit` 用 concat、`transform` 逐 run 的模式）。
- [ ] `Dataset` 层是否支持组水平分析（如总平均、组统计），还是只负责逐被试处理。
- [ ] 是否提供命令行入口（`meeg-utils run config.yaml`）作为 YAML 的补充。
