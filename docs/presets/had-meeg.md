# `had-meeg` 预设

以 [HAD-MEEG](https://github.com/colehank/HAD-MEEG) 的预处理代码为蓝本（审查时的版本：`7d3aa5a`），**只提供修正后的版本**。下面逐条列出与原代码不同的地方：原做法、问题、修正和依据。

```python
import meeg_utils as meu

pipe = meu.Pipeline.preset("had-meeg", datatype="meg")   # 或 datatype="eeg"
clean = pipe.fit_transform("sub-01_ses-meg_task-action_run-1_meg.ds")
pipe.qc_            # 每一步的质量指标
pipe.plot(inst=raw) # 每一步的质量图
```

预设只是普通的 `Pipeline`，任何参数都可以改：`pipe.set_params(ica__threshold=0.9)`；也可以导出为 YAML 再修改。

## 1. 流程

两种数据类型共用前两步，之后分开：

| # | 步骤名 | MEG（CTF） | EEG |
|---|---|---|---|
| 1 | `filter` | `Filter(0.1, 100)` | 同左 |
| 2 | `resample` | `Resample(250)` | 同左 |
| 3 | `bads` | `BadChannels("maxwell", origin=(0, 0, 0.04))`，在 grade-0 补偿的副本上检测（CTF 没有 cross-talk / fine-cal 文件，`"auto"` 时自动不用） | `BadChannels("prep")`，PREP 全部判据 |
| 4 | `interpolate` | `Interpolate(origin=(0, 0, 0.04))`，MNE 场插值 | `Interpolate()`，球面样条 |
| 5 | `line_noise` | `LineNoise("zapline-plus")`，工频取自数据 | 同左 |
| 6 | `reference` | `Reference(eeg=None, ctf_grade=3)` | `Reference(eeg="average")` |
| 7 | `ica` | `ICA(20, picks="meg", labeler="megnet")`，在 1–100 Hz 副本上拟合，extended infomax | `ICA(20, picks="eeg", labeler="iclabel")`，同左 |

每个 run 单独处理（见设计文档 §4.5）。跨 run 的头位置对齐不在这个单 run 预设里，见 §3。

## 2. 与原代码的差异（预处理阶段）

严重程度沿用审查时的分级：高 = 结果明显错误；中 = 结果可能有偏差或无法复现；低 = 影响很小或只是不够稳妥。

### 2.1 工频去除：固定比例改为自适应（低）

- **原做法**：MEG 用 meegkit `dss_line`，固定去掉通道数 22% 的成分（`src/prep/line_noise.py:71`）；EEG 用 `dss_line_iter`（`:58`）。工频写死为 50 Hz（`:50`、`:69`）。
- **问题**：去掉的成分数与数据中工频的实际强弱无关。我们用 128 通道、30 个源的仿真量化过：去掉 28 个成分与去掉 2 个相比，工频抑制没有增加，25–45 Hz 的信号只多损失约 0.20 dB，所以影响很小。但固定比例在别的数据上没有保障。
- **修正**：两种数据都用 ZapLine-plus（Klug & Kloosterman, 2022，mne-denoise 实现）。它按数据自适应地选择成分数，并会检查是否清除过度。工频从数据读取（BIDS `PowerLineFrequency`），读不到就报错。`qc_["line_noise"]` 里有抑制量、失真量，以及清除过度和清除不足的通道比例。
- 另外，原代码处理完后用 `RawArray(data, info)` 重建对象（`:44`），丢掉了 `first_samp`。CTF 和 Curry 数据的 `first_samp` 通常为 0，所以对这个数据集多半没有影响；本库的 Step 一律保留 `first_samp`、`meas_date` 和标注。

### 2.2 MEG ICA：40 个成分改为 20 个（中）

- **原做法**：MEG ICA 取 40 个成分，EEG 取 20 个（`src/prep/ica/ica.py:84-86`）。
- **问题**：MEGnet 是按 20 个成分设计和训练的，mne-icalabel 在成分数不是 20 时会发出警告。成分数不同，分类结果没有经过验证。
- **修正**：MEG 也取 20 个成分。

### 2.3 ICA 成分剔除：按最大概率标签改为按概率阈值（中）

- **原做法**：只要最大概率的标签不是 brain/other，就剔除该成分（`ica.py:113`、`:138`），之后通过网页工具人工复核（`scripts/step-1_preprocessing.py`）。
- **问题**：最大概率标签本身可能只有很低的概率。原流程靠人工复核兜底，但自动批处理时没有人工这一环，低置信度的成分也会被剔除。
- **修正**：只在某个伪迹类别的概率不低于 0.8 时自动剔除（`ICA(threshold=0.8)`）。`pipe["ica"].qc_` 里有每个成分的标签、概率和剔除的方差比例；`pipe["ica"].plot()` 会画出成分地形图，标题写着标签和概率，剔除的标红。
- **人工复核仍然可以做**，而且推荐做：

  ```python
  pipe.fit(raw)                           # 拟合到 ICA 为止
  figs = pipe.plot(inst=raw)["ica"]       # 地形图、属性图、剔除前后的对比（用 ICA 收到的数据画）
  pipe["ica"].relabel({3: "eye blink", 7: "brain"})
  clean = pipe.transform(raw)
  ```

  人工改动的标签记录在 `qc_["manual_labels"]` 里，用 `meu.io.save_derivative(..., pipeline=pipe)` 保存时会一起写进旁注文件。

### 2.4 EEG 坏道检测：PREP 全部判据，并在稳健平均参考下检测（中）

- **原做法**：PyPREP 只运行相关性、偏离度和 RANSAC 三个判据（`src/prep/bad_chs.py:71-73`），再加上初始化时自动做的 NaN/平坦检测；检测直接在记录本身的参考上进行，平均参考在之后才做（`src/prep/pipe_single.py:171`）。
- **问题**：
  - 漏掉了高频噪声和低信噪比判据。PREP 原文（Bigdely-Shamlo et al., 2015）的流程是全部判据一起用。
  - PREP 原文先估计稳健平均参考，再在这个参考下检测坏道。不这样做时，紧挨参考电极的通道信号很小、与其他通道相关性低，会被相关性判据误判。真实数据中的例子：ds007353 sub-01 的 EEG（Neuroscan，参考电极在 Cz 附近），C1 和 Cz 在 1–30 Hz 只有约 0.4 µV（周围电极 2–3 µV），原做法把它们判为坏道并插值；在稳健平均参考下它们是正常通道。
- **修正**：运行全部判据（`find_all_bads`），并用 PyPREP 的 `Reference` 做稳健平均参考后再检测（`BadChannels(robust_reference=True)`，默认；数据本身不被重参考）。原参考下的检测结果仍记录在 `qc_["bads"]["eeg_before_reference"]`，便于对比。`pipe["bads"].plot("scores")` 画出每个判据下各通道的分数和阈值。

### 2.5 只滤波一次（中）

- **原做法**：预处理时先做 0.1–100 Hz 滤波（`src/prep/pipe_single.py:109-111`）；应用 ICA 前又对数据重新做一次 0.1–100 Hz 滤波和重采样（`ica.py:443`、`:433`）；分段时再做一次 0.1–40 Hz（`src/epo/epoching.py:220`）。
- **问题**：同一个 0.1 Hz 高通重复使用，通带边缘的衰减会叠加，实际的频率响应与参数描述的不一致，也难以复现。
- **修正**：预处理只滤波一次。ICA 在内部的 1–100 Hz 副本上拟合，但剔除成分直接作用在输入数据上，不再重新滤波。分段阶段如果需要 40 Hz 低通，只加一个低通。

### 2.6 坏道记录（中）

- **原做法**：插值后输出的 `channels.tsv` 里所有通道都是 `good`，插值过的只在描述栏写 `fixed`（`bad_chs.py:115`）。
- **问题**：下游分析无法据此知道哪些通道是插值得到的。插值通道没有独立信息，会影响秩和 ICA 的成分数。
- **修正**：`qc_["interpolate"]["interpolated"]` 记录了插值的通道，`meu.io.save_derivative` 会把它写进 derivatives 的 JSON 旁注文件。ICA 拟合前会按数据的秩限制成分数，并在 `qc_` 里说明。

### 2.7 EEG 在 ICA 之后不再重做平均参考（不影响结果）

- **原做法**：ICA 去除伪迹后，再次设为平均参考（`pipe_single.py:194`）。
- **说明**：ICA 拟合在平均参考的数据上，剔除的成分也位于各通道之和为零的子空间内，去除后数据仍然是平均参考，所以再次参考没有作用。端到端测试检查了这一点：输出数据各通道之和在数值精度内为零。

### 2.8 其他

- **CTF 补偿等级**：原代码在预处理中途调用 `apply_gradient_compensation(3)`（`pipe_single.py:174`），而 HAD-MEEG 的 CTF 数据本来就是 grade 3，这一步不起作用。预设保留这一步（`Reference(ctf_grade=3)`），等级的前后变化记录在 `qc_` 里。
- **错误处理**：原来的批处理遇到异常只记一条日志就继续（`src/prep/pipe_batch.py:92`），最后可能悄悄少了几个 run。本库的 Step 和 Pipeline 遇错即抛出异常，不会跳过；之后的批处理命令（`meu run`）也会汇总报告每个失败的 run。

## 3. 分段阶段（`stage="epochs"`）

原代码的分段阶段（`src/epo/epoching.py`）对应预设的 `stage="epochs"`，作用在预处理后的单个 run 上，最后用 `meu.epochs.combine` 合并同一被试的各个 run：

```python
import meeg_utils as meu

# EEG
epo = meu.Pipeline.preset("had-meeg", datatype="eeg", stage="epochs")
runs = [epo.fit_transform(path) for path in preprocessed_eeg_runs]   # 每个 run 单独
eeg_epochs = meu.epochs.combine(runs)                                  # 坏道取并集

# MEG：先求各 run 的平均头位置，所有 run 对齐到它
dest = meu.io.average_dev_head_t(preprocessed_meg_runs)
epo = meu.Pipeline.preset("had-meeg", datatype="meg", stage="epochs", head_destination=dest)
meg_epochs = meu.epochs.combine([epo.fit_transform(p) for p in preprocessed_meg_runs])
```

| # | 步骤名 | MEG | EEG |
|---|---|---|---|
| 1 | `drop_mastoids` | — | `DropChannels(["M1", "M2"])` |
| 2 | `reference` | — | `Reference("average")` |
| 3 | `lowpass` | `Filter(None, 40)` | 同左 |
| 4 | `resample` | `Resample(200)` | 同左 |
| 5 | `align` | `HeadAlign(head_destination)` | — |
| 6 | `epoch` | `Epoch("video on", -0.1, 2.0)` | 同左 |
| 7 | `baseline` | `Baseline((None, 0), "mean")` | 同左 |

`combine` 会检查各 run 的通道、采样率、时间轴是否一致，按事件名称统一事件编码；MEG 各 run 的头位置相差超过 2 mm 时报错，并提示先用 `HeadAlign` 对齐。原代码的 `metadata`（来自单独的事件表）没有放进预设，需要时用 `Epoch(metadata=path)` 传入。

与原代码的差异：

### 3.1 跨 run 头位置对齐的方向反了（高）

- **原做法**：`_map_meg_or_eeg_channels(ref_info, this_info)` 计算出映射矩阵后，作用在当前 run 的数据上（`epoching.py:268`）。
- **问题**：这个函数的参数顺序是 `(info_from, info_to)`，所以这样写算出的是从参考位置到当前位置的映射，把数据往反方向移动了。仿真（`tests/test_steps/test_head.py`）：头移动 10 mm / 5° 时，以参考位置的正向投影为真值，各情况的相对误差如下：

  | | 不对齐 | 正确对齐 | 原代码（方向反） |
  |---|---|---|---|
  | MAG | 32% | 1.1% | 64% |
  | GRAD | 49% | 3.2% | 101% |

  方向反了比不对齐还差一倍。
- **修正**：用 `S.HeadAlign`，目标位置默认取各 run 的平均头位置，而不是第一个 run：

  ```python
  dest = meu.io.average_dev_head_t(run_paths)
  aligned = S.HeadAlign(dest, origin=(0.0, 0.0, 0.04)).fit_transform(clean)
  ```

  `qc_` 里记录每个 run 的头移动距离和旋转角度，`plot()` 画出对齐前后传感器相对头的位置。预设中的 `align` 步骤即此修正。

### 3.2 100 ms 基线上做 z 分数（中）

- **原做法**：`baseline=(None, 0)`，`tmin=-0.1`，采样率 200 Hz，用 `mode="zscore"` 做基线校正（`epoching.py:56`、`:290`）。
- **问题**：每个 epoch 每个通道只用 20 个样本估计标准差，估计值噪声很大；除以它会把基线方差偶然偏小的试次放大。
- **修正**：基线只减均值（预设中 `Baseline((None, 0), "mean")`）。`Baseline` 会拒绝在过短的基线上逐个 epoch 做 z 分数：*n* 个样本估计标准差的相对标准误约为 1/√(2(*n*−1))，21 个样本为 16%，超过默认上限 10% 时报错。确实需要标准化时，可以用 `Baseline(mode="zscore", scale="pooled")`，它用所有 epoch 合并的基线标准差（逐通道）。

### 3.3 M1/M2 先参与平均参考，再改为 misc（中）

- **原做法**：先设平均参考（包含 M1/M2），再把 M1/M2 改成 misc 类型并丢弃（`epoching.py:210-211`）。
- **问题**：最终保留的通道不再是平均参考（各通道之和不为零），而且参考是否包含乳突电极是一个隐含的决定，没有记录。
- **修正**：保留原代码去掉 M1/M2 的意图，但先去掉再做平均参考（预设中 `drop_mastoids` 在 `reference` 之前）。测试检查了输出的各通道之和为零。
- **另外**：原代码分段时又做了一次 0.1–40 Hz 滤波（第三次 0.1 Hz 高通，见 §2.5），预设只加 40 Hz 低通。
