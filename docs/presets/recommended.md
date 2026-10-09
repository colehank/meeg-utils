# 推荐型预设：`eeg-erp`、`eeg-rest`、`meg-erp`、`meg-rest`

推荐型预设是本库建议的默认流程。规则：**每个参数都能追溯到文献或权威流程的默认值**；找不到公认默认值的参数，要么由用户必须给出（如事件名、静息态 epoch 长度），要么在下表中明确标为“本库选择”并说明理由。

```python
import meeg_utils as meu

# 1. 每个 run 单独预处理
pre = meu.Pipeline.preset("eeg-erp")                      # MEG：meu.Pipeline.preset("meg-erp", system="neuromag")
clean = [pre.fit_transform(run) for run in runs]          # fit_transform 在每个 run 上重新拟合

# 2. 分段，再合并各 run（事件码按名称统一）
ep = meu.Pipeline.preset("eeg-erp", stage="epochs", event_id=["target", "standard"])
epochs = meu.epochs.combine([ep.fit_transform(run) for run in clean])
```

更简洁的写法是用 `meu.process`（或命令行 `meu run --preset eeg-erp`）批量处理，每个 run 自动得到一份干净的流程副本。

预设只是普通的 `Pipeline`：`set_params` 改参数、`to_yaml` 导出、`insert_after` 增删步骤都可以。

## 1. 预处理阶段（`stage="preprocessing"`，每个 run 单独处理）

### 1.1 EEG（`eeg-erp` 与 `eeg-rest` 相同）

| # | 步骤 | 参数 | 来源 |
|---|---|---|---|
| 1 | `bridged` | `BridgedElectrodes()`：MNE `compute_bridged_electrodes` 的默认值（电距离，0.5–30 Hz，2 s 窗） | MNE 默认值；电距离方法见 Tenke & Kayser (2001)。**本库选择**纳入这一步：桥接电极会让后面的 PREP 和 ICA 出错，且需要电极位置的步骤（PREP RANSAC、插值）本来就要求有 montage |
| 2 | `highpass` | `Filter(0.1, None)` | Tanner, Morgan-Short & Luck (2015, *Psychophysiology* 52:997)：0.3 Hz 及以上的高通会在 ERP 中造成假效应，建议 ≤ 0.1 Hz。此处不加低通：ICLabel 需要 1–100 Hz 的信息 |
| 3 | `line_noise` | `LineNoise("zapline-plus")`，工频取自数据（BIDS `PowerLineFrequency`） | Klug & Kloosterman (2022, *Hum Brain Mapp* 43:2743)。MNE-BIDS-Pipeline 默认不去工频（`notch_freq=None`）；**本库选择** ZapLine-plus 是因为它只去除与工频相关的空间成分，不像陷波那样在频谱上挖洞 |
| 4 | `bads` | `BadChannels("prep")`：PyPREP 默认值，全部判据（含 RANSAC） | Bigdely-Shamlo et al. (2015, *Front Neuroinform* 9:16) |
| 5 | `interpolate` | `Interpolate()`：球面样条 | Perrin et al. (1989)；MNE 默认 |
| 6 | `reference` | `Reference("average")` | ICLabel 的训练数据是平均参考（Pion-Tonachini et al., 2019, *NeuroImage* 198:181；mne-icalabel 文档）；MNE-BIDS-Pipeline 默认 `eeg_reference="average"`（`_config.py:337`） |
| 7 | `ica` | `ICA(n_components=0.999999, method="infomax"（extended）, labeler="iclabel", threshold=0.8)`，在 1–100 Hz 副本上拟合 | extended Infomax 与 1–100 Hz：ICLabel 的训练条件（同上）；`0.999999`：MNE-BIDS-Pipeline 默认（`_config.py:1519`，为避免秩亏数据出问题，几乎保留全部主成分）；`0.8`：MNE-BIDS-Pipeline 默认的 ICLabel 排除阈值（`ica_exclusion_thresholds`，`_config.py:1618`） |

### 1.2 MEG（`meg-erp` 与 `meg-rest` 相同；必须给出 `system=`）

前段按系统选择降噪方法：

| 系统 | 步骤 | 来源 |
|---|---|---|
| Neuromag / MEGIN | `bads = BadChannels("maxwell")` → `sss = Maxwell()`（SSS，`int_order=8`、`ext_order=3`，不做 tSSS，不做头动校正；cross-talk 和 fine-calibration 文件从 BIDS 读取，缺失则报错） | MNE `find_bad_channels_maxwell` 默认值；SSS 参数为 MNE-BIDS-Pipeline 默认（`mf_int_order`、`mf_ext_order`、`mf_st_duration=None`、`mf_mc=False`、`mf_cal_missing="raise"`、`mf_ctc_missing="raise"`，`_config.py:625–795`），也是 MaxFilter 的默认阶数（Taulu & Kajola, 2005） |
| CTF | `bads = BadChannels("maxwell")`（在 grade 0 副本上检测）→ `interpolate` → `reference = Reference(ctf_grade=3)` | 三阶合成梯度计是 CTF 的标准降噪（Vrba & Robinson, 2001, *Methods* 25:249） |
| KIT / Ricoh | `regression = Regression("ref_meg")`：参考磁强计的最小二乘回归 | MNE `EOGRegression`（`regress_artifact`）的做法。KIT 自带的 CALM（Adachi et al., 2001）是它的自适应版本，本库未实现。KIT 没有经过验证的自动坏道检测，请在 `info["bads"]` 中手动标记 |
| BTi、Artemis123、OPM 等 | 不提供：报错并提示用各步骤自行组合（OPM 用 `S.HFC`） | 没有可引用的默认流程 |

后段各系统相同：

| # | 步骤 | 参数 | 来源 |
|---|---|---|---|
| 1 | `highpass` | `Filter(0.1, None)` | 同 EEG（Tanner et al., 2015 的论证针对 ERP，同样适用于 ERF） |
| 2 | `resample` | `Resample(250)` | MEGnet 要求 250 Hz（Treacher et al., 2021, *NeuroImage* 241:118402；mne-icalabel 在输入不符时报警）。放在去工频之前：在 HAD-MEEG 的 CTF 数据（273 通道、1200 Hz、6 分钟）上，ZapLine-plus 在原采样率下需要 15 GB 以上内存 |
| 3 | `line_noise` | `LineNoise("zapline-plus")` | 同 EEG；50/60 Hz 及其低于 125 Hz 的谐波在 250 Hz 下保留 |
| 4 | `ica` | `ICA(20, picks="meg", method="infomax", labeler="megnet", threshold=0.8)`，在 1–100 Hz 副本上拟合 | 20 个成分、Infomax、1–100 Hz：MEGnet 的训练条件（同上，mne-icalabel 检查这些条件）；`0.8` 同 EEG。MEGnet 还要求记录至少 60 s |

MEG 与 EEG 同时采集时，MEG 预设只处理 MEG 通道（滤波和工频去除作用于全部通道）；需要分别处理时，用 `S.ByChannelType` 组合两个预设的步骤。

## 2. 分段阶段（`stage="epochs"`，作用于预处理后的每个 run）

| 预设 | 步骤 | 来源 |
|---|---|---|
| `*-erp` | `lowpass = Filter(None, 40)` | MNE-BIDS-Pipeline 默认 `h_freq=40`（`_config.py:878`） |
| `meg-*` | `align = HeadAlign(head_destination)`，只在给出 `head_destination` 时加入 | 跨 run 头位置对齐（设计文档 §4.5）；目标位置通常为 `meu.io.average_dev_head_t(runs)`。不给出时，`meu.epochs.combine` 仍会检查各 run 头位置相差是否超过 2 mm |
| `*-erp` | `epoch = Epoch(event_id, tmin=-0.2, tmax=0.5, baseline=(None, 0))`；`event_id` 必须给出 | MNE-BIDS-Pipeline 默认 `epochs_tmin`、`epochs_tmax`、`baseline`（`_config.py:1174–1207`） |
| `*-rest` | `epoch = FixedLengthEpochs(epoch_duration)`；`epoch_duration` 必须给出 | MNE-BIDS-Pipeline 同样要求用户设定 `rest_epochs_duration`（无默认值）。它决定后续频谱的频率分辨率（1 / 时长） |
| 全部 | `autoreject = AutoReject("local", n_interpolate=[4, 8, 16])` | Jas et al. (2017, *NeuroImage* 159:417)；`[4, 8, 16]` 为 MNE-BIDS-Pipeline 默认 `autoreject_n_interpolate`（`_config.py:1731`） |

MNE-BIDS-Pipeline 的行号对应版本 1.10.1 的 `mne_bids_pipeline/_config.py`。

## 3. 验证

- `tests/test_steps/test_recommended.py`：在模拟的 oddball 实验上跑完两个阶段（`eeg-erp`），靶刺激的 P300 峰值仍在 250–350 ms，靶与非靶的差异仍大于 3 µV（模拟值 7 µV，平均参考后会变小），autoreject 丢弃的 epoch 少于 20%。
- Neuromag 测试数据上跑 `meg-erp` 到 ICA 之前（测试数据只有 10 s，MEGnet 需要 60 s）：坏道被检测并由 SSS 重建，采样率降到 250 Hz，通道不变。
- 每个预设都能导出 YAML 再读回，参数不变。
