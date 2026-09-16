# ONNX 导出路线探索记录（已封存）

> 时间：2026-09-15
> 状态：**封存**——sherpa-onnx 当前版本（1.13.8）不支持 SeACoParaformer 的热词路径，导出的 ONNX 无法带热词运行。
> 结论：热词是 98.3% 准确率的关键支柱，放弃热词不可接受。改走 FunASR 流式版路线。
> 本文按时间顺序记录探索过程、遇到的问题、尝试的解决方法，供将来重启该路线时参考。

---

## 0. 路线目标

原计划 W05：把 FunASR SeACoParaformer（热词版，PC 上 PyTorch 已验证 98.3%）导出为 ONNX，
用 sherpa-onnx（C++ 推理引擎）加载，一步解决「实时流式识别」和「X5 可部署」两个问题。

```
FunASR SeACoParaformer (PyTorch, 98.3%✓)
   │  funasr/bin/export.py
   ▼
model.onnx ──► sherpa-onnx OfflineRecognizer (流式C++推理, X5可部署)
```

## 1. 环境准备阶段

### 1.1 安装 sherpa-onnx
- `pip install sherpa-onnx` → **1.13.8 安装成功**（PC Windows 版，自带预编译二进制）

### 1.2 funasr 导出工具的依赖补齐（一系列连锁缺依赖）
funasr 的导出入口 `funasr/bin/export.py` 依赖缺失，逐个补装：

| 缺失依赖 | 处理 |
|---|---|
| `hydra` | `pip install hydra-core`（装到 1.3.7） |
| `onnxscript` / `onnx` / `onnxruntime` | `pip install onnxscript onnx onnxruntime`（onnx 1.22.0 / ort 1.24.4） |

### 1.3 导出命令的 hydra 语法坑
funasr 导出用 hydra 配置系统，**不支持** `--model=xxx` 传统参数风格：

```
✗ python export.py --model="iic/..." --export-dir=... 
  → error: unrecognized arguments

✗ python export.py model="iic/..." export-dir=...
  → Could not override 'model'. To append to your config use +model=...

✓ python export.py '+model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"' \
                   '+export-dir="./models/sherpa"' '+type="onnx"' '+device="cpu"'
```

## 2. torch 版本问题（第一次实质障碍）

**现象**：torch 2.11.0+cu128 下导出报错：
```
RuntimeError: Failed to convert 'dynamic_axes' to 'dynamic_shapes'.
Please provide 'dynamic_shapes' directly.
```
**原因**：torch 2.9+ 的 ONNX 导出 API 改版（`dynamic_axes` 旧参数被移除），funasr 导出代码用的是旧 API。

**解决**：降级 torch 到旧 API 兼容版本。查可装版本后选 **torch 2.6.0+cu126**（保留 CUDA、funasr 兼容）：
```
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126
```
降级后导出**跑通**，产出：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.onnx` | 957MB | SeACoParaformer 主模型（fp32） |
| `model_eb.onnx` | 34MB | 疑似热词编码器（Embedding/Bias 相关） |
| `tokens.json` | — | 词表（8404） |
| `am.mvn` / `config.yaml` | — | 归一化参数/模型配置 |

**注意**：`export-dir` 参数实际未生效，产物落在 modelscope 缓存目录
`C:\Users\zzhhy\.cache\modelscope\hub\models\iic\speech_seaco_paraformer.../`，需手工拷贝到项目 `models/sherpa/`。

## 3. sherpa-onnx 加载阶段（第二次障碍 → 最终卡点）

### 3.1 tokens 格式转换
sherpa 需要 `tokens.txt`（`token id` 逐行），funasr 给的是 `tokens.json`（list）。
写转换脚本：`models/sherpa/tokens.txt`（8404 行）✓

### 3.2 sherpa-onnx 1.13 API 变化
网上资料/旧版教程的构造方式全部失效：
```
✗ sherpa_onnx.OfflineRecognizer(**kwargs)  → TypeError: takes no arguments
✓ sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=..., tokens=..., num_threads=...)
  （工厂方法，内部走 OfflineModelConfig/OfflineParaformerModelConfig/OfflineRecognizerConfig）
```
另外 1.13 的 `hotwords_file/hotwords_score` 在 `OfflineRecognizerConfig` 上存在，但仅对 transducer 类模型有效。

### 3.3 metadata 缺失（逐个补齐的完整过程）
`from_paraformer` 加载报错 `'vocab_size' does not exist in the metadata`——
funasr 导出的 ONNX **不写 sherpa 需要的 metadata**。按报错顺序逐个补：

| 报错要求 | 取值来源 | 键名 |
|---|---|---|
| vocab_size | 8404（词表条数） | `vocab_size` |
| lfr_window_size | config.yaml `frontend_conf.lfr_m` = 7 | `lfr_window_size` |
| lfr_window_shift | funasr 标准 lfr_block = 4 | `lfr_window_shift` |
| neg_mean | am.mvn（Kaldi Nnet 格式，`<AddShift>` 段，560 维） | `neg_mean` |
| inv_stddev | am.mvn `<Rescale>` 段；**注意 sherpa 键名是 `inv_stddev` 不是 `inv_std`** | `inv_stddev` |

补齐后 `from_paraformer` **构造成功**。

### 3.4 最终卡点：bias_embed 输入无人喂
解码时报错：
```
Non-zero status code returned while running MatMul node. Name:'/src_attn/linear_k_v/MatMul'
Missing Input: bias_embed
```
**根因分析**：
- 检查导出的 model.onnx 计算图输入：`['speech', 'speech_lengths', 'bias_embed']`
- `bias_embed` 是 SeACoParaformer 热词机制的第三个输入（热词经 bias 编码器生成的嵌入）
- sherpa-onnx 的 paraformer 推理路径只喂 2 个输入（speech/speech_lengths），**没有实现 SeACo 热词路径**
- 证据链：
  1. `grep -ril seaco sherpa_onnx包目录` → 无结果；`bias_embed` → 无结果
  2. sherpa-onnx 官方 [asr-models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) 全部 paraformer 包中**没有任何 SeACo 包**
  3. 下载官方 `sherpa-onnx-paraformer-zh-2023-09-14`（最接近的历史包）验证：其 onnx 输入只有 `['speech','speech_lengths']`，comment 注明基座是普通 `speech_paraformer-large-vad-punc`（非 SeACo）
- 结论：**sherpa-onnx 上游已放弃/从未完整支持 SeACo 热词**，自己编译 sherpa-onnx 补 seaco 路径工作量数月级，不值得

### 3.5 无热词基线验证（证实热词不可弃）
用官方 int8 普通 Paraformer 冒烟 6 条关键样本（此前靠热词救回的句子）：

| 音频 | funasr SeACo+热词 | sherpa 普通 int8（无热词） |
|---|---|---|
| 冰糖_set_flow_a_0 | 流速设为三点零毫升每分钟 ✓ | 流速设为**三**毫升每分钟（丢"点零"）|
| 茉莉_emg_stop 急停 | 急停 ✓ | **吉田** ✗ |
| 白桦_ctl_stop_acq 停止采集 | 停止采集 ✓ | **平直采集** ✗ |
| 苏打_emg_stop 急停 | 急停 ✓ | 急停 ✓ |
| 茉莉_qry_baseline 基线稳定了吗 | 基线稳定了吗 ✓ | **西线**稳定了吗 ✗ |
| 白桦_hlp_bubble 怎么排气泡 | 怎么排气泡 ✓ | 什么排气**炮** ✗ |

无热词错误率显著上升，**证实热词是 98.3% 的必要条件，不可妥协** → 路线封存。

## 4. 当前环境状态（封存时快照）

```
funasr 1.4.14 / torch 2.6.0+cu126 / sherpa-onnx 1.13.8
models/sherpa/  ← 导出的onnx+补齐metadata+tokens.txt（保留，勿删）
funasr_env/tmp/sherpa-onnx-paraformer-zh-2023-09-14/  ← 官方普通paraformer int8包（保留作对照）
```
**torch 已从 2.11 降到 2.6.0**（导出需要；funasr 推理/评测在 2.6 下工作正常）。

## 5. 决策：改走 FunASR 流式版

| 备选 | 状态 |
|---|---|
| ~~A. ONNX→sherpa-onnx~~ | **封存**（本文件），sherpa 无 SeACo 热词路径 |
| B. FunASR 流式版（Paraformer-streaming / 2pass） | **转正**：PyTorch 环境内直接流式；PC 侧验证实时性；X5 部署问题推迟到 M7 后再评估（届时可能换板卡或用微调后的新模型重新导出） |
| C. sherpa-onnx 老版本（1.10.46 等）可能存在的 seaco 支持 | 未验证（用户中止）；将来重启 onnx 路线时可作为第一优先验证项 |

## 6. 将来重启 ONNX 路线的检查清单

1. 先装 `sherpa-onnx==1.10.46`（2024 年中版本）验证其 C++ 侧是否有 seaco/bias_embed 路径
2. 若无：考虑用 onnxruntime 自己写 bias_embed 喂入的推理包装（Python 层可做，绕开 sherpa；流式切窗逻辑自己实现）
3. 或关注 sherpa-onnx 上游是否恢复 seaco 支持（watch repo）
4. metadata 补齐脚本可复用（本文 §3.3 的 5 个键）
