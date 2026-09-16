# icvoice —— 分析仪器智能语音交互系统

基于 **FunASR SeACoParaformer（热词）+ 流式 Paraformer 双路识别** 的色谱仪/离子色谱仪离线语音交互系统。

目标硬件：RDK X5（8×A55 + 10TOPS BPU，Ubuntu 22.04）。当前阶段在 PC（GPU）侧开发验证。

## 架构（组合双路方案）

```
麦克风 → VAD断句 → L1 流式online(600ms块, 边说边出字, 体验层, 不执行)
                → L2 离线SeACo+109热词(整句重识别, 唯一执行依据)
                        → 拼音纠错(pypinyin) → 意图规则引擎(36意图) → 动作JSON
                                                        → TTS播报 / 文字帮助 / 仪器执行(TCP)
```

## 当前指标（TTS 合成测试集，180 条，4 中文音色 × 随机参数）

| 层 | 指标 |
|---|---|
| ASR 句级（离线 SeACo+热词） | 98.3% |
| 语音→意图+槽位（端到端） | **100%**（拼音纠错兜住 ASR 剩余错误） |

## 目录

```
config/hotwords_funasr.txt   109 条短语热词（色谱仪领域，动宾结构）
docs/设计方案.md              总体设计（里程碑 M0-M9 / 工作包 W01-W24）
docs/项目执行日志.md          活文档：按日期追加的执行记录/踩坑/决策
docs/onnx_route_封存记录.md   ONNX+sherpa-onnx 路线探索存档（sherpa 不支持 SeACo 热词，封存）
funasr_env/                  实验脚本（ASR/TTS/评测/实时管线）
tests/asr_testset/           测试集（manifest/结果统计；wav 本地不入库）
```

## 快速开始

```powershell
pip install funasr modelscope pypinyin sounddevice scipy openai

# 1. 实时语音交互（回车开始说话，回车结束）
python funasr_env/realtime_pipeline.py

# 2. 生成 TTS 测试集（需环境变量 MIMO_API_KEY，小米 MiMo TTS）
python funasr_env/make_testset_v2.py

# 3. 评测
python funasr_env/eval_v2.py
```

## 关键结论（详见执行日志）

- MiMo TTS 的 **Chloe/Mia 是英文音色**，中文文本必须配中文音色（冰糖/茉莉/苏打/白桦），否则拉丁幻觉
- TTS 批量生成必须带时长 sanity check（出现过 1.1s 文本合成出 207s 音频的事故）
- 热词是准确率的必要条件：78→109 短语热词使 96.1%→98.3%；无热词的普通 Paraformer 仅约 64%
- 流式 online 版术语识别弱（64.4%）但延迟好，只作体验层；执行层 100% 依赖离线 SeACo
- NLU 拼音纠错对"进样→禁样/柱温→助瘟"类 ASR 错误的覆盖率 100%（64 条失败样本实测）
