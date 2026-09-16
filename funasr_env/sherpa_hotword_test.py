# -*- coding: utf-8 -*-
"""sherpa-onnx + SeACoParaformer(onnx) 热词识别验证
前置: funasr_env/export_onnx.sh 已导出模型到 models/sherpa/<model_dir>/
跑法: python funasr_env/sherpa_hotword_test.py [wav路径]
     无参数时用测试集第1条做冒烟
"""
import os, sys, wave, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODEL_DIR_CANDIDATES = sorted(
    (os.path.join(ROOT, "models", "sherpa", d) for d in os.listdir(os.path.join(ROOT, "models", "sherpa"))),
    key=os.path.getmtime,
) if os.path.exists(os.path.join(ROOT, "models", "sherpa")) else []

def find_model_dir():
    # 模型文件直接放在 models/sherpa/ 下（export-dir 未生效，从 modelscope 缓存收集而来）
    d = os.path.join(ROOT, "models", "sherpa")
    if os.path.exists(os.path.join(d, "model.onnx")):
        return d
    raise FileNotFoundError(f"{d} 下没有 model.onnx")

def load():
    import sherpa_onnx
    d = find_model_dir()
    files = os.listdir(d)
    print("模型目录:", d)
    print("文件:", files)
    paraformer = os.path.join(d, "model.onnx")
    tokens = os.path.join(d, "tokens.txt")
    hotword_file = os.path.join(d, "hotwords.txt")

    model_config = sherpa_onnx.OfflineModelConfig(
        paraformer=sherpa_onnx.OfflineParaformerModelConfig(
            model=paraformer,
        ),
        tokens=tokens,
        num_threads=4,
        debug=False,
    )
    cfg = sherpa_onnx.OfflineRecognizerConfig(
        model_config=model_config,
        decoding_method="greedy_search",
    )
    if os.path.exists(hotword_file):
        cfg.hotwords_file = hotword_file
        cfg.hotwords_score = 1.5
        print("使用热词文件:", hotword_file)
    asr = sherpa_onnx.OfflineRecognizer(cfg)
    return asr

def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"需16kHz，实际{w.getframerate()}"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0

if __name__ == "__main__":
    asr = load()
    wav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "asr_testset", "gen_wav_v2", "冰糖_set_flow_a_0.wav")
    s = asr.create_stream()
    s.accept_waveform(16000, read_wav(wav))
    asr.decode_stream(s)
    print("识别:", s.result.text)
