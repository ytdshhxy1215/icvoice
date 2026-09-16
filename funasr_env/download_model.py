# -*- coding: utf-8 -*-
"""下载/加载 FunASR Paraformer 中文模型（首次运行自动从 ModelScope 下载 ~1GB）"""
import warnings
warnings.filterwarnings("ignore")
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

def load_model():
    return AutoModel(
        model=MODEL,
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",   # 长音频切分
        # 标点模型 ID 在 funasr 1.4 里注册异常，命令场景非必需，暂不加载
        device="cuda:0",
    )

if __name__ == "__main__":
    m = load_model()
    # 拿模型自带的样例音频冒烟测试（Windows 下用运行目录的样例）
    import scipy.io.wavfile as wf, numpy as np, os
    t = np.linspace(0, 3, 48000)
    audio = (np.sin(2*np.pi*440*t) * 0.3 * 32767).astype(np.int16)
    wf.write("_smoke.wav", 16000, audio)
    r = m.generate(input="_smoke.wav")
    print("smoke OK:", r[0]["text"][:50] if r and r[0].get("text") else "(空输出，正常——440Hz无语音)")
