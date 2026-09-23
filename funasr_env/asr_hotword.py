# -*- coding: utf-8 -*-
"""热词版 ASR：SeACoParaformer + 领域热词
用法:
  python funasr_env/asr_hotword.py              # 麦克风实时说
  python funasr_env/asr_hotword.py xx.wav       # 识别指定音频
"""
import warnings, sys, os
warnings.filterwarnings("ignore")
from funasr import AutoModel

MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 热词表：config/hotwords_funasr.txt（空格分隔短语，109条，按FunASR最佳实践设计）
_HW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "hotwords_funasr.txt")
HOTWORDS = open(_HW_PATH, encoding="utf-8").read().strip()

def load_model(hotwords: str = HOTWORDS):
    return AutoModel(
        model=MODEL,
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        hotword=hotwords,
        device="cuda:0",
        disable_update=True,
        disable_pbar=True,
    )

def test_wav(path):
    m = load_model()
    r = m.generate(input=path)
    print("识别:", r[0]["text"])

def test_mic():
    import sounddevice as sd, numpy as np, scipy.io.wavfile as wf
    m = load_model()
    SR = 16000
    print("按回车开始说话(说完回车结束, 最长15s)...")
    input()
    buf = []

    def cb(indata, frames, t, status):
        buf.append(indata.copy())
        print(".", end="", flush=True)

    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", callback=cb):
        sd.sleep(15000) if False else input()  # 回车前一直录
    audio = np.concatenate(buf)[:, 0]
    print(f"\n录得 {len(audio)/SR:.1f}s")
    wf.write("_mic_test.wav", SR, audio)
    r = m.generate(input="_mic_test.wav")
    print("识别:", r[0]["text"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_wav(sys.argv[1])
    else:
        # 默认无参数=麦克风测试；--mic 才强制麦克风（原EOFe问题防止脚本误触发等待）
        print("无参数默认麦克风模式。热词条数:", len(HOTWORDS.split()))
