# -*- coding: utf-8 -*-
"""FunASR 流式 Paraformer 部署与冒烟测试
模型: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online (2pass 流式)
验证:
  1. 模型可加载
  2. 分块喂音频模拟流式（chunk=600ms）, 边喂边出中间结果
  3. 对 3 条测试音频拿最终结果
"""
import warnings, os, sys, time, wave
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from funasr import AutoModel

MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

def read_wav(path):
    w = wave.open(path)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return w.getframerate(), d.astype(np.float32) / 32768.0

def main():
    print("加载流式模型（首次下载约1GB）...", flush=True)
    t0 = time.time()
    m = AutoModel(
        model=MODEL,
        disable_update=True,
        disable_pbar=True,
    )
    print(f"加载完成 {time.time()-t0:.1f}s")

    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tests = [
        "tests/asr_testset/gen_wav_v2/冰糖_set_flow_a_0.wav",
        "tests/asr_testset/gen_wav_v2/茉莉_emg_stop.wav",
        "tests/asr_testset/gen_wav_v2/白桦_ctl_stop_acq.wav",
    ]
    chunk_size_ms = 600   # 流式块大小（2pass: 600ms 看一眼, 句尾全量重解）

    for path in tests:
        sr, x = read_wav(os.path.join(ROOT, path))
        chunk = int(sr * chunk_size_ms / 1000)
        print(f"\n== {os.path.basename(path)} ({len(x)/sr:.1f}s)")
        # 模拟流式：分块喂入, 每 chunk 打印当前中间结果
        n = (len(x) + chunk - 1) // chunk
        t_start = time.time()
        for i in range(n):
            seg = x[i*chunk:(i+1)*chunk]
            is_final = (i == n - 1)
            res = m.generate(input=seg, cache={}, is_final=is_final, chunk_size=chunk_size_ms)
            text = res[0]["value"][0] if res and res[0].get("value") else ""
            t_elapsed = time.time() - t_start
            print(f"  [{t_elapsed:5.2f}s] {'FINAL' if is_final else 'partial'}: {text}")
        # 音频播完时刻
        print(f"  (音频实际时长 {len(x)/sr:.1f}s)")

if __name__ == "__main__":
    main()
