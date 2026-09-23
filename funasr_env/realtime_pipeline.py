# -*- coding: utf-8 -*-
"""实时语音交互链路 v1（组合方案）
麦克风 → 能量VAD断句 → 双路识别：
  L1 流式online(600ms块) 边说边出字（体验层，不执行）
  L2 说完后整句 SeACo+热词（执行层依据）→ 拼音纠错 → 意图识别(规则) → 动作JSON
用法: python funasr_env/realtime_pipeline.py
操作: 按回车开始说话 → 说完按回车 → 显示双路结果与意图
(纯PC验证版; 唤醒词/TTS播报/仪器执行在后续包接入)
"""
import warnings, os, sys, time, json, re, threading, queue
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd
import scipy.signal as ss
from funasr import AutoModel
from pypinyin import lazy_pinyin

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SR = 16000

# ---------------- 模型加载 ----------------
OFFLINE_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ONLINE_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

def load_models():
    hw = open(os.path.join(ROOT, "config", "hotwords_funasr.txt"), encoding="utf-8").read().strip()
    t0 = time.time()
    offline = AutoModel(model=OFFLINE_MODEL, vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        hotword=hw, disable_update=True, disable_pbar=True)
    print(f"[load] 离线SeACo+热词 {time.time()-t0:.1f}s")
    t0 = time.time()
    online = AutoModel(model=ONLINE_MODEL, disable_update=True, disable_pbar=True)
    print(f"[load] 流式online {time.time()-t0:.1f}s")
    return offline, online

# ---------------- 拼音纠错 + 意图规则 ----------------
def py(s):
    return " ".join(lazy_pinyin(s.strip()))

def norm(t):
    t = re.sub(r"\s+", "", t).lower()
    # 汉字数字→阿拉伯
    N = {c: str(i) for i, c in enumerate("零一二三四五六七八九")}; N["两"] = "2"
    U = {"十": 10, "百": 100, "千": 1000}
    out, i = [], 0
    while i < len(t):
        if t[i] in N or t[i] in U:
            j, tot, cur = i, 0, 0
            while j < len(t) and (t[j] in N or t[j] in U):
                if t[j] in N: cur = int(N[t[j]])
                else:
                    if cur == 0: cur = 1
                    tot += cur * U[t[j]]; cur = 0
                j += 1
            out.append(str(tot + cur)); i = j
        else:
            out.append(t[i]); i += 1
    t = "".join(out)
    t = t.replace("点", ".")   # 一点零→1.0
    for p in "，。？！、；：": t = t.replace(p, "")
    return t

# 意图表：意图 -> (模板关键词列表, 槽位抽取正则)
INTENTS = [
    # 具体问句/控制词放在泛化关键词之前，避免“暂停运行”被“停运行”、
    # “进样量默认是多少”被“进样”抢先命中。
    ("pause_run",      ["暂停"]),
    ("help_gradient",  ["怎么设置梯度", "如何设置梯度"]),
    ("help_flow_max",  ["流速最大", "最大流速"]),
    ("help_amount",    ["进样量默认", "默认进样量"]),
    ("help_bubble",    ["怎么排气泡", "如何排气泡", "什么排气泡", "怎么排空气"]),
    ("start_inject",   ["开始进样", "进样"]),
    ("stop_run",       ["停止运行", "停运行"]),
    ("start_acq",      ["开始采集", "开始数据采集", "开始测量"]),
    ("stop_acq",       ["停止采集", "停采集", "停止测量"]),
    ("eq_column",      ["平衡色谱柱", "平衡柱", "平衡"]),
    ("wash_needle",    ["洗针"]),
    ("purge_bubble",   ["排气泡", "排气管"]),
    ("query_pressure", ["压力多少", "压力是多少", "查看压力", "现在压力"]),
    ("query_step",     ["运行到哪", "到哪一步", "哪一步了"]),
    ("query_temp",     ["柱温多少", "温度到了没有", "柱温到了", "现在柱温", "查看柱温"]),
    ("query_flow",     ["流速是多少", "流速多少", "当前流速", "查看流速"]),
    ("query_status",   ["运行状态", "什么状态", "状态怎么样", "系统状态"]),
    ("query_baseline", ["基线稳定", "基线"]),
    ("peak_area",      ["峰面积"]),
    ("save_spectra",   ["保存这张谱图", "保存谱图", "保存图谱"]),
    ("export_report",  ["导出报告", "生成报告"]),
    ("compare_chrom",  ["比较这两张色谱图", "比较色谱图", "比较两张"]),
    ("retention_time", ["保留时间"]),
    ("load_method",    ["载入方法", "调用方法"]),
    ("set_default",    ["设为默认方法", "设为默认"]),
    ("save_method",    ["保存当前方法", "保存方法"]),
    ("load_program",   ["载入梯度程序", "载入程序", "载入梯度"]),
    ("emergency_stop", ["急停", "紧急停止"]),
    ("cancel_op",      ["取消刚才", "取消操作", "取消"]),
]
# 参数设置类带槽位（顺序敏感：长模板在前，set_temp 排除"柱温箱"避免抢配）
PARAM_INTENTS = [
    ("set_temp2",      r"(?:把柱温箱温度设到|柱温箱温度设到|柱温箱温度调到)(?P<value>[0-9.]+)(?:摄氏度|度)", "temp"),
    ("set_flow",       r"(?:流速|流量).{0,6}?(?:设为|设置为|调到|设到|调至|设个|要到)?(?P<value>[0-9.]+)毫升每分钟", "flow"),
    ("set_temp",       r"柱温(?!箱).{0,6}?(?:设为|设置为|调到|设到|调至)?(?P<value>[0-9.]+)(?:度|摄氏度)", "temp"),
    ("set_wavelength", r"(?:检测波长|波长).{0,6}?(?:设为|设置为|调到|设到|调至)?(?P<value>[0-9.]+)纳米", "wavelength"),
    ("set_amount",     r"进样量.{0,6}?(?:设为|设置为|调到|设到|调至)?(?P<value>[0-9.]+)微升", "amount_ul"),
    ("set_pressure",   r"压力上限.{0,6}?(?:设为|设置为|调到|设到|调至)?(?P<value>[0-9.]+)兆帕", "pressure_mpa"),
    ("set_current",    r"抑制器电流.{0,6}?(?:设为|设置为|调到|设到|调至)?(?P<value>[0-9.]+)毫安", "current_ma"),
]

def parse_intent(text):
    """规则+拼音纠错 意图识别, 返回动作JSON"""
    t = norm(text)
    # 1) 参数类(正则, 值已归一化为阿拉伯)
    for name, pat, slot in PARAM_INTENTS:
        m = re.search(pat, t)
        if m:
            val = m.group("value")
            return {"intent": name, "slots": {slot: float(val) if val else None},
                    "text": text, "source": "rule"}
    # 2) 关键词意图
    for name, kws in INTENTS:
        for kw in kws:
            if kw in t:
                return {"intent": name, "slots": {}, "text": text, "source": "rule"}
    # 3) 拼音纠错: 意图关键词逐个拼音比对
    t_py = py(t)
    best, best_ratio = None, 0.0
    import difflib
    for name, kws in INTENTS:
        for kw in kws:
            r = difflib.SequenceMatcher(None, py(kw), t_py).ratio()
            # 或在句中找相似片段
            for i in range(max(0, len(t)-len(kw)-2)):
                frag = t[i:i+len(kw)+2]
                r2 = difflib.SequenceMatcher(None, py(kw), py(frag)).ratio()
                r = max(r, r2)
            if r > best_ratio:
                best, best_ratio = name, r
    if best and best_ratio >= 0.75:
        return {"intent": best, "slots": {}, "text": text,
                "source": "pinyin_fix", "ratio": round(best_ratio, 3)}
    return {"intent": "unknown", "slots": {}, "text": text, "source": "none"}

# ---------------- 录音 ----------------
def record_until_enter():
    """回车开始, 回车结束。返回16k float32"""
    q = queue.Queue()
    def cb(indata, frames, t, status):
        q.put(indata.copy())
    input(">> 按回车开始说话...")
    chunks = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=1600, callback=cb):
        input(">> 说话中... 说完按回车结束\n")
        while not q.empty():
            chunks.append(q.get())
    while not q.empty():  # 回车后再收一次残余
        try: chunks.append(q.get(timeout=0.1))
        except queue.Empty: break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:, 0].astype(np.float32) / 32768.0

# ---------------- 主流程 ----------------
def main():
    offline, online = load_models()
    cs = [0, 10, 5]
    stride = cs[1] * 960  # 600ms @16k
    session = 0
    while True:
        audio = record_until_enter()
        if len(audio) < SR * 0.3:
            print("(太短, 重来)\n"); continue
        session += 1
        # ---- L1 流式(体验层) ----
        t1 = time.time()
        cache, streaming_parts = {}, []
        n = (len(audio) + stride - 1) // stride
        for i in range(n):
            res = online.generate(input=audio[i*stride:(i+1)*stride], cache=cache,
                                  is_final=(i == n-1), chunk_size=cs,
                                  encoder_chunk_look_back=4, decoder_chunk_look_back=1)
            if res and res[0].get("text"):
                streaming_parts.append(res[0]["text"])
        streaming_text = "".join(streaming_parts)
        t_stream = time.time() - t1
        # ---- L2 离线SeACo+热词(执行层) ----
        t2 = time.time()
        r = offline.generate(input=audio)
        final_text = r[0]["text"] if r and r[0].get("text") else ""
        t_final = time.time() - t2
        # ---- NLU ----
        act = parse_intent(final_text)
        # ---- 展示 ----
        print(f"\n===== 会话 {session} ({len(audio)/SR:.1f}s) =====")
        print(f"[流式online(体验层)] {streaming_text or '(无)'}   {t_stream:.2f}s")
        print(f"[离线SeACo+热词(执行层)] {final_text}   {t_final:.2f}s")
        print(f"[意图] {json.dumps(act, ensure_ascii=False)}")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出")
