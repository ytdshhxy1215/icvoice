# -*- coding: utf-8 -*-
"""测试集 v2 生成器：多音色 × 意图模板 × 随机参数值
- 4 个中文音色（冰糖/茉莉/苏打/白桦），上一轮 Chloe 是英文音色（拉丁幻觉根源）
- 数值参数随机化并固定随机种子（可复现）
- 输出: tests/asr_testset/gen_wav_v2/{voice}_{utt}.wav + manifest_v2.tsv
- 支持断点续跑（已存在的 wav 跳过）
"""
import os, sys, base64, random, time, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI

API_KEY = os.environ.get("MIMO_API_KEY", "")  # 设置环境变量 MIMO_API_KEY
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "asr_testset", "gen_wav_v2")
os.makedirs(OUT, exist_ok=True)

VOICES = ["冰糖", "茉莉", "苏打", "白桦"]   # 全中文音色
STYLE_ZH = "平稳自然的中文口播，语速适中，像实验员日常下操作指令，吐字清晰，不要夸张起伏。"

# ---------- 数字转中文 ----------
DIGITS = "零一二三四五六七八九"
def int2cn(n):
    if n < 10: return DIGITS[n]
    if n == 10: return "十"
    if n < 20: return "十" + DIGITS[n % 10]
    if n < 100:
        s = DIGITS[n // 10] + "十"
        return s + DIGITS[n % 10] if n % 10 else s
    if n < 1000:
        s = DIGITS[n // 100] + "百"
        r = n % 100
        if r == 0: return s
        if r < 10: return s + "零" + DIGITS[r]
        return s + int2cn(r)
    return str(n)

def float2cn(x):
    s = f"{x:g}"
    if "." in s:
        a, b = s.split(".")
        return int2cn(int(a)) + "点" + "".join(DIGITS[int(c)] for c in b)
    return int2cn(int(s))

# ---------- 固定文本模板（无参数，覆盖热词.txt 七大类）----------
FIXED = [
    # CONTROL 流程控制
    ("ctl_start_inject",   "开始进样"),
    ("ctl_stop_run",       "停止运行"),
    ("ctl_eq_column",      "平衡色谱柱"),
    ("ctl_wash_needle",    "洗针"),
    ("ctl_purge_bubble",   "排气泡"),
    ("ctl_start_acq",      "开始数据采集"),
    ("ctl_stop_acq",       "停止采集"),
    ("ctl_pause",          "暂停运行"),
    # QUERY 状态查询
    ("qry_pressure",       "现在压力多少"),
    ("qry_step",           "运行到哪一步了"),
    ("qry_temp_reach",     "柱温到了没有"),
    ("qry_flow_now",       "当前流速是多少"),
    ("qry_status",         "系统运行状态怎么样"),
    ("qry_baseline",       "基线稳定了吗"),
    # DATA 数据与结果
    ("dat_peak_area",      "看看峰面积"),
    ("dat_save_spec",      "保存这张谱图"),
    ("dat_export",         "导出报告"),
    ("dat_compare",        "比较这两张色谱图"),
    ("dat_retention",      "看一下保留时间"),
    # METHOD 方法管理
    ("mtd_load_a",         "载入方法A"),
    ("mtd_set_default",    "把这个设为默认方法"),
    ("mtd_save",           "保存当前方法"),
    ("mtd_load_program",   "载入梯度程序"),
    # EMERGENCY 安全紧急
    ("emg_stop",           "急停"),
    ("emg_cancel",         "取消刚才的操作"),
    # HELP 帮助咨询
    ("hlp_gradient",       "怎么设置梯度"),
    ("hlp_flow_max",       "流速最大能到多少"),
    ("hlp_amount_default", "进样量默认是多少"),
    ("hlp_bubble",         "怎么排气泡"),
]

# ---------- 数值模板：(utt_id, 模板, 取值池) ----------
# 值池含易混淆数字（1/4/7/10/14/40）与常用值
VALUE_POOLS = {
    "flow":       [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "temp":       [25, 35, 40, 45, 60, 80],
    "wavelength": [210, 254, 280],
    "amount":     [5, 10, 20, 50, 100],
    "pressure":   [10, 15, 20, 25, 30],
    "current":    [50, 100, 200, 300],
}
VALUE_TMPL = [
    ("set_flow_a",    "流速设为{v}毫升每分钟",           "flow"),
    ("set_flow_b",    "把流速调到{v}毫升每分钟",         "flow"),
    ("set_temp_a",    "柱温{v}度",                       "temp"),
    ("set_temp_b",    "把柱温箱温度设到{v}摄氏度",       "temp"),
    ("set_wave_a",    "检测波长设为{v}纳米",             "wavelength"),
    ("set_amount_a",  "进样量设为{v}微升",               "amount"),
    ("set_press_a",   "压力上限设为{v}兆帕",             "pressure"),
    ("set_curr_a",    "抑制器电流设为{v}毫安",           "current"),
]
N_VALUE_INSTANCES = 2   # 每模板每音色 2 个随机值

def build_jobs():
    """生成 (utt_id, voice, category, text) 任务列表，固定种子可复现"""
    rng = random.Random(42)
    jobs = []
    for voice in VOICES:
        for uid, text in FIXED:
            jobs.append((uid, voice, "fixed", text))
        for uid, tmpl, pool in VALUE_TMPL:
            for k in range(N_VALUE_INSTANCES):
                v = rng.choice(VALUE_POOLS[pool])
                text = tmpl.format(v=float2cn(v))
                jobs.append((f"{uid}_{k}", voice, "value", text))
    return jobs

def synth(client, text, voice):
    completion = client.chat.completions.create(
        model="mimo-v2.5-tts",
        messages=[
            {"role": "user", "content": STYLE_ZH},
            {"role": "assistant", "content": text},
        ],
        audio={"format": "wav", "voice": voice},
    )
    return base64.b64decode(completion.choices[0].message.audio.data)

def main():
    jobs = build_jobs()
    total = len(jobs)
    print(f"任务总数: {total}（{len(VOICES)} 音色 × {total//len(VOICES)} 条）")
    client = OpenAI(api_key=API_KEY, base_url="https://api.xiaomimimo.com/v1")

    manifest, done, fail = [], 0, 0
    for i, (uid, voice, cat, text) in enumerate(jobs):
        fname = f"{voice}_{uid}.wav"
        fpath = os.path.join(OUT, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
            done += 1
            manifest.append((fname, voice, cat, text))
            continue
        for attempt in range(3):
            try:
                wav = synth(client, text, voice)
                with open(fpath, "wb") as f:
                    f.write(wav)
                done += 1
                manifest.append((fname, voice, cat, text))
                print(f"[{done}/{total}] {fname}  {text}")
                time.sleep(0.2)
                break
            except Exception as e:
                print(f"RETRY{attempt+1} {fname}: {e}")
                time.sleep(2)
        else:
            fail += 1
            print(f"FAIL {fname}")

    with open(os.path.join(OUT, "..", "manifest_v2.tsv"), "w", encoding="utf-8") as f:
        f.write("file\tvoice\tcategory\ttext\n")
        for row in manifest:
            f.write("\t".join(row) + "\n")
    print(f"\n完成 {done}/{total}，失败 {fail}。manifest: tests/asr_testset/manifest_v2.tsv")

if __name__ == "__main__":
    main()
