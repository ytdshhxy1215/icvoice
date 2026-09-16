# -*- coding: utf-8 -*-
"""小米 MiMo TTS 批量合成测试语音 → 输出 16kHz wav 供 FunASR 识别测试
用法:
  python funasr_env/make_test_audio.py            # 生成全部
  python funasr_env/make_test_audio.py 只生成部分
输出: tests/asr_testset/gen_wav/*.wav + labels.tsv
"""
import os, sys, base64, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI

API_KEY = os.environ.get("MIMO_API_KEY", "")  # 设置环境变量 MIMO_API_KEY

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "asr_testset", "gen_wav")
os.makedirs(OUT, exist_ok=True)

# 语速/风格指令（中文口吻），让合成贴近真人近讲
STYLE_ZH = "平稳自然的普通话女声，语速适中偏快，像实验员日常下指令的口吻，不要夸张起伏。"

SENTENCES = [
    # (文件名, 要合成的语句) —— 覆盖热词.txt 的 7 大类操作
    ("set_flow",       "流速设为一点零毫升每分钟"),
    ("set_flow2",      "把流速调到两点五毫升每分钟"),
    ("set_temp",       "柱温三十五度"),
    ("set_temp2",      "把柱温箱温度设到六十摄氏度"),
    ("set_wavelength", "检测波长设为二百五十四纳米"),
    ("set_amount",     "进样量设为十微升"),
    ("set_pressure",   "压力上限设为十五兆帕"),
    ("set_gradient",   "载入梯度洗脱程序，梯度从百分之五升到百分之九十五"),
    ("control1",       "开始进样"),
    ("control2",       "停止运行"),
    ("control3",       "平衡色谱柱"),
    ("control4",       "洗针"),
    ("control5",       "排气泡"),
    ("control6",       "开始数据采集"),
    ("control7",       "停止采集"),
    ("query1",         "现在压力多少"),
    ("query2",         "运行到哪一步了"),
    ("query3",         "柱温到了没有"),
    ("query4",         "当前流速是多少"),
    ("query5",         "系统运行状态怎么样"),
    ("data1",          "看看峰面积"),
    ("data2",          "保存这张谱图"),
    ("data3",          "导出报告"),
    ("data4",          "比较这两张色谱图"),
    ("method1",        "载入方法A"),
    ("method2",        "把这个设为默认方法"),
    ("help1",          "怎么设置梯度"),
    ("help2",          "流速最大能到多少"),
    ("help3",          "进样量默认是多少"),
    ("emerg1",         "急停"),
    ("emerg2",         "取消刚才的操作"),
    ("long1",          "开始进样，然后平衡色谱柱，等压力稳定后开始采集数据"),
]

def synth_one(client, name, text):
    completion = client.chat.completions.create(
        model="mimo-v2.5-tts",
        messages=[
            {"role": "user", "content": STYLE_ZH},
            {"role": "assistant", "content": text},
        ],
        audio={"format": "wav", "voice": "Chloe"},
    )
    wav = base64.b64decode(completion.choices[0].message.audio.data)
    p = os.path.join(OUT, f"{name}.wav")
    with open(p, "wb") as f:
        f.write(wav)
    return p

def main():
    client = OpenAI(api_key=API_KEY, base_url="https://api.xiaomimimo.com/v1")
    labels = []
    for name, text in SENTENCES:
        try:
            p = synth_one(client, name, text)
            print("OK", p, "(", text, ")")
            labels.append((name, text))
        except Exception as e:
            print("FAIL", name, e)
    # 写标注文件（原始文本；后续转写对比时注意 ASR 输出会是汉字数字，需归一化）
    with open(os.path.join(OUT, "labels.tsv"), "w", encoding="utf-8") as f:
        f.write("utt_id\ttext\n")
        for n, t in labels:
            f.write(f"{n}\t{t}\n")
    print(f"\n完成 {len(labels)}/{len(SENTENCES)}，输出目录: {OUT}")

if __name__ == "__main__":
    main()
