# -*- coding: utf-8 -*-
"""批量识别测试音频并与标注对比（数字汉字归一化后对比）
用法: python funasr_env/eval_gen.py
"""
import os, warnings
warnings.filterwarnings("ignore")
from funasr import AutoModel
from asr_hotword import HOTWORDS, load_model  # 复用热词与模型加载

HERE = os.path.dirname(os.path.abspath(__file__))
WAVDIR = os.path.join(HERE, "..", "tests", "asr_testset", "gen_wav")

# 数字汉字→阿拉伯数字 归一化表（识别和标注两侧都用）
CN_NUM = {"零":"0","一":"1","两":"2","二":"2","三":"3","四":"4","五":"5",
          "六":"6","七":"7","八":"8","九":"9"}
CN_UNIT = {"十":10, "百":100, "千":1000}

def cn2num(s):
    """把文本流里的中文数字串转阿拉伯数字（简单连读规则）"""
    out, i = [], 0
    while i < len(s):
        if s[i] in CN_NUM or s[i] in CN_UNIT:
            j = i; total = 0; cur = 0; ok = False
            while j < len(s) and (s[j] in CN_NUM or s[j] in CN_UNIT):
                ch = s[j]
                if ch in CN_NUM:
                    cur = int(CN_NUM[ch]); ok = True
                else:  # 单位
                    if cur == 0: cur = 1
                    total += cur * CN_UNIT[ch]; cur = 0; ok = True
                j += 1
            total += cur
            out.append(str(total) if ok else s[i:j]); i = j
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def norm(t):
    t = cn2num(t.strip())
    return t.replace(" ", "").replace("，", "").replace("。", "").replace("？", "").replace("！", "").replace("．",".").replace("小数点","点")

def main():
    m = load_model()
    labels = []
    with open(os.path.join(WAVDIR, "labels.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            u, t = line.rstrip("\n").split("\t")
            labels.append((u, t))

    correct, total = 0, 0
    rows = []
    for utt, ref in labels:
        p = os.path.join(WAVDIR, f"{utt}.wav")
        r = m.generate(input=p)
        hyp = r[0]["text"]
        nref, nhyp = norm(ref), norm(hyp)
        ok = (nref == nhyp)
        total += 1; correct += ok
        mark = "PASS" if ok else "FAIL"
        rows.append((mark, utt, ref, hyp))
        print(f"[{mark}] {utt}\n  ref: {ref}\n  hyp: {hyp}")

    print(f"\n===== 句级准确率(归一化后): {correct}/{total} = {100*correct/total:.1f}% =====")

if __name__ == "__main__":
    main()
