# -*- coding: utf-8 -*-
"""测试集 v2 评测：识别 gen_wav_v2 全部音频，按音色/类别/意图统计
- 全部用热词（config/hotwords_funasr.txt）
- 归一化口径：去空格、去标点、数字汉字→阿拉伯、英文小写
- 输出: tests/asr_testset/results/eval_v2.tsv + summary_v2.txt
- 断点续跑：识别结果已有则跳过（增量写 eval_v2.tsv）
"""
import os, re, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
WAVDIR = os.path.join(HERE, "..", "tests", "asr_testset", "gen_wav_v2")
RESULT = os.path.join(HERE, "..", "tests", "asr_testset", "results")
os.makedirs(RESULT, exist_ok=True)
EVAL_TSV = os.path.join(RESULT, "eval_v2.tsv")

DIGITS = "零一二三四五六七八九"
def cn2num(s):
    N = {c: str(i) for i, c in enumerate(DIGITS)}
    N["两"] = "2"
    U = {"十": 10, "百": 100, "千": 1000}
    out, i = [], 0
    while i < len(s):
        if s[i] in N or s[i] in U:
            j, tot, cur = i, 0, 0
            while j < len(s) and (s[j] in N or s[j] in U):
                if s[j] in N: cur = int(N[s[j]])
                else:
                    if cur == 0: cur = 1
                    tot += cur * U[s[j]]; cur = 0
                j += 1
            v = tot + cur
            out.append(str(v)); i = j
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def norm(t):
    t = re.sub(r"\s+", "", t).lower()
    t = cn2num(t)
    for p in "，。？！、；：":
        t = t.replace(p, "")
    return t

def load_done():
    done = {}
    if os.path.exists(EVAL_TSV):
        with open(EVAL_TSV, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    done[parts[0]] = parts
    return done

def main():
    import sys
    sys.path.insert(0, HERE)
    from asr_hotword import load_model
    m = load_model()

    manifest = []
    with open(os.path.join(WAVDIR, "..", "manifest_v2.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            fn, voice, cat, text = line.rstrip("\n").split("\t")
            manifest.append((fn, voice, cat, text))

    done = load_done()
    rows = dict(done)
    todo = [x for x in manifest if x[0] not in done]
    print(f"总 {len(manifest)} 条，已识别 {len(manifest)-len(todo)}，待识别 {len(todo)}")

    for k, (fn, voice, cat, ref) in enumerate(todo):
        r = m.generate(input=os.path.join(WAVDIR, fn))
        hyp = r[0]["text"]
        rows[fn] = (fn, voice, cat, ref, hyp, "PASS" if norm(ref) == norm(hyp) else "FAIL")
        if (k + 1) % 20 == 0:
            print(f"... {k+1}/{len(todo)}")
            with open(EVAL_TSV, "w", encoding="utf-8") as f:
                f.write("file\tvoice\tcategory\tref\thyp\tresult\n")
                for v in rows.values():
                    f.write("\t".join(v) + "\n")

    with open(EVAL_TSV, "w", encoding="utf-8") as f:
        f.write("file\tvoice\tcategory\tref\thyp\tresult\n")
        for v in rows.values():
            f.write("\t".join(v) + "\n")

    # ---- 汇总 ----
    lines = []
    def stat(name, sel):
        sub = [v for v in rows.values() if sel(v)]
        if not sub: return
        ok = sum(1 for v in sub if v[5] == "PASS")
        lines.append(f"{name:24s} {ok:3d}/{len(sub):3d} = {100*ok/len(sub):5.1f}%")

    stat("总体", lambda v: True)
    lines.append("")
    for voice in sorted({v[1] for v in rows.values()}):
        stat(f"音色[{voice}]", lambda v, vc=voice: v[1] == vc)
    lines.append("")
    for cat in sorted({v[2] for v in rows.values()}):
        stat(f"类别[{cat}]", lambda v, c=cat: v[2] == c)
    lines.append("")
    fails = [v for v in rows.values() if v[5] == "FAIL"]
    lines.append(f"失败明细（{len(fails)} 条）:")
    for fn, voice, cat, ref, hyp, _ in fails:
        lines.append(f"  [{voice}] {fn}\n    ref: {ref}\n    hyp: {hyp}")

    summary = "\n".join(lines)
    with open(os.path.join(RESULT, "summary_v2.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary.encode("gbk", "replace").decode("gbk") if os.name == "nt" else summary)
    print("\nwritten:", EVAL_TSV)

if __name__ == "__main__":
    main()
