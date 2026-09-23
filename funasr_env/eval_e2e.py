# -*- coding: utf-8 -*-
"""端到端评测（纯离线单路）：eval_v2.tsv 的 ASR 结果 → 意图/槽位比对
- 期望意图由 utt_id 前缀映射（manifest_v2.tsv）
- 数值模板同时校验槽位值（与 ref 归一化后的数字一致）
- 依赖：先跑 eval_v2.py 生成 tests/asr_testset/results/eval_v2.tsv
- 输出: results/e2e_intent.json + summary_e2e.txt
"""
import os, re, json, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "..", "tests", "asr_testset", "results")

# utt_id 前缀 -> 期望意图
EXPECT = {
    "ctl_start_inject": "start_inject", "ctl_stop_run": "stop_run",
    "ctl_eq_column": "eq_column", "ctl_wash_needle": "wash_needle",
    "ctl_purge_bubble": "purge_bubble", "ctl_start_acq": "start_acq",
    "ctl_stop_acq": "stop_acq", "ctl_pause": "pause_run",
    "qry_pressure": "query_pressure", "qry_step": "query_step",
    "qry_temp_reach": "query_temp", "qry_flow_now": "query_flow",
    "qry_status": "query_status", "qry_baseline": "query_baseline",
    "dat_peak_area": "peak_area", "dat_save_spec": "save_spectra",
    "dat_export": "export_report", "dat_compare": "compare_chrom",
    "dat_retention": "retention_time", "mtd_load_a": "load_method",
    "mtd_set_default": "set_default", "mtd_save": "save_method",
    "mtd_load_program": "load_program", "emg_stop": "emergency_stop",
    "emg_cancel": "cancel_op", "hlp_gradient": "help_gradient",
    "hlp_flow_max": "help_flow_max", "hlp_amount_default": "help_amount",
    "hlp_bubble": "help_bubble",
    "set_flow_a": "set_flow", "set_flow_b": "set_flow",
    "set_temp_a": "set_temp", "set_temp_b": "set_temp2",
    "set_wave_a": "set_wavelength", "set_amount_a": "set_amount",
    "set_press_a": "set_pressure", "set_curr_a": "set_current",
}

def main():
    import sys
    sys.path.insert(0, HERE)
    from realtime_pipeline import parse_intent, norm

    man = {}
    with open(os.path.join(HERE, "..", "tests", "asr_testset", "manifest_v2.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            fn, voice, cat, text = line.rstrip("\n").split("\t")
            man[fn] = (voice, cat, text)

    out, n_ok = [], 0
    with open(os.path.join(RESULT, "eval_v2.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            fn, voice, cat, ref, hyp = parts[:5]
            uid = fn[:-4].split("_", 1)[1]          # 去音色前缀
            base = uid if uid in EXPECT else uid.rsplit("_", 1)[0]
            expect = EXPECT[base]
            act = parse_intent(hyp)
            ok = act["intent"] == expect
            slot_ok = True
            if cat == "value":
                mnum = re.findall(r"\d+(?:\.\d+)?", norm(ref))
                exp_v = float(mnum[0]) if mnum else None
                got = list(act["slots"].values())
                slot_ok = bool(got) and exp_v is not None and abs(float(got[0]) - exp_v) < 1e-6
                ok = ok and slot_ok
            n_ok += ok
            out.append({"file": fn, "voice": voice, "ref": ref, "hyp": hyp,
                        "intent": act["intent"], "expect": expect,
                        "slots": act["slots"], "src": act["source"],
                        "slot_ok": slot_ok, "ok": ok})

    lines = [f"端到端意图准确率: {n_ok}/{len(out)} = {100*n_ok/len(out):.1f}%"]
    from collections import Counter
    src = Counter(o["src"] for o in out)
    lines.append("NLU来源: " + ", ".join(f"{k}={v}" for k, v in src.items()))
    for vc in sorted({o["voice"] for o in out}):
        sub = [o for o in out if o["voice"] == vc]
        lines.append(f"音色[{vc}] {sum(o['ok'] for o in sub)}/{len(sub)}")
    fails = [o for o in out if not o["ok"]]
    lines.append(f"失败 {len(fails)} 条:")
    for o in fails:
        lines.append(f"  [{o['voice']}] {o['file']}\n    hyp: {o['hyp']}\n"
                     f"    got: {o['intent']} expect: {o['expect']} slots: {o['slots']}")
    summary = "\n".join(lines)
    with open(os.path.join(RESULT, "summary_e2e.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    with open(os.path.join(RESULT, "e2e_intent.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(summary.encode("gbk", "replace").decode("gbk") if os.name == "nt" else summary)

if __name__ == "__main__":
    main()
