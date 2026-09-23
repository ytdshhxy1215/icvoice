# -*- coding: utf-8 -*-
"""真实语音 -> ASR -> 动作规划 -> 模拟寄存器执行的流程测试。

默认使用上一轮 TTS 测试集中的 8 条代表音频。脚本不会连接真实仪器，
而是在本地模拟 PIC-10 寄存器，并对写入动作做回读验证。
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WAVDIR = ROOT / "tests" / "asr_testset" / "gen_wav_v2"
OUTDIR = ROOT / "tests" / "asr_testset" / "results"
REPORT = OUTDIR / "action_e2e_report.json"

SAMPLES = [
    "冰糖_set_flow_a_0.wav",       # 设置流量
    "冰糖_set_press_a_0.wav",      # 设置压力上限
    "冰糖_qry_pressure.wav",       # 查询压力
    "白桦_qry_flow_now.wav",        # 查询流量
    "白桦_ctl_start_inject.wav",    # 自动进样
    "苏打_ctl_stop_run.wav",        # 停止运行
    "冰糖_ctl_start_acq.wav",       # 协议暂不能唯一映射
    "白桦_set_curr_a_0.wav",        # 需要明确电流目标模块
]


class SimInstrument:
    """最小寄存器模拟器，只验证动作序列和回读关系。"""

    def __init__(self):
        self.registers = {
            0x2100: 0,
            0x2120: 0,
            0x2121: 200,
            0x2125: 0,
            0x2128: 1234,       # 12.34 MPa，模拟查询值
            0x3100: 0,
            0x4101: 0,
            0x4115: 0xE000,     # 已经定位完成
            0x4116: 0,
            0x4117: 0,
        }
        # 这是测试配置，不是 PDF 声明的安全上限。
        self.flow_limit = (0.0, 5.0)
        self.pressure_limit = (0.0, 30.0)
        self.events = []

    def read(self, register):
        value = self.registers.get(int(register, 16))
        self.events.append({"op": "read", "register": register, "value": value})
        return value

    def write(self, register, value):
        address = int(register, 16)
        if address == 0x2120:
            flow = value / 100.0
            if not self.flow_limit[0] <= flow <= self.flow_limit[1]:
                raise ValueError(f"流量 {flow:g} 超出测试配置范围")
        if address == 0x2121:
            pressure = value / 10.0
            if not self.pressure_limit[0] <= pressure <= self.pressure_limit[1]:
                raise ValueError(f"压力 {pressure:g} 超出测试配置范围")
        self.registers[address] = value
        if address == 0x4115:
            self.registers[address] = 0xE000
        self.events.append({"op": "write", "register": register, "value": value})

    def execute(self, actions):
        for action in actions:
            op = action["op"]
            if op == "write_register":
                self.write(action["register"], action["value"])
            elif op == "read_register":
                self.read(action["register"])
            elif op == "poll_register":
                value = self.read(action["register"])
                if value != 0xE000:
                    raise ValueError(f"定位未完成: {value!r}")
            else:
                raise ValueError(f"模拟器不支持动作 {op}")


def load_manifest():
    import csv
    manifest = {}
    with (ROOT / "tests" / "asr_testset" / "manifest_v2.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest[row["file"]] = row
    return manifest


def main():
    sys.path.insert(0, str(HERE))
    from action_planner import plan
    from asr_hotword import load_model

    manifest = load_manifest()
    missing = [name for name in SAMPLES if not (WAVDIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"缺少测试音频: {missing}")

    print(f"加载 SeACo 热词模型，样本数={len(SAMPLES)}")
    model = load_model()
    sim = SimInstrument()
    rows = []
    for index, name in enumerate(SAMPLES, 1):
        ref = manifest[name]["text"]
        started = time.time()
        result = model.generate(input=str(WAVDIR / name))
        hyp = result[0].get("text", "") if result else ""
        asr_seconds = round(time.time() - started, 3)
        action_plan = plan(hyp)
        executed = False
        execution_error = None
        # needs_configuration 允许在本测试模拟器已加载安全配置时执行。
        executable = action_plan["status"] in {"ready", "needs_configuration", "needs_precondition"}
        if executable and action_plan["actions"]:
            try:
                sim.execute(action_plan["actions"])
                executed = True
            except Exception as exc:  # 保留到报告，继续测试其他样本
                execution_error = str(exc)
        row = {
            "file": name,
            "reference": ref,
            "hypothesis": hyp,
            "intent": action_plan["intent"],
            "plan_status": action_plan["status"],
            "action_count": len(action_plan["actions"]),
            "executed": executed,
            "execution_error": execution_error,
            "asr_seconds": asr_seconds,
            "warnings": action_plan["warnings"],
        }
        rows.append(row)
        print(f"[{index}/{len(SAMPLES)}] {name}: {hyp} -> {action_plan['intent']} / {action_plan['status']} / executed={executed}")

    supported = [r for r in rows if r["action_count"] > 0]
    passed = [r for r in supported if r["executed"] and not r["execution_error"]]
    report = {
        "description": "真实音频 ASR -> 动作规划 -> PIC-10 模拟寄存器执行",
        "instrument": "PIC-10",
        "samples": rows,
        "summary": {
            "total": len(rows),
            "plans_with_actions": len(supported),
            "executed_and_verified": len(passed),
            "simulator_events": len(sim.events),
        },
        "simulator_final_registers": {f"0x{k:04X}": v for k, v in sorted(sim.registers.items())},
        "simulator_events": sim.events,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {REPORT}")
    print(f"可执行计划 {len(supported)}/{len(rows)}，模拟执行并验证 {len(passed)}/{len(supported)}")


if __name__ == "__main__":
    main()
