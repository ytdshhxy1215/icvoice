# -*- coding: utf-8 -*-
"""将 ASR 文本转换为安全的仪器动作计划。

这是协议对接前的确定性基线：只生成 JSON 计划，不打开串口、不发送报文。
后续可以把同一个输出 schema 交给本地 LLM，再由校验器和 InstrumentClient 执行。
"""
import argparse
import json
import sys

HERE = __import__("os").path.dirname(__file__)
sys.path.insert(0, HERE)
from realtime_pipeline import parse_intent  # noqa: E402


def write(register, value, description, unit="raw"):
    return {
        "op": "write_register",
        "register": register,
        "value": value,
        "unit": unit,
        "description": description,
    }


def read(register, description):
    return {"op": "read_register", "register": register, "description": description}


def plan(text):
    parsed = parse_intent(text)
    intent = parsed["intent"]
    slots = parsed.get("slots", {})
    base = {
        "schema_version": "instrument-action-plan.v1",
        "source_text": text,
        "intent": intent,
        "status": "ready",
        "confidence": 1.0 if parsed.get("source") == "rule" else 0.8,
        "actions": [],
        "warnings": [],
    }

    if intent == "set_flow":
        value = float(slots["flow"])
        if value < 0.0:
            base.update(status="rejected", warnings=["流量不能为负数"])
            return base
        base["actions"] = [
            write("0x2120", round(value * 100), f"设置泵流量为 {value:g} mL/min"),
            read("0x2120", "回读确认泵流量"),
        ]
        base["status"] = "needs_configuration"
        base["warnings"].append("协议未给出泵流量安全上限，执行前需加载仪器配置")
        return base

    if intent == "set_pressure":
        value = float(slots["pressure_mpa"])
        if value < 0.0:
            base.update(status="rejected", warnings=["压力上限不能为负数"])
            return base
        base["actions"] = [
            write("0x2121", round(value * 10), f"设置泵最大压力为 {value:g} MPa"),
            read("0x2121", "回读确认泵最大压力"),
        ]
        base["status"] = "needs_configuration"
        base["warnings"].append("协议未给出泵压力安全上限，执行前需加载仪器配置")
        return base

    if intent == "query_pressure":
        base["actions"] = [read("0x2128", "读取当前泵压力，原始值除以 100")]
        return base
    if intent == "query_flow":
        base["actions"] = [read("0x2120", "读取当前泵流量，原始值除以 100")]
        return base
    if intent == "start_inject":
        base.update(status="needs_precondition")
        base["actions"] = [
            read("0x4115", "读取当前进样器位置"),
            {"op": "poll_register", "register": "0x4115", "until": "OPH:OPL == E000", "timeout_ms": 30000},
            write("0x4117", 1, "落针"),
            write("0x4116", 0x50, "开始进样"),
        ]
        base["warnings"].append("目标瓶号必须在执行前由上位机状态或用户明确提供")
        return base
    if intent in {"stop_run", "emergency_stop"}:
        base["actions"] = [
            write("0x4116", 0x00, "停止进样"),
            write("0x4101", 0, "停止自动进样器"),
            write("0x3100", 0, "停止伏安分析"),
            write("0x2125", 0, "停止泵"),
            read("0x4101", "验证自动进样器状态"),
            read("0x2125", "验证泵状态"),
        ]
        if intent == "emergency_stop":
            base["status"] = "needs_policy"
        base["warnings"].append("急停动作必须由现场安全策略确认是否同时停止全部模块")
        return base

    if intent == "set_current":
        base.update(status="needs_clarification")
        base["warnings"].append("协议同时存在电导电流和淋洗液发生器电流，需明确目标模块")
        return base

    base.update(status="unsupported")
    base["warnings"].append("当前协议动作注册表没有该意图的确定映射")
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+")
    args = parser.parse_args()
    print(json.dumps(plan("".join(args.text)), ensure_ascii=False, indent=2))
