import json
import sys
from collections import defaultdict

TRACE = {
    "resnet50":         "traces/resnet50_bs16_kernel_trace.json",
    "resnext50_32x4d":  "traces/resnext50_32x4d_bs16_kernel_trace.json",
}


def load_active_window(path):
    """找到 ACTIVE_STEP 标记，返回 (ts_start, ts_end)。"""
    with open(path) as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    for e in events:
        if e.get("name") == "ACTIVE_STEP" and e.get("ph") == "X":
            return e["ts"], e["ts"] + e.get("dur", 0), events
    raise RuntimeError(f"ACTIVE_STEP marker not found in {path}")


def collect_kernels(path):
    ts0, ts1, events = load_active_window(path)
    stats = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for e in events:
        if e.get("cat") != "kernel":
            continue
        ts = e.get("ts", -1)
        if not (ts0 <= ts <= ts1):
            continue
        name = e["name"]
        stats[name]["count"] += 1
        stats[name]["total_us"] += e.get("dur", 0.0)
    return stats, (ts1 - ts0)


def main():
    data = {}
    windows = {}
    for model, path in TRACE.items():
        stats, win_us = collect_kernels(path)
        data[model] = stats
        windows[model] = win_us
        print(f"[{model}] active step window: {win_us:.0f} us, "
              f"total kernel launches: {sum(s['count'] for s in stats.values())}")

    # 按 resnet 和 resnext 的计数差排序，差异大的排前面
    all_names = set(data["resnet50"]) | set(data["resnext50_32x4d"])
    rows = []
    for name in all_names:
        c_rn = data["resnet50"].get(name, {"count": 0, "total_us": 0.0})
        c_rx = data["resnext50_32x4d"].get(name, {"count": 0, "total_us": 0.0})
        rows.append((name, c_rn["count"], c_rx["count"],
                     c_rn["total_us"], c_rx["total_us"]))
    rows.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)

    print("\n=== Kernel comparison (active step only) ===")
    print(f"{'count_RN':>8} {'count_RX':>8} {'d_count':>8} "
          f"{'us_RN':>10} {'us_RX':>10}  kernel_name")
    for name, cn, cx, tn, tx in rows:
        short = name if len(name) <= 100 else name[:97] + "..."
        print(f"{cn:>8} {cx:>8} {cx - cn:>+8d} "
              f"{tn:>10.1f} {tx:>10.1f}  {short}")


if __name__ == "__main__":
    main()