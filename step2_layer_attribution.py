import json
from collections import defaultdict

TRACES = [
    ("traces/resnet50_bs16_kernel_trace.json", "ResNet50"),
    ("traces/resnext50_32x4d_bs16_kernel_trace.json", "ResNeXt50_32x4d"),
]

FWD_LAUNCHER = "aten::cudnn_convolution"
BWD_LAUNCHER = "aten::convolution_backward"


def load_events(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def active_window(events):
    for e in events:
        if e.get("name") == "ACTIVE_STEP" and e.get("ph") == "X":
            return e["ts"], e["ts"] + e.get("dur", 0)
    return -1e18, 1e18


def analyze(path, label):
    events = load_events(path)
    ts0, ts1 = active_window(events)

    kernels = [e for e in events if e.get("cat") == "kernel"
               and ts0 <= e.get("ts", 0) <= ts1]
    kernel_by_ext = defaultdict(list)
    for k in kernels:
        ext = (k.get("args") or {}).get("External id")
        if ext is not None:
            kernel_by_ext[ext].append(k)

    fwd_ops, bwd_ops = [], []
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        ts = e.get("ts", -1)
        if not (ts0 <= ts <= ts1):
            continue
        n = e.get("name")
        if n == FWD_LAUNCHER:
            fwd_ops.append(e)
        elif n == BWD_LAUNCHER:
            bwd_ops.append(e)
    fwd_ops.sort(key=lambda e: e["ts"])
    bwd_ops.sort(key=lambda e: e["ts"])

    def op_kernels(op):
        ext = (op.get("args") or {}).get("External id")
        return kernel_by_ext.get(ext, []) if ext is not None else []

    lines = []
    matched_fwd = sum(1 for op in fwd_ops if op_kernels(op))
    matched_bwd = sum(1 for op in bwd_ops if op_kernels(op))
    lines.append(f"=== {label} ===")
    lines.append(f"active window: {ts1 - ts0:.0f} us")
    lines.append(f"forward launcher ops: {len(fwd_ops)}, with kernels: {matched_fwd}")
    lines.append(f"backward launcher ops: {len(bwd_ops)}, with kernels: {matched_bwd}")
    lines.append("")

    for tag, ops in (("forward", fwd_ops), ("backward", bwd_ops)):
        for i, op in enumerate(ops):
            ks = op_kernels(op)
            shapes = (op.get("args") or {}).get("Input Dims")
            lines.append(f"[{tag} {i:02d}] shapes={shapes}  kernels={len(ks)}")
            agg = defaultdict(int)
            for k in ks:
                agg[k["name"]] += 1
            for kn, c in sorted(agg.items(), key=lambda x: -x[1]):
                lines.append(f"    [{c}] {kn[:120]}")
            lines.append("")

    out = f"traces/{label}_layer_attribution_v2.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:8]))
    print(f"... wrote {out}")


if __name__ == "__main__":
    for p, l in TRACES:
        analyze(p, l)