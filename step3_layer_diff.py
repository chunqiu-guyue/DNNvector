import re
from collections import defaultdict

FILES = {
    "RN": "traces/ResNet50_layer_attribution_v2.txt",
    "RX": "traces/ResNeXt50_32x4d_layer_attribution_v2.txt",
}


def parse(path):
    """返回 {phase: {layer_idx: (shapes, {kernel_name: count})}}"""
    out = {"forward": {}, "backward": {}}
    cur_phase = cur_idx = cur_shapes = None
    cur_agg = None
    head_re = re.compile(r"\[(forward|backward) (\d+)\] shapes=(.*?)  kernels=(\d+)")
    kern_re = re.compile(r"^\s+\[(\d+)\] (.+)$")
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = head_re.match(line)
            if m:
                if cur_phase is not None:
                    out[cur_phase][cur_idx] = (cur_shapes, cur_agg)
                cur_phase, cur_idx = m.group(1), int(m.group(2))
                cur_shapes = m.group(3)
                cur_agg = defaultdict(int)
                continue
            m = kern_re.match(line)
            if m and cur_agg is not None:
                cur_agg[m.group(2).strip()] += int(m.group(1))
    if cur_phase is not None:
        out[cur_phase][cur_idx] = (cur_shapes, cur_agg)
    return out


def short(name, n=95):
    return name if len(name) <= n else name[:n - 3] + "..."


def fmt_set(agg):
    return {short(k): c for k, c in agg.items()}


def diff_phase(phase, rn, rx):
    idxs = sorted(set(rn) | set(rx))
    print(f"\n================ {phase.upper()} ================")
    for i in idxs:
        s_rn, a_rn = rn.get(i, ("<missing>", {}))
        s_rx, a_rx = rx.get(i, ("<missing>", {}))
        set_rn, set_rx = fmt_set(a_rn), fmt_set(a_rx)
        if set_rn == set_rx:
            continue
        n_rn = sum(a_rn.values())
        n_rx = sum(a_rx.values())
        print(f"\n[{phase} {i:02d}] kernels: RN={n_rn}  RX={n_rx}")
        print(f"  shapes RN: {s_rn[:110]}")
        print(f"  shapes RX: {s_rx[:110]}")
        only_rn = set(set_rn) - set(set_rx)
        only_rx = set(set_rx) - set(set_rn)
        both = set(set_rn) & set(set_rx)
        for k in sorted(only_rn):
            print(f"  - RN-only [{set_rn[k]}] {k}")
        for k in sorted(only_rx):
            print(f"  + RX-only [{set_rx[k]}] {k}")
        for k in sorted(both):
            if set_rn[k] != set_rx[k]:
                print(f"  ~ both RN={set_rn[k]} RX={set_rx[k]}: {k}")


if __name__ == "__main__":
    rn = parse(FILES["RN"])
    rx = parse(FILES["RX"])
    for phase in ("forward", "backward"):
        diff_phase(phase, rn[phase], rx[phase])