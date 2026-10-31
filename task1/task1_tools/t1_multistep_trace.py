#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t1_multistep_trace.py —— 多步稳态采样，供稳定性分析切窗。

采 warmup=3、active=5，每步插入 record_function("STEP#i") 标记。
输出  *_multi5_kernel_trace.json / *_multi5_meta.json
用法  python3 t1_multistep_trace.py --output-dir ./task1_out
"""

import argparse
import json
import os
import sys


def build_model(name):
    import torchvision
    if name == "resnet50":
        return torchvision.models.resnet50(weights=None)
    if name == "resnext50_32x4d":
        return torchvision.models.resnext50_32x4d(weights=None)
    raise ValueError(f"unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=3,
                    help="前 N 步不记录（cuDNN 算法选择在此期间收敛），默认 3")
    ap.add_argument("--active", type=int, default=5,
                    help="记录的稳态步数，默认 5（至少 3 才有统计意义）")
    ap.add_argument("--output-dir", default="./task1_out")
    args = ap.parse_args()

    if args.active < 3:
        sys.stderr.write("[ERROR] --active 至少 3，否则算不出变异系数\n")
        sys.exit(1)

    import torch
    import torch.nn.functional as F
    from torch.profiler import profile, schedule, ProfilerActivity

    if not torch.cuda.is_available():
        sys.stderr.write("[ERROR] 无 CUDA 设备，本脚本必须在真实 GPU 上跑\n")
        sys.exit(1)

    device = torch.device("cuda")
    model = build_model(args.model).to(device)
    model.train()

    inputs = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                         device=device)
    targets = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.model}_bs{args.batch_size}_multi{args.active}"
    trace_path = os.path.join(args.output_dir, tag + "_kernel_trace.json")
    meta_path = os.path.join(args.output_dir, tag + "_meta.json")

    total_steps = args.warmup + args.active
    # wait=warmup：前 warmup 步整体不记录；warmup=0：不再额外预热；
    # active：连续记录 active 步；repeat=1：只做一轮。
    prof_schedule = schedule(wait=args.warmup, warmup=0, active=args.active, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=None,
        record_shapes=True,
        profile_memory=False,          # 多步必须关闭，否则 trace 体积爆炸
    ) as prof:
        for step in range(total_steps):
            # 每步都打标记。未记录的步不会出现在 trace 里，正好用来切窗。
            with torch.profiler.record_function(f"STEP#{step}"):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(trace_path)

    meta = {
        "model": args.model,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "num_classes": args.num_classes,
        "mode": "train",
        "warmup_steps": args.warmup,
        "active_steps": args.active,
        "total_steps": total_steps,
        "recorded_step_range": [args.warmup, total_steps - 1],
        "trace_path": trace_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "note": "profile_memory=off；显存侧证据沿用单步 mem_snapshot.pickle",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] trace: {trace_path}")
    print(f"[OK] meta : {meta_path}")
    print(f"[INFO] 记录步范围 STEP#{args.warmup} .. STEP#{total_steps - 1}")


if __name__ == "__main__":
    main()
