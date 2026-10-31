import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import profile, schedule, ProfilerActivity

INPUT_CHANNELS = 3
INPUT_SIZE = 224
NUM_CLASSES = 1000


def build_model(name: str):
    import torchvision
    if name == "resnet50":
        return torchvision.models.resnet50(weights=None)
    if name == "resnext50_32x4d":
        return torchvision.models.resnext50_32x4d(weights=None)
    raise ValueError(f"Unknown model: {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet50", "resnext50_32x4d"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="./traces")
    parser.add_argument("--active-step", type=int, default=2,
                        help="0-indexed step marked as steady-state active step (must be >= 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.active_step < 1:
        sys.stderr.write("[ERROR] --active-step must be >= 1 so cuDNN reaches steady state.\n")
        sys.exit(1)
    if not torch.cuda.is_available():
        sys.stderr.write("[ERROR] No CUDA device. This script requires a real GPU.\n")
        sys.exit(1)

    device = torch.device("cuda")
    model = build_model(args.model).to(device)
    model.train()

    inputs = torch.randn(args.batch_size, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)
    targets = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)
    criterion = F.cross_entropy

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.model}_bs{args.batch_size}"
    trace_path = os.path.join(args.output_dir, f"{tag}_kernel_trace.json")
    snapshot_path = os.path.join(args.output_dir, f"{tag}_mem_snapshot.pickle")
    meta_path = os.path.join(args.output_dir, f"{tag}_meta.json")

    total_steps = args.active_step + 1
    # 只记录 active step：前 active_step 步全部 wait，第 active_step 步为 active。
    prof_schedule = schedule(wait=args.active_step, warmup=0, active=1, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for step in range(total_steps):
            if step == args.active_step:
                # 只对 active step 开启 tensor 级内存历史，避免 warmup 步的
                # cuDNN workspace / 内存池扩展污染 tensor 周期统计。
                torch.cuda.memory._record_memory_history(
                    enabled="all", context="all", stacks="all"
                )
                with torch.profiler.record_function("ACTIVE_STEP"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                torch.cuda.synchronize()
                torch.cuda.memory._dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=False)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(trace_path)

    meta = {
        "model": args.model,
        "batch_size": args.batch_size,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "mode": "train",
        "total_steps": total_steps,
        "active_step": args.active_step,
        "trace_path": trace_path,
        "snapshot_path": snapshot_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] trace:    {trace_path}")
    print(f"[OK] snapshot: {snapshot_path}")
    print(f"[OK] meta:     {meta_path}")


if __name__ == "__main__":
    main()