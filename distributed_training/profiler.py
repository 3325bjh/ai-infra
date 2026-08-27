"""PyTorch Profiler 示例。

示例：
    python profiler.py --profile-memory --record-shapes --export-trace traces/trace.json
    python profiler.py --steps 12 --schedule --trace-dir traces
"""

import argparse
import os

import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function, schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Profiler 示例")
    parser.add_argument("--batch-size", type=int, default=5, help="输入 batch 大小。")
    parser.add_argument("--steps", type=int, default=1, help="执行的前向传播步数。")
    parser.add_argument("--record-shapes", action="store_true", help="记录算子输入形状。")
    parser.add_argument("--profile-memory", action="store_true", help="记录 Tensor 内存分配和释放。")
    parser.add_argument("--with-stack", action="store_true", help="记录 Python/TorchScript 调用栈，开销较大。")
    parser.add_argument("--group-by-input-shape", action="store_true", help="按输入 shape 分组算子统计。")
    parser.add_argument("--export-trace", type=str, default=None, help="导出单次采样的 Chrome trace JSON 文件。")
    parser.add_argument("--schedule", action="store_true", help="使用 wait/warmup/active 周期采样长任务。")
    parser.add_argument("--wait", type=int, default=1, help="每个周期不采样的步数。")
    parser.add_argument("--warmup", type=int, default=1, help="每个周期预热采样步数。")
    parser.add_argument("--active", type=int, default=2, help="每个周期正式记录步数。")
    parser.add_argument("--repeat", type=int, default=1, help="采样周期重复次数，0 表示直到训练结束。")
    parser.add_argument("--trace-dir", type=str, default="traces", help="周期采样 trace 的输出目录。")
    return parser.parse_args()


def choose_device():
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        return torch.device("cuda"), activities + [ProfilerActivity.CUDA], "cuda_time_total"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), activities + [ProfilerActivity.XPU], "xpu_time_total"
    return torch.device("cpu"), activities, "cpu_time_total"


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps 必须至少为 1。")
    device, activities, sort_by = choose_device()
    model = models.resnet18().to(device).eval()
    inputs = torch.randn(args.batch_size, 3, 224, 224, device=device)

    def trace_handler(prof) -> None:
        os.makedirs(args.trace_dir, exist_ok=True)
        path = os.path.join(args.trace_dir, f"trace_step_{prof.step_num}.json")
        prof.export_chrome_trace(path)
        print(f"已导出 trace：{path}", flush=True)
        print(prof.key_averages().table(sort_by=sort_by, row_limit=10), flush=True)

    profiler_kwargs = {
        "activities": activities,
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
        "with_stack": args.with_stack,
    }
    if args.schedule:
        profiler_kwargs["schedule"] = schedule(
            wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.repeat
        )
        profiler_kwargs["on_trace_ready"] = trace_handler

    with torch.no_grad(), profile(**profiler_kwargs) as prof:
        for step in range(args.steps):
            with record_function("model_inference"):
                model(inputs)
            if args.schedule:
                prof.step()  # 通知 profiler：一个训练/推理 step 已结束。

    if not args.schedule:
        print(
            prof.key_averages(group_by_input_shape=args.group_by_input_shape).table(
                sort_by=sort_by, row_limit=10
            )
        )
        if args.export_trace:
            parent = os.path.dirname(args.export_trace)
            if parent:
                os.makedirs(parent, exist_ok=True)
            prof.export_chrome_trace(args.export_trace)
            print(f"已导出 trace：{args.export_trace}")


if __name__ == "__main__":
    main()
