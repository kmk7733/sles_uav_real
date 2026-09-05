#!/usr/bin/env python3
"""Run the tiny depth policy TRT engine at a fixed rate and report latency.

Emulates a deployed policy node: one inference per control tick. Uses torch
CUDA tensors as the device buffers so no pycuda is needed.

Usage:
    python3 trt_policy_runner.py engine.plan --rate 30 --duration 25
    python3 trt_policy_runner.py engine.plan --rate 0   # flat out
"""

import argparse
import statistics
import time

import tensorrt as trt
import torch

TRT_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--rate", type=float, default=30.0, help="Hz; 0 = flat out")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--tag", default="")
    ap.add_argument("--blocking-sync", action="store_true",
                    help="wait on a blocking CUDA event (thread sleeps) instead of "
                         "torch.cuda.synchronize(), which spin-waits and burns a core")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # allocate a torch CUDA tensor per binding; pass data_ptr() to TRT
    bufs, ptrs = {}, []
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(ctx.get_binding_shape(i))
        dt = TRT_DTYPE[engine.get_binding_dtype(i)]
        t = torch.zeros(shape, dtype=dt, device="cuda")
        if engine.binding_is_input(i):
            t.normal_()
        bufs[name] = t
        ptrs.append(int(t.data_ptr()))
        print(f"  binding {i}: {name:<8} {shape} {dt} "
              f"{'in' if engine.binding_is_input(i) else 'out'}")

    stream = torch.cuda.current_stream().cuda_stream

    if args.blocking_sync:
        done = torch.cuda.Event(blocking=True)

        def infer():
            ctx.execute_async_v2(ptrs, stream)
            done.record()
            done.synchronize()
    else:
        def infer():
            ctx.execute_async_v2(ptrs, stream)
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        infer()

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    lat, miss = [], 0
    t_end = time.perf_counter() + args.duration
    next_tick = time.perf_counter()

    while time.perf_counter() < t_end:
        if period:
            now = time.perf_counter()
            if next_tick > now:
                time.sleep(next_tick - now)
            elif now - next_tick > period:      # fell a whole tick behind
                miss += 1
            next_tick += period

        t0 = time.perf_counter()
        infer()
        lat.append((time.perf_counter() - t0) * 1e3)

    lat.sort()
    n = len(lat)
    print(f"\n--- TRT inference {args.tag} "
          f"(rate={'max' if not period else f'{args.rate:g}Hz'}, {n} inferences) ---")
    print(f"  mean {statistics.mean(lat):7.3f} ms | p50 {lat[n // 2]:7.3f} | "
          f"p90 {lat[int(n * 0.90)]:7.3f} | p99 {lat[int(n * 0.99)]:7.3f} | "
          f"max {lat[-1]:7.3f} | min {lat[0]:6.3f}")
    if period:
        print(f"  achieved {n / args.duration:.2f} Hz, missed ticks: {miss}")


if __name__ == "__main__":
    main()
