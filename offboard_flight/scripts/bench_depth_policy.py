#!/usr/bin/env python3
"""Inference-time benchmark for the tiny depth->velocity policy on Jetson Xavier.

Architecture (as specified):
    input 1 x 180 x 320  (640x360 depth, downsampled 2x)
      Conv 5x5 s2, 16ch          -> 16 x 90 x 160
      DW-sep Conv s2, 32ch       -> 32 x 45 x 80
      DW-sep Conv s2, 64ch       -> 64 x 23 x 40
      DW-sep Conv s2, 96ch       -> 96 x 12 x 20
      Conv 1x1, 96 -> 32
      AdaptiveAvgPool -> 32 x 3 x 5
      flatten                    -> 480
    concat [img_feat(480), state(state_dim)]
      MLP 480+state -> 256 -> 128 -> action

No dataset required; runs on synthetic tensors.

Usage:
    python3 bench_depth_policy.py                 # fp32 + fp16, batch 1
    python3 bench_depth_policy.py --batch 1 4 8
    python3 bench_depth_policy.py --onnx out.onnx # also export ONNX for trtexec
"""

import argparse
import statistics
import time

import torch
import torch.nn as nn


# ---------------------------------------------------------------- model

class DWSep(nn.Module):
    """Depthwise-separable conv: 3x3 depthwise + 1x1 pointwise, BN + ReLU."""

    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1,
                            groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class TinyDepthPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = DWSep(16, 32, 2)
        self.block2 = DWSep(32, 64, 2)
        self.block3 = DWSep(64, 96, 2)
        self.reduce = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 5))          # -> 32 x 3 x 5 = 480
        self.mlp = nn.Sequential(
            nn.Linear(480 + state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, depth, state):
        x = self.stem(depth)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.reduce(x)
        x = self.pool(x).flatten(1)
        return self.mlp(torch.cat([x, state], dim=1))


# ---------------------------------------------------------------- helpers

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def shape_trace(model, device):
    """Print the actual tensor shape after each stage (verifies the spec)."""
    x = torch.zeros(1, 1, 180, 320, device=device)
    rows = [("input", tuple(x.shape))]
    with torch.no_grad():
        for name in ("stem", "block1", "block2", "block3", "reduce"):
            x = getattr(model, name)(x)
            rows.append((name, tuple(x.shape)))
        x = model.pool(x)
        rows.append(("pool", tuple(x.shape)))
        rows.append(("flatten", (1, x.numel())))
    print("\n--- shape trace ---")
    for n, s in rows:
        print(f"  {n:<8} {s}")


def bench(model, depth, state, iters=300, warmup=50):
    """Return per-iteration latencies in ms (GPU-synchronised)."""
    with torch.no_grad():
        for _ in range(warmup):
            model(depth, state)
        torch.cuda.synchronize()

        lat = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(depth, state)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1e3)
    return lat


def report(tag, lat, batch):
    lat = sorted(lat)
    mean = statistics.mean(lat)
    p50 = lat[len(lat) // 2]
    p99 = lat[int(len(lat) * 0.99) - 1]
    print(f"  {tag:<22} mean {mean:7.3f} ms | p50 {p50:7.3f} | p99 {p99:7.3f} "
          f"| min {lat[0]:6.3f} | {1000.0 / mean:7.1f} inf/s "
          f"| {1000.0 * batch / mean:7.1f} img/s")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--state-dim", type=int, default=6,
                    help="velocity(3) + goal dir(2) + yaw(1)")
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--cpu", action="store_true", help="also benchmark CPU fp32")
    ap.add_argument("--onnx", type=str, default=None,
                    help="export ONNX to this path (for trtexec)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = TinyDepthPolicy(args.state_dim, args.action_dim).to(dev).eval()
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}  cudnn {torch.backends.cudnn.version()}")
    print(f"params      : {count_params(model):,}")
    print(f"state_dim   : {args.state_dim}   action_dim: {args.action_dim}")
    shape_trace(model, dev)

    print(f"\n--- latency ({args.iters} iters, {args.warmup} warmup) ---")
    for b in args.batch:
        depth = torch.randn(b, 1, 180, 320, device=dev)
        state = torch.randn(b, args.state_dim, device=dev)

        report(f"fp32 batch={b}", bench(model, depth, state, args.iters, args.warmup), b)

        half = TinyDepthPolicy(args.state_dim, args.action_dim).to(dev).eval().half()
        report(f"fp16 batch={b}",
               bench(half, depth.half(), state.half(), args.iters, args.warmup), b)
        del half

    if args.cpu:
        cm = TinyDepthPolicy(args.state_dim, args.action_dim).eval()
        d = torch.randn(1, 1, 180, 320)
        s = torch.randn(1, args.state_dim)
        with torch.no_grad():
            for _ in range(10):
                cm(d, s)
            lat = []
            for _ in range(50):
                t0 = time.perf_counter()
                cm(d, s)
                lat.append((time.perf_counter() - t0) * 1e3)
        report("cpu fp32 batch=1", lat, 1)

    if args.onnx:
        m = TinyDepthPolicy(args.state_dim, args.action_dim).to(dev).eval()
        torch.onnx.export(
            m,
            (torch.randn(1, 1, 180, 320, device=dev),
             torch.randn(1, args.state_dim, device=dev)),
            args.onnx,
            input_names=["depth", "state"], output_names=["action"],
            opset_version=13,
        )
        print(f"\nONNX written to {args.onnx}")
        print(f"  trtexec --onnx={args.onnx} --fp16 --iterations=1000")


if __name__ == "__main__":
    main()
