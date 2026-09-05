# Tiny depth policy — inference-time benchmark on Xavier

Feasibility measurement for replacing (or augmenting) the MPPI planner with a
small learned depth→velocity policy. The question was only *how fast does it
run onboard*, so this measures latency on synthetic tensors with random
weights. No dataset, no training, no accuracy claim.

Every number in this document was measured on this vehicle's Jetson Xavier on
2026-08-06, against the live ZED + grid + MPPI stack, not taken from a
datasheet. Where something is unverified it says so.

---

## 0. Conclusion

**Inference is not a bottleneck at any control rate this vehicle uses**, and at
10 Hz there is no reason to deploy TensorRT.

| | latency (loaded) | budget @10 Hz | margin |
|---|---|---|---|
| PyTorch eager | p99 **30.3 ms** | 100 ms | 3.3× |
| TensorRT FP16 | p99 **3.9 ms** | 100 ms | 26× |

Recommendation: **ship PyTorch, keep the ONNX export path.** TensorRT buys
8× on p99 that a 10 Hz loop cannot spend, and costs version-locked engine
files, manual buffer/binding management, and a second numerical path to
validate against training. Revisit only if the control rate goes above ~30 Hz.

For scale: the existing CPU MPPI `plan()` takes **22.6 ms**, i.e. ~66× the
policy's TensorRT inference and comparable to PyTorch's worst case.

---

## 1. Architecture measured

```
640 x 360 depth
      | downsample 2x
1 x 180 x 320
      | Conv 5x5 s2, 16ch          -> 16 x  90 x 160
      | DW-sep Conv s2, 32ch       -> 32 x  45 x  80
      | DW-sep Conv s2, 64ch       -> 64 x  23 x  40
      | DW-sep Conv s2, 96ch       -> 96 x  12 x  20
      | Conv 1x1, 96 -> 32         -> 32 x  12 x  20
      | AdaptiveAvgPool            -> 32 x   3 x   5
      | flatten                    -> 480
concat [img_feat 480, velocity 3, goal dir 2, yaw 1]
      | MLP 486 -> 256 -> 128 -> 3
desired velocity / waypoint
```

Shapes above are the *measured* trace from `bench_depth_policy.py`, not
hand-derived. Depthwise-separable block = 3x3 depthwise + 1x1 pointwise, each
followed by BN + ReLU.

**171,843 params · 12.78 M MAC · 25.6 MFLOP per inference.**

| layer | output | MAC | share |
|---|---|---|---|
| Conv 5x5 s2 (stem) | 16×90×160 | 5.76 M | **45%** |
| DW-sep → 32ch | 32×45×80 | 2.36 M | 18% |
| DW-sep → 64ch | 64×23×40 | 2.15 M | 17% |
| DW-sep → 96ch | 96×12×20 | 1.61 M | 13% |
| 1×1 Conv 96→32 | 32×12×20 | 0.74 M | 6% |
| MLP | — | 0.16 M | 1% |

The stem is 45% of the whole network — it is the only layer operating at
90×160. If this ever needs to get cheaper, start there.

Sanity check: 25.6 MFLOP against ~1.1 TFLOPS FP32 (512 CUDA cores @ 1109 MHz)
is ~0.02 ms theoretical. Measured 0.34 ms = **6% of peak**, which is expected
for a network this small — it is bound by kernel launch and memory, not
arithmetic. The measurement is conservative, not optimistic.

---

## 2. Conditions

Jetson Xavier, JetPack R35.3.1, CUDA 11.4, TensorRT 8.5.2.2, PyTorch
2.0.0+nv23.05, cuDNN 8600. Power mode **`MODE_15W_6CORE`**, `jetson_clocks`
NOT applied. MAXN would raise the ceiling; not measured.

"Loaded" = the real stack, all launched for this test and torn down after:

| node | what it does | observed cost |
|---|---|---|
| ZED2i wrapper (HD720) | CUDA stereo depth + tracking | 74–77% CPU, most of the GPU |
| `depth_to_grid.py` | cloud → 224×204 @ 5 cm `/grid_map` | ~310% CPU (3 cores) |
| `mpc_node.py` (MPPI) | 256 rollouts × 30 horizon @ 10 Hz | ~1 core |

Aggregate: **all 6 CPU cores at 85–95%, GPU bursting 60–75%.** This is a
saturated machine, which is the point.

MAVROS was deliberately not launched — nothing in this test could command the
vehicle. `mpc_node` only publishes a `nav_msgs/Path`.

---

## 3. Inference latency

Measured with `trt_policy_runner.py`, which runs one inference per control
tick like a deployed node would (not a throughput loop).

### TensorRT FP16, batch 1

| condition | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| **full stack loaded, 30 Hz** | **1.92** | 1.79 | 2.60 | 3.91 | 6.97 |
| idle system, 30 Hz | 3.41 | 3.69 | 3.80 | 3.99 | 5.32 |
| idle system, flat out | 0.50 | 0.48 | 0.58 | 1.07 | 2.59 |

`trtexec` compute-only, for reference: idle p50 0.340 / p99 0.353 ms →
loaded p50 0.409 / p99 1.897 ms. H2D copy of the 180×320 depth is 0.032 ms,
D2H of the action 0.004 ms — both negligible.

FP32 vs FP16 is 0.347 vs 0.340 ms — **a 2% difference.** There is not enough
arithmetic here for reduced precision to matter, so INT8 is not worth pursuing.

### PyTorch eager, batch 1

| condition | mean | p50 | p99 |
|---|---|---|---|
| idle | 4.67 | 4.61 | 6.39 |
| **full stack loaded** | 9.60 | **7.00** | **30.3** |
| idle, batch=8 | 3.84 | 3.73 | 5.33 |
| CPU fp32, idle | 117.4 | 111.8 | 173.4 |

Note batch=8 is *faster* than batch=1 (3.73 vs 4.61 ms p50). Feeding 8× the
data costs nothing, which means those ~4.6 ms are **entirely kernel-launch
overhead**, not computation. That is the whole PyTorch↔TensorRT gap: ~20
layers × per-launch cost, paid through the Python interpreter and CUDA driver
on an already-saturated CPU. TensorRT fuses Conv+BN+ReLU into single kernels,
prebuilds the graph, autotunes kernel selection at build time, and reuses
preallocated buffers.

---

## 4. Finding: low duty cycle traps the GPU at minimum clock

**The idle system is slower than the loaded one.** Same engine, same code.

GPU clock table is 114.75 MHz … 1109.25 MHz, governor `nvhost_podgov`. During
30 Hz inference on an otherwise idle machine the clock sits pinned at
**114.75 MHz — the floor, 9.7× below max** (read directly from
`/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq`). A 3%-duty workload never
convinces the governor to ramp.

With the ZED stack running, the GPU is busy enough to stay clocked up, so
inference lands at 1.92 ms instead of 3.41 ms. **The real deployment condition
is the favourable one**, and benchmarking the policy in isolation understates it.

Consequence: if the policy is ever run *without* the ZED loaded — bench
testing, or if depth moves off-GPU — pin clocks with `sudo jetson_clocks` or
you will silently get the 3.4 ms path instead of 0.5 ms.

---

## 5. Effect on the running stack — A/B/A

Does adding the policy starve perception or planning?

| | grid | MPPI traj | ZED pose |
|---|---|---|---|
| A1 no inference | 9.80 Hz | 10.00 Hz | 30.00 Hz |
| B inference @30 Hz | 9.40 Hz | 10.00 Hz | 30.04 Hz |
| A2 no inference | 9.72 Hz | 10.00 Hz | 29.96 Hz |

Max inter-message gap stayed ~200 ms in all three phases. The policy costs the
grid ~4% of its rate, leaves MPPI at its 10 Hz target untouched, and does not
affect pose. Steady-state CPU of the inference process is **4.2% of one core**.

**Process startup is the disruptive part, not steady-state inference.** Torch +
TensorRT init costs ~1.6 GB RSS and several seconds of heavy CPU; including it
in the window drove the grid to 7.6 Hz with a 947 ms stall. Build the engine /
load the model and run warmup inferences **before takeoff**, never in flight.

---

## 6. Comparison: the existing CPU MPPI

Measured offline with `bench_mppi.py` at the `mpc.launch` parameters
(K=256, N=30, M=1) on a synthetic 224×204 @ 5 cm grid — **idle machine, no
contention**, so these are best-case:

| | time | frequency |
|---|---|---|
| **MPPI `plan()`** | **22.6 ms** (p50 22.1, p99 29.6) | every plan, 10 Hz |
| `compute_cost_to_go()` | 18.0 ms | every 0.5 s |
| EDT build | 2.9 ms | every new grid msg |

The MPPI does **7,680 rollout-steps**, whose actual float count is *lower* than
the CNN's 25.6 MFLOP. It is 66× slower anyway. `cProfile` breakdown per `plan()`:

| stage | time | why |
|---|---|---|
| `_simulate` | 8.7 ms | horizon 30 as a sequential Python loop |
| `sample` (incl. `randn` 3.0 ms) | 6.0 ms | 23,040 gaussians per plan |
| `_reward` | 4.4 ms | |
| `_validate` | 3.1 ms | |

`step_vec` is called 60× per plan on a (256, 6) array — 94 µs per call for a
trivial amount of arithmetic. That is numpy/Python call overhead, not compute.

### On moving MPPI to GPU

**Not worth it at K=256.** The bottleneck is the sequential 30-step horizon
loop, which stays sequential on GPU; each step would pay 30–50 µs of kernel
launch, and a (256, 6) array leaves most of 512 cores idle.

GPU MPPI wins by making **K nearly free**, not by reducing latency — per-step
cost is mostly fixed overhead, so K=256→4096 would cost little. That is a
search-quality argument, not a speed one.

Cheaper wins first, if 22.6 ms ever needs to shrink:
- pre-generate the noise buffer and reuse it (−3.0 ms, ~13%, nearly free)
- `numba @njit` on the `_simulate` loop (targets the 8.7 ms)
- the 18 ms `compute_cost_to_go` is only every 0.5 s, but it lands inside one tick

At a 10 Hz / 100 ms budget, 22.6 ms is already 4× margin. No action needed now.

---

## 7. Caveats

- **Random weights, synthetic input.** Latency is weight-independent so timing
  is unaffected, but nothing here says the architecture *works*.
- **Preprocessing is not measured.** The 640×360 → 320×180 downsample, depth
  NaN/invalid handling, normalization, and ROS transport are all excluded and
  may well exceed the inference cost. The sensor→action path is the number
  that actually matters for control and it is **not yet measured**.
- `MODE_15W_6CORE` throughout; MAXN not measured.
- Loaded numbers depend on scene content — `depth_to_grid` cost varies with
  point count. Hence A/B/A rather than a single before/after.

### Measurement pitfalls hit here (re-read before re-measuring)

- `ps -o pcpu` is a **lifetime average**, not instantaneous. It reported 193%
  for a process actually using 4.2%, inflated by torch/TRT startup. Use
  `top -b -n2` and read the second sample.
- Let a process finish initializing before opening the measurement window, or
  startup cost contaminates the result (this produced a fake 23% grid-rate drop).
- `rostopic hz` piped into `grep` shows nothing due to Python stdout buffering
  — use `stdbuf -oL`, or subscribe directly (`rates.py` pattern).
- `pkill -f <pattern>` will match and kill the invoking shell if the pattern
  appears anywhere in its own command line. Use `"[p]attern"`.

---

## 8. Reproducing

```bash
# architecture trace, params, PyTorch fp32/fp16/CPU, ONNX export
python3 offboard_flight/scripts/bench_depth_policy.py --batch 1 4 8 --cpu \
        --onnx /tmp/tiny_depth_policy.onnx

# TensorRT: build engine, then benchmark
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/tiny_depth_policy.onnx --fp16 \
        --saveEngine=/tmp/tiny_depth_policy_fp16.plan --buildOnly
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/tiny_depth_policy.onnx --fp16 \
        --iterations=1000 --noDataTransfers --useSpinWait

# deployed-style, one inference per tick (this is the number to trust)
python3 offboard_flight/scripts/trt_policy_runner.py \
        /tmp/tiny_depth_policy_fp16.plan --rate 30 --duration 30

# CPU MPPI, same params as mpc.launch
python3 offboard_flight/scripts/bench_mppi.py

# GPU clock, to confirm the DVFS effect
cat /sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq
```

To reproduce the loaded condition:

```bash
roscore &
ROS_NAMESPACE=rogx2 roslaunch zed_wrapper zed2i.launch cam_pos_x:=0.106 cam_pos_z:=0.0671
roslaunch zed_rtabmap_example zed_depth_grid.launch world_frame:=map open_rviz:=false
roslaunch mpc_controller mpc.launch pose_topic:=/rogx2/zed2i/zed_node/pose
```

Files added by this work: `bench_depth_policy.py` (architecture + PyTorch/ONNX),
`trt_policy_runner.py` (rate-limited TensorRT runner, `--blocking-sync`
optional — it moved p99 from 5.0 to 4.1 ms, marginal), `bench_mppi.py` (CPU
MPPI timing).

---

## 9. Open next steps

1. **Measure the full sensor→action path** including downsample and
   preprocessing. This is the only remaining unknown that could change the
   conclusion.
2. Consider doing the 640×360→320×180 downsample on GPU, or configuring the ZED
   to output the lower resolution directly.
3. `torch.jit.trace` is an untested middle option — PyTorch-level code with some
   fusion, no engine files. Not measured.
