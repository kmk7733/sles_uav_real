// FrontierMPPI sample-batch evaluation on the GPU: one thread per sample.
//
// Reproduces, in double precision and in the same operation order, the NumPy
// path in planner/mppi.py (PlanarMPPI.plan, K > 1):
//   PlanarDynamics.clip_inputs -> CappedDynamics._rollout_batch
//   -> inputs_ok & states_ok & validator.nodes_safe
// plus the per-node lookups FrontierMPPI._cost needs (geodesic or Euclidean
// goal distance, clearance) and first_blocking_class's terminal ray. The cost
// SUMS stay in NumPy (same reduction order). Build with -fmad=false so no
// multiply-add is contracted; sqrt and division are IEEE (no fast-math).
// sin/cos/atan2 (heading wrap only) may differ from libm by an ulp.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

extern "C" {

typedef struct {
    int K, N, H, W, use_geo, pad;
    double dt, half, v_max, om_max, alpha_max, a_max_eff, dmax;
    double alpha_lim, a_lim, j_lim, v_lim, om_lim;
    double res, ox, oy, r_safe, gx, gy, penalty;
    double xi0[6], prev_clip[2], prev_raw[2];
} MppiParams;

}

static double *d_uin = 0, *d_uout = 0, *d_x = 0, *d_dgeo = 0, *d_cl = 0, *d_clear = 0, *d_field = 0, *d_t = 0;
static unsigned char *d_flags = 0;
static signed char *d_classes = 0;
static int *d_cls = 0;
static size_t cap_u = 0, cap_uo = 0, cap_x = 0, cap_k = 0, cap_n1 = 0, cap_map = 0, cap_t = 0;
static int last_K = 0, last_N = 0;

static int grow(void **ptr, size_t *cap, size_t need) {
    if (need <= *cap) return 0;
    if (*ptr) cudaFree(*ptr);
    *ptr = 0; *cap = 0;
    if (cudaMalloc(ptr, need) != cudaSuccess) return -1;
    *cap = need;
    return 0;
}

__device__ static inline long long cell(double v, double o, double res) {
    return (long long)floor((v - o) / res);
}

__global__ void eval_kernel(MppiParams p, const double *uin, double *uout, double *X,
                            unsigned char *flags, double *dgeo, double *cl,
                            const double *clear_tab, const double *field) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= p.K) return;
    const int N = p.N;
    const double *u = uin + (size_t)k * N * 3;
    double *uo = uout + (size_t)k * N * 3;
    double *x = X + (size_t)k * (N + 1) * 6;

    // --- clip_inputs: alpha box, acceleration disc, then the jerk chain ----
    double px = p.prev_clip[0], py = p.prev_clip[1];
    for (int j = 0; j < N; ++j) {
        double al = fmin(fmax(u[3 * j + 2], -p.alpha_max), p.alpha_max);
        double ax = u[3 * j], ay = u[3 * j + 1];
        double n = sqrt(ax * ax + ay * ay);
        double f = fmin(1.0, p.a_max_eff / fmax(n, 1e-12));
        ax = ax * f;
        ay = ay * f;
        double dx = ax - px, dy = ay - py;
        double dn = sqrt(dx * dx + dy * dy);
        double sc = fmin(1.0, p.dmax / fmax(dn, 1e-12));
        ax = px + dx * sc;
        ay = py + dy * sc;
        px = ax; py = ay;
        uo[3 * j] = ax; uo[3 * j + 1] = ay; uo[3 * j + 2] = al;
    }

    // --- CappedDynamics._rollout_batch ------------------------------------
    double r0 = p.xi0[0], r1 = p.xi0[1], r2 = p.xi0[4];
    double q0 = p.xi0[2], q1 = p.xi0[3], q2 = p.xi0[5];
    for (int i = 0; i < 6; ++i) x[i] = p.xi0[i];
    for (int j = 0; j < N; ++j) {
        double a0 = uo[3 * j], a1 = uo[3 * j + 1], a2 = uo[3 * j + 2];
        double n0 = q0 + p.dt * a0, n1 = q1 + p.dt * a1, n2 = q2 + p.dt * a2;
        double nn = sqrt(n0 * n0 + n1 * n1);
        if (nn > p.v_max) {
            double s = p.v_max / fmax(nn, 1e-12);
            a0 = (n0 * s - q0) / p.dt;
            a1 = (n1 * s - q1) / p.dt;
        }
        if (fabs(n2) > p.om_max) {
            double b = fmin(fmax(n2, -p.om_max), p.om_max);
            a2 = (b - q2) / p.dt;
        }
        uo[3 * j] = a0; uo[3 * j + 1] = a1; uo[3 * j + 2] = a2;
        n0 = q0 + p.dt * a0; n1 = q1 + p.dt * a1; n2 = q2 + p.dt * a2;
        double s0 = r0 + p.dt * q0 + p.half * a0;
        double s1 = r1 + p.dt * q1 + p.half * a1;
        double ang = r2 + p.dt * q2 + p.half * a2;
        double psi = atan2(sin(ang), cos(ang));
        double *xn = x + (size_t)(j + 1) * 6;
        xn[0] = s0; xn[1] = s1; xn[2] = n0; xn[3] = n1; xn[4] = psi; xn[5] = n2;
        r0 = s0; r1 = s1; r2 = psi; q0 = n0; q1 = n1; q2 = n2;
    }

    // --- inputs_ok on the applied inputs ----------------------------------
    int in_ok = 1;
    double qx = p.prev_raw[0], qy = p.prev_raw[1];
    for (int j = 0; j < N; ++j) {
        double a0 = uo[3 * j], a1 = uo[3 * j + 1], a2 = uo[3 * j + 2];
        if (!(isfinite(a0) && isfinite(a1) && isfinite(a2))) in_ok = 0;
        if (!(fabs(a2) <= p.alpha_lim)) in_ok = 0;
        if (!(sqrt(a0 * a0 + a1 * a1) <= p.a_lim)) in_ok = 0;
        double dx = a0 - qx, dy = a1 - qy;
        if (!(sqrt(dx * dx + dy * dy) <= p.j_lim)) in_ok = 0;
        qx = a0; qy = a1;
    }

    // --- states_ok, nodes_safe, per-node lookups ---------------------------
    int st_ok = 1, safe = 1;
    for (int j = 0; j <= N; ++j) {
        const double *xn = x + (size_t)j * 6;
        for (int i = 0; i < 6; ++i) if (!isfinite(xn[i])) st_ok = 0;
        if (!(sqrt(xn[2] * xn[2] + xn[3] * xn[3]) <= p.v_lim)) st_ok = 0;
        if (!(fabs(xn[5]) <= p.om_lim)) st_ok = 0;
        long long ix = cell(xn[0], p.ox, p.res), iy = cell(xn[1], p.oy, p.res);
        int inside = ix >= 0 && ix < p.W && iy >= 0 && iy < p.H;
        long long jx = ix < 0 ? 0 : (ix > p.W - 1 ? p.W - 1 : ix);
        long long jy = iy < 0 ? 0 : (iy > p.H - 1 ? p.H - 1 : iy);
        double c = inside ? clear_tab[jy * p.W + jx] : 0.0;
        cl[(size_t)k * (N + 1) + j] = c;
        if (!(c >= p.r_safe)) safe = 0;
        double ex = xn[0] - p.gx, ey = xn[1] - p.gy;
        double eu = sqrt(ex * ex + ey * ey);
        double dg = eu;
        if (p.use_geo) {
            double f = field[jy * p.W + jx];
            dg = (!inside || !isfinite(f)) ? p.penalty + eu : f;
        }
        dgeo[(size_t)k * (N + 1) + j] = dg;
    }
    flags[k] = (unsigned char)(in_ok | (st_ok << 1) | (safe << 2));
}

__device__ static inline int ray_class(const signed char *classes, int H, int W, double res,
                                       double ox, double oy, double xs, double ys) {
    long long ix = cell(xs, ox, res), iy = cell(ys, oy, res);
    int inb = ix >= 0 && ix < W && iy >= 0 && iy < H;
    if (!inb) return 1;   // OCCUPIED
    return classes[iy * W + ix];
}

__global__ void frontier_kernel(int K, int N, double gx, double gy, int m, const double *t,
                                int H, int W, double res, double ox, double oy,
                                const signed char *classes, const double *X, int *out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const double *xn = X + ((size_t)k * (N + 1) + N) * 6;
    double px = xn[0], py = xn[1];
    double dx = gx - px, dy = gy - py;
    int c = ray_class(classes, H, W, res, ox, oy, px + t[0] * dx, py + t[0] * dy);
    if (c == 0) {
        for (int i = 0; i < m; ++i) {
            c = ray_class(classes, H, W, res, ox, oy, px + t[i] * dx, py + t[i] * dy);
            if (c != 0) break;
        }
    }
    out[k] = c;
}

extern "C" {

int mppi_eval(const MppiParams *p, const double *uin, const double *clear_tab, const double *field,
              double *uout, double *X, unsigned char *flags, double *dgeo, double *cl) {
    size_t nu = (size_t)p->K * p->N * 3 * sizeof(double);
    size_t nx = (size_t)p->K * (p->N + 1) * 6 * sizeof(double);
    size_t nn = (size_t)p->K * (p->N + 1) * sizeof(double);
    size_t nm = (size_t)p->H * p->W * sizeof(double);
    if (grow((void **)&d_uin, &cap_u, nu) || grow((void **)&d_uout, &cap_uo, nu)
        || grow((void **)&d_x, &cap_x, nx)) return -1;
    if (nn > cap_n1) {
        if (d_dgeo) cudaFree(d_dgeo);
        if (d_cl) cudaFree(d_cl);
        d_dgeo = d_cl = 0; cap_n1 = 0;
        if (cudaMalloc((void **)&d_dgeo, nn) != cudaSuccess || cudaMalloc((void **)&d_cl, nn) != cudaSuccess) return -1;
        cap_n1 = nn;
    }
    if ((size_t)p->K > cap_k) {
        if (d_flags) cudaFree(d_flags);
        if (d_cls) cudaFree(d_cls);
        d_flags = 0; d_cls = 0; cap_k = 0;
        if (cudaMalloc((void **)&d_flags, p->K) != cudaSuccess || cudaMalloc((void **)&d_cls, p->K * sizeof(int)) != cudaSuccess) return -1;
        cap_k = p->K;
    }
    if (nm > cap_map) {
        if (d_clear) cudaFree(d_clear);
        if (d_field) cudaFree(d_field);
        if (d_classes) cudaFree(d_classes);
        d_clear = d_field = 0; d_classes = 0; cap_map = 0;
        if (cudaMalloc((void **)&d_clear, nm) != cudaSuccess || cudaMalloc((void **)&d_field, nm) != cudaSuccess
            || cudaMalloc((void **)&d_classes, (size_t)p->H * p->W) != cudaSuccess) return -1;
        cap_map = nm;
    }
    if (cudaMemcpy(d_uin, uin, nu, cudaMemcpyHostToDevice) != cudaSuccess) return -2;
    if (cudaMemcpy(d_clear, clear_tab, nm, cudaMemcpyHostToDevice) != cudaSuccess) return -2;
    if (p->use_geo && cudaMemcpy(d_field, field, nm, cudaMemcpyHostToDevice) != cudaSuccess) return -2;
    int threads = 32, blocks = (p->K + threads - 1) / threads;   // 6 blocks at K=192: every SM
    eval_kernel<<<blocks, threads>>>(*p, d_uin, d_uout, d_x, d_flags, d_dgeo, d_cl, d_clear, d_field);
    if (cudaGetLastError() != cudaSuccess) return -3;
    if (cudaMemcpy(uout, d_uout, nu, cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    if (cudaMemcpy(X, d_x, nx, cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    if (cudaMemcpy(flags, d_flags, p->K, cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    if (cudaMemcpy(dgeo, d_dgeo, nn, cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    if (cudaMemcpy(cl, d_cl, nn, cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    last_K = p->K; last_N = p->N;
    return 0;
}

// Terminal-node ray to the goal over the X of the last mppi_eval call.
int mppi_frontier(double gx, double gy, int m, const double *t, int H, int W, double res,
                  double ox, double oy, const signed char *classes, int *out) {
    if (last_K <= 0) return -1;
    if ((size_t)m * sizeof(double) > cap_t) {
        if (d_t) cudaFree(d_t);
        d_t = 0; cap_t = 0;
        if (cudaMalloc((void **)&d_t, (size_t)m * sizeof(double)) != cudaSuccess) return -1;
        cap_t = (size_t)m * sizeof(double);
    }
    if ((size_t)H * W * sizeof(double) > cap_map) return -1;
    if (cudaMemcpy(d_t, t, (size_t)m * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return -2;
    if (cudaMemcpy(d_classes, classes, (size_t)H * W, cudaMemcpyHostToDevice) != cudaSuccess) return -2;
    int threads = 32, blocks = (last_K + threads - 1) / threads;
    frontier_kernel<<<blocks, threads>>>(last_K, last_N, gx, gy, m, d_t, H, W, res, ox, oy, d_classes, d_x, d_cls);
    if (cudaGetLastError() != cudaSuccess) return -3;
    if (cudaMemcpy(out, d_cls, last_K * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return -4;
    return 0;
}

int mppi_params_size(void) { return (int)sizeof(MppiParams); }

// Create the CUDA context (and load this module) before the first plan: on
// Xavier this takes seconds, which the first 10 Hz tick cannot absorb.
int mppi_warmup(void) { return cudaFree(0) == cudaSuccess ? 0 : -1; }

}
