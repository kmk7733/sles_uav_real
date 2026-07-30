"""Single source of truth for the real-drone closed loop.

This is the exact plant + geometric tracker the HPA demos were distilled in
(see collect_demos.py). run_hpa / run_haa / run_switch all build their plant and
tracker from here so the three configs can never silently drift apart again --
the BC policy is only valid when replayed in the loop it was trained against, and
the HAA tube rho is only valid for these tracker gains.

Real DJI-class drone: 0.62 x 0.50 m -> body radius 0.31 m, mass 3.2 kg,
heading-rate limit 30 deg/s.
"""
MASS = 3.2          # [kg]
TAU_RATE = 0.10     # [s] first-order body-rate lag
DRAG = 0.1

# Geometric tracker gains (the closed loop the tube rho is derived for).
TRK_KP, TRK_KV, TRK_KATT = 2.0, 2.0, 6.0
TRK_WMAX = 1.5            # [rad/s] ||omega|| cap
YAW_RATE_MAX = 0.5236    # [rad/s] hard yaw-rate cap = 30 deg/s

BODY_RADIUS = 0.31  # [m] true body radius

# --- Tube (ISS) disturbance bound, single source of truth for HAA + switch ----
# The tube rho = TUBE_GAMMA * TUBE_WMAX / TRK_KP + TUBE_DELTA_MODEL bounds the
# closed-loop planar tracking error ||p_true - p_plan||; HAA tightens obstacles by
# it (X (-) Z) and the switch's recoverable/margin gate uses it. Both run_haa and
# run_switch default to these so the tube can never drift between the two.
#
# Calibrated 2026-06-29 against the realized tracking error: across the 16-scenario
# suite (centered + randomized starts, disturbance-free) max ||p_true - p_plan|| was
# ~0.078 m. w_max=0.20 -> rho = 1.1*0.20/2.0 = 0.110 m gives a ~1.4x margin over
# that error while keeping headroom for wind / depth-noise disturbances. The prior
# w_max=0.30 -> rho=0.165 was ~2.2x over-conservative: it inflated the planning
# radius to 0.475 m (vs 0.42 m now), closing the 1.0 m gaps (tight_door / double_door)
# in both the collision check AND the geodesic cost-to-go, so HAA timed out rather
# than threading them.
#
# Verified 2026-07-06 with the disturbance + noise now available (mppi_quad.verify_tube):
#   * clean perception + OU disturbance at the design bound w_max=0.20:  worst realized
#     ||p_true - p_plan|| = 0.075 m over the 8-scenario x 2-seed suite  <=  rho=0.110 m.
#   * noisy SGBM perception (delta_model += PERC_SLACK -> rho=0.170 m) + same disturbance:
#     worst tracking error 0.075 m  <=  rho=0.170 m (margin +0.095 m).
# So the DYNAMICS tube holds under the design-bound disturbance in both modes. The
# feedback tracker rejects the OU accel well, so w_k barely moves the tracking error;
# rho is set by the 6D->12D model-reduction error, not the disturbance. Separately,
# noisy perception DOES induce occasional HAA-only collisions (clean: 0/16, noisy: ~4/16
# on randomized start/goal) because the occupancy map is imperfect -- a perception, not
# a tracking-tube, failure. That is the r_perc / HAP-robustness regime (raise PERC_SLACK,
# lower p_occ, or use the conservative unknown=occupied view), not a reason to grow rho.
TUBE_WMAX = 0.20         # [m/s^2] disturbance accel bound (radius of W, planar)
TUBE_GAMMA = 1.1         # transient/percentile factor (>= 1)
TUBE_DELTA_MODEL = 0.0   # [m] additive slack for perception / unmodeled error (clean)

# --- Perception inflation r_perc (paper: r_eff = r_Q + r_track + r_perc) -------
# With clean ray-cast depth there is no stereo error, so r_perc = 0 (the tube rho
# already covers r_track). When the SGBM stereo front-end is enabled
# (mppi_quad.stereo.NoiseConfig.enabled), obstacle boundaries are corrupted by
# matching error, invalid disparities, and pose/angular uncertainty; PERC_SLACK
# is the extra obstacle inflation that buys back that error. It was sized from
# the co-valid stereo-vs-truth depth error on the CORE scene (p95 ~0.25 m lateral
# near boundaries, but the §VII occupancy blur already absorbs most of it) plus a
# margin. Pass it as `delta_model=tube_delta_model(noise_on)` when building rho so
# HAA tightens X (-) Z by r_track (rho) + r_perc together. Re-derive if the
# NoiseConfig severity or sgbm params change.
PERC_SLACK = 0.06        # [m] r_perc added to the tube when stereo noise is on


def tube_delta_model(noise_on=False):
    """Perception slack for tube_radius(delta_model=...): PERC_SLACK when the
    noisy stereo front-end is active, else the clean TUBE_DELTA_MODEL."""
    return (TUBE_DELTA_MODEL + PERC_SLACK) if noise_on else TUBE_DELTA_MODEL


def make_plant():
    from .quadrotor import Quadrotor
    return Quadrotor(mass=MASS, tau_rate=TAU_RATE, drag=DRAG)


def make_tracker():
    from .tracker import GeometricTracker
    return GeometricTracker(mass=MASS, kp=TRK_KP, kv=TRK_KV, k_att=TRK_KATT,
                            omega_max=TRK_WMAX, yaw_rate_max=YAW_RATE_MAX)
