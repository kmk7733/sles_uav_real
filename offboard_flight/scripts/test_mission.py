#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed-loop test of mission_node.py with no ROS and no aircraft.

Stubs rospy and the message types, then flies a first-order vehicle model that
chases whatever setpoint the node publishes. The point is to run the states
that only happen at the END of a flight -- LAND, DISARM, DONE -- before they
run on a real aircraft, and to pin the guards (planner silence, RC takeover,
geofence, non-finite setpoints) that are hard to provoke on the bench.

    python3 test_mission.py
"""

import math
import sys
import types

# --------------------------------------------------------------- fake clock

class Clock(object):
    t = 0.0

    @classmethod
    def now(cls):
        return cls.t


# ------------------------------------------------------------- fake messages

class _Vec(object):
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Quat(object):
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class _Header(object):
    def __init__(self):
        self.stamp = 0.0
        self.frame_id = ""


class Pose(object):
    def __init__(self):
        self.position = _Vec()
        self.orientation = _Quat()


class PoseStamped(object):
    def __init__(self):
        self.header = _Header()
        self.pose = Pose()


class PositionTarget(object):
    IGNORE_PX = 1
    IGNORE_PY = 2
    IGNORE_PZ = 4
    IGNORE_VX = 8
    IGNORE_VY = 16
    IGNORE_VZ = 32
    IGNORE_AFX = 64
    IGNORE_AFY = 128
    IGNORE_AFZ = 256
    FORCE = 512
    IGNORE_YAW = 1024
    IGNORE_YAW_RATE = 2048

    def __init__(self):
        self.header = _Header()
        self.coordinate_frame = 1
        self.position = _Vec()
        self.velocity = _Vec()
        self.acceleration_or_force = _Vec()
        self.type_mask = 0
        self.yaw = 0.0
        self.yaw_rate = 0.0


class State(object):
    def __init__(self):
        self.connected = True
        self.armed = False
        self.mode = "POSCTL"


class ExtendedState(object):
    LANDED_STATE_UNDEFINED = 0
    LANDED_STATE_ON_GROUND = 1
    LANDED_STATE_IN_AIR = 2

    def __init__(self):
        self.landed_state = 0


class Bool(object):
    def __init__(self, data=False):
        self.data = data


class String(object):
    def __init__(self, data=""):
        self.data = data


class TriggerResponse(object):
    def __init__(self, success=False, message=""):
        self.success = success
        self.message = message


# ----------------------------------------------------------------- fake rospy

LOG = []


class _Rate(object):
    def __init__(self, hz):
        self.dt = 1.0 / hz

    def sleep(self):
        Clock.t += self.dt
        HARNESS.step(self.dt)


class _Pub(object):
    def __init__(self, name):
        self.name = name
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)
        HARNESS.on_publish(self.name, msg)


class _Sub(object):
    def __init__(self, topic, cb):
        HARNESS.subs.setdefault(topic, []).append(cb)


class _Srv(object):
    def __init__(self, name, handler):
        HARNESS.services[name] = handler


class ServiceException(Exception):
    pass


def _fmt(msg, args):
    try:
        return msg % args if args else msg
    except Exception:
        return msg


def _log(level):
    def f(msg, *args):
        LOG.append((level, _fmt(msg, args)))
    return f


def _log_throttle(level):
    def f(_period, msg, *args):
        LOG.append((level, _fmt(msg, args)))
    return f


rospy = types.ModuleType("rospy")
rospy.init_node = lambda *a, **k: None
rospy.get_param = lambda name, default=None: HARNESS.params.get(name, default)
rospy.Subscriber = lambda topic, typ, cb, **k: _Sub(topic, cb)
rospy.Publisher = lambda topic, typ, **k: _Pub(topic)
rospy.Service = lambda name, typ, handler: _Srv(name, handler)
rospy.ServiceProxy = lambda name, typ: HARNESS.proxy(name)
rospy.ServiceException = ServiceException
rospy.Rate = _Rate
rospy.get_time = Clock.now
rospy.get_name = lambda: "/rogx2/mission_node"
rospy.is_shutdown = lambda: HARNESS.shutdown
rospy.Time = types.SimpleNamespace(now=Clock.now)
rospy.Duration = lambda s: s
for _lvl in ("loginfo", "logwarn", "logerr", "logfatal", "logdebug"):
    setattr(rospy, _lvl, _log(_lvl))
    setattr(rospy, _lvl + "_throttle", _log_throttle(_lvl))
sys.modules["rospy"] = rospy


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_mod("geometry_msgs")
_mod("geometry_msgs.msg", PoseStamped=PoseStamped, Pose=Pose)
_mod("mavros_msgs")
_mod("mavros_msgs.msg", PositionTarget=PositionTarget, State=State,
     ExtendedState=ExtendedState)
_mod("mavros_msgs.srv", CommandBool=object, SetMode=object)
_mod("std_msgs")
_mod("std_msgs.msg", Bool=Bool, String=String)
_mod("std_srvs")
_mod("std_srvs.srv", Trigger=object, TriggerResponse=TriggerResponse)
_mod("pyquaternion", Quaternion=object)


class _Controller(object):
    """The two constructors mission_node uses, same masks as guidance_library."""

    takeoff_height = 1.0
    takeoff_speed = 0.3

    def construct_target(self, x, y, z, yaw):
        t = PositionTarget()
        t.header.stamp = Clock.now()
        t.position.x, t.position.y, t.position.z = x, y, z
        t.type_mask = (PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY
                       + PositionTarget.IGNORE_VZ + PositionTarget.IGNORE_AFX
                       + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                       + PositionTarget.FORCE
                       + PositionTarget.IGNORE_YAW_RATE)
        t.yaw = yaw
        return t

    def construct_target_velocity(self, vx, vy, vz, yaw):
        t = PositionTarget()
        t.header.stamp = Clock.now()
        t.velocity.x, t.velocity.y, t.velocity.z = vx, vy, vz
        t.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY
                       + PositionTarget.IGNORE_PZ + PositionTarget.IGNORE_AFX
                       + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                       + PositionTarget.FORCE
                       + PositionTarget.IGNORE_YAW_RATE)
        t.yaw = yaw
        return t


_mod("guidance_library", Controller=_Controller)


# ------------------------------------------------------------------- harness

class Harness(object):
    """A vehicle that chases setpoints, plus a planner that talks to the node."""

    def __init__(self):
        self.reset()

    def reset(self, **params):
        Clock.t = 0.0
        del LOG[:]
        self.params = {"~hover_settle": 1.0, "~mission_timeout": 30.0}
        self.params.update(params)
        self.subs = {}
        self.services = {}
        self.shutdown = False
        self.published = []

        self.x, self.y, self.z, self.yaw = -2.9, 0.32, 0.0, 0.0
        self.z_ground = 0.0
        self.armed = False
        self.mode = "POSCTL"
        self.landed = ExtendedState.LANDED_STATE_ON_GROUND
        self.pilot_takeover_at = None
        self.planner_alive = True
        self.planner_goal = (2.0, 0.0)
        self.planner_z = 1.14        # deliberately NOT the takeoff height
        self.arrived = False
        self.states = []
        self.tick = 0

    # ---- services the node calls
    def proxy(self, name):
        if "arming" in name:
            def arm(v):
                # A landed vehicle refuses to disarm only if still flying.
                if not v and self.z - self.z_ground > 0.25:
                    return types.SimpleNamespace(success=False)
                self.armed = bool(v)
                return types.SimpleNamespace(success=True)
            return arm

        def set_mode(custom_mode=""):
            self.mode = custom_mode
            return types.SimpleNamespace(mode_sent=True)
        return set_mode

    def call(self, srv):
        return self.services["~" + srv](None)

    # ---- what the node publishes
    def on_publish(self, topic, msg):
        if topic == "mavros/setpoint_raw/local":
            self.published.append((Clock.now(), msg))

    # ---- one control period
    def step(self, dt):
        self.tick += 1
        if self.pilot_takeover_at is not None and Clock.t >= \
                self.pilot_takeover_at:
            self.mode = "POSCTL"

        # vehicle: track the last setpoint, first order, only while armed
        if self.published and self.armed and self.mode == "OFFBOARD":
            sp = self.published[-1][1]
            if sp.type_mask & PositionTarget.IGNORE_PZ:
                self.z += sp.velocity.z * dt
                self.x += sp.velocity.x * dt
                self.y += sp.velocity.y * dt
            else:
                tau = 0.25
                a = min(dt / tau, 1.0)
                self.x += a * (sp.position.x - self.x)
                self.y += a * (sp.position.y - self.y)
                self.z += a * (sp.position.z - self.z)
        self.z = max(self.z, self.z_ground)
        self.landed = (ExtendedState.LANDED_STATE_ON_GROUND
                       if self.z - self.z_ground < 0.05
                       else ExtendedState.LANDED_STATE_IN_AIR)

        self._pub_pose()
        self._pub_state()
        self._pub_planner()

    def _emit(self, topic, msg):
        for cb in self.subs.get(topic, []):
            cb(msg)

    def _pub_pose(self):
        p = PoseStamped()
        p.header.stamp = Clock.now()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = \
            self.x, self.y, self.z
        p.pose.orientation.w = math.cos(self.yaw / 2.0)
        p.pose.orientation.z = math.sin(self.yaw / 2.0)
        self._emit("mavros/local_position/pose", p)

    def _pub_state(self):
        s = State()
        s.connected, s.armed, s.mode = True, self.armed, self.mode
        self._emit("mavros/state", s)
        e = ExtendedState()
        e.landed_state = self.landed
        self._emit("mavros/extended_state", e)

    def _pub_planner(self):
        if not self.planner_alive:
            return
        gx, gy = self.planner_goal
        d = math.hypot(gx - self.x, gy - self.y)
        self.arrived = self.arrived or d < 0.25
        self._emit("/goal_arrive_tf", Bool(self.arrived))
        # a step toward the goal, like the planner's own bounded reference
        step = min(0.35, d)
        t = PositionTarget()
        t.header.stamp = Clock.now()
        if d > 1e-6:
            t.position.x = self.x + (gx - self.x) / d * step
            t.position.y = self.y + (gy - self.y) / d * step
        else:
            t.position.x, t.position.y = gx, gy
        t.position.z = self.planner_z
        self._emit("commander/set_pose", t)


HARNESS = Harness()

import mission_node                                            # noqa: E402


def fly(max_seconds=120.0, **params):
    HARNESS.reset(**params)
    node = mission_node.MissionNode()
    orig = node._enter

    def traced(state, why=""):
        if state != node.state:
            HARNESS.states.append(state)
        orig(state, why)
    node._enter = traced
    HARNESS.node = node
    r = HARNESS.call("start")
    assert r.success, r.message

    def is_shutdown():
        return HARNESS.shutdown or Clock.t > max_seconds
    rospy.is_shutdown = is_shutdown
    node.run()
    return node


# ---------------------------------------------------------------------- tests

CHECKS = [0, 0]


def check(ok, what):
    CHECKS[0] += 1
    if ok:
        CHECKS[1] += 1
        print("  ok   %s" % what)
    else:
        print("  FAIL %s" % what)


def logged(fragment):
    return any(fragment in m for _, m in LOG)


print("NOMINAL MISSION -- start, takeoff, hover, fly to goal, land, disarm")
node = fly(max_seconds=90.0)
seq = HARNESS.states
print("       states: %s" % " -> ".join(seq))
check("STREAM" in seq and "TAKEOFF" in seq, "streams then takes off")
check(seq.index("TAKEOFF") < seq.index("HOVER") < seq.index("MISSION"),
      "TAKEOFF before HOVER before MISSION")
check("LAND" in seq and "DISARM" in seq and seq[-1] == "DONE",
      "lands, disarms, ends in DONE")
check("HOLD" not in seq, "no HOLD in the nominal run")
check(not HARNESS.armed, "vehicle is disarmed at the end")
check(abs(HARNESS.z - HARNESS.z_ground) < 0.05, "vehicle is on the ground")
check(abs(HARNESS.x - 2.0) < 0.4 and abs(HARNESS.y) < 0.4,
      "landed at the goal (%.2f, %.2f)" % (HARNESS.x, HARNESS.y))

# The whole reason for the auto-start default: nothing may arm on its own.
print("\nNO START CALL -- nothing arms")
HARNESS.reset()
n2 = mission_node.MissionNode()
rospy.is_shutdown = lambda: Clock.t > 5.0
n2.run()
check(not HARNESS.armed, "still disarmed after 5 s")
check(n2.state == "WAIT", "still in WAIT")
check(len(HARNESS.published) == 0, "published no setpoints")

print("\nARM/OFFBOARD ARE READ FROM /mavros/state")
# The FCU refuses OFFBOARD; the node must not walk on to TAKEOFF regardless of
# what the service call returned.
HARNESS.reset()
n3 = mission_node.MissionNode()
HARNESS.node = n3
real_proxy = HARNESS.proxy


def stubborn(name):
    if "arming" in name:
        return real_proxy(name)

    def set_mode(custom_mode=""):
        return types.SimpleNamespace(mode_sent=True)     # accepted, ignored
    return set_mode


HARNESS.proxy = stubborn
n3.srv_mode = stubborn("mavros/set_mode")
rospy.is_shutdown = lambda: Clock.t > 10.0
n3.start_req = True
n3.run()
check(n3.state == "STREAM", "stays in STREAM while OFFBOARD is refused")
check(not HARNESS.armed, "never armed")
check(len(HARNESS.published) > 100, "kept the setpoint stream alive anyway")
HARNESS.proxy = real_proxy

print("\nPLANNER GOES SILENT MID-MISSION -> HOLD, and it does not land")
HARNESS.reset()
n4 = mission_node.MissionNode()
HARNESS.node = n4
n4.start_req = True
_step = HARNESS.step


def step_then_silence(dt):
    _step(dt)
    if HARNESS.node.state == "MISSION" and HARNESS.node.elapsed > 1.0:
        HARNESS.planner_alive = False


HARNESS.step = step_then_silence
rospy.is_shutdown = lambda: Clock.t > 40.0
n4.run()
HARNESS.step = _step
check(n4.state == "HOLD", "ends in HOLD (state=%s)" % n4.state)
check(HARNESS.armed, "still armed -- silence is not a reason to land")
check(logged("planner silent"), "said why")

print("\nRC TAKEOVER -> PILOT, and the setpoint stream stops")
HARNESS.reset()
n5 = mission_node.MissionNode()
HARNESS.node = n5
n5.start_req = True
HARNESS.pilot_takeover_at = 9.0
rospy.is_shutdown = lambda: Clock.t > 25.0
n5.run()
check(n5.state == "PILOT", "ends in PILOT (state=%s)" % n5.state)
n_before = len([1 for t, _ in HARNESS.published if t < 9.0])
n_after = len([1 for t, _ in HARNESS.published if t > 11.0])
check(n_before > 0 and n_after == 0, "published nothing after the takeover")

print("\nGEOFENCE -> HOLD")
HARNESS.reset()
n6 = mission_node.MissionNode(); HARNESS.node = n6
n6.start_req = True
n6.fence_r = 0.5
rospy.is_shutdown = lambda: Clock.t > 40.0
n6.run()
check(n6.state == "HOLD", "ends in HOLD (state=%s)" % n6.state)
check(logged("FENCE"), "said FENCE")

print("\nNON-FINITE SETPOINT IS DROPPED")
HARNESS.reset()
n7 = mission_node.MissionNode(); HARNESS.node = n7
n7.start_req = True
_pp = HARNESS._pub_planner


def poison():
    if HARNESS.node.state == "MISSION" and HARNESS.node.elapsed > 1.0:
        t = PositionTarget()
        t.position.x = float("nan")
        t.position.z = 1.0
        HARNESS._emit("commander/set_pose", t)
        HARNESS._emit("/goal_arrive_tf", Bool(False))
    else:
        _pp()


HARNESS._pub_planner = poison
rospy.is_shutdown = lambda: Clock.t > 30.0
n7.run()
HARNESS._pub_planner = _pp
check(n7.state == "HOLD", "ends in HOLD (state=%s)" % n7.state)
check(logged("non-finite"), "said non-finite")
check(all(not any(v != v for v in (m.position.x, m.position.y, m.position.z))
          for _, m in HARNESS.published), "no NaN ever reached mavros")

print("\nLANDING THAT DOES NOT CONFIRM -> HOLD, never a disarm in the air")
HARNESS.reset()
n8 = mission_node.MissionNode(); HARNESS.node = n8
n8.start_req = True
n8.land_timeout = 3.0


def never_lands(dt):
    _step(dt)
    HARNESS.z = max(HARNESS.z, 1.0)          # the vehicle refuses to descend
    HARNESS.landed = ExtendedState.LANDED_STATE_IN_AIR


rospy.is_shutdown = lambda: Clock.t > 60.0
_orig_begin = n8._begin_land


def begin(pose, why):
    HARNESS.step = never_lands
    _orig_begin(pose, why)


n8._begin_land = begin
n8.run()
HARNESS.step = _step
check(n8.state == "HOLD", "ends in HOLD (state=%s)" % n8.state)
check(HARNESS.armed, "still armed -- did NOT disarm in the air")
check(logged("no touchdown"), "said no touchdown")

print("\nVELOCITY LANDING (path_generation.py's profile) still lands")
HARNESS.reset()
n9 = mission_node.MissionNode(); HARNESS.node = n9
n9.start_req = True
n9.land_mode = "velocity"
rospy.is_shutdown = lambda: Clock.t > 90.0
n9.run()
check(n9.state == "DONE", "ends in DONE (state=%s)" % n9.state)
check(not HARNESS.armed, "disarmed")

print("\nHOVER HANDS THE ALTITUDE OVER TO THE PLANNER")
HARNESS.reset()
n10 = mission_node.MissionNode(); HARNESS.node = n10
n10.start_req = True
rospy.is_shutdown = lambda: Clock.t > 90.0
n10.run()
check(abs(n10.hover_z - HARNESS.planner_z) < 0.05,
      "hover_z %.3f slewed onto the planner's %.2f" % (n10.hover_z,
                                                       HARNESS.planner_z))

print("\n%d/%d checks passed" % (CHECKS[1], CHECKS[0]))
sys.exit(0 if CHECKS[1] == CHECKS[0] else 1)
