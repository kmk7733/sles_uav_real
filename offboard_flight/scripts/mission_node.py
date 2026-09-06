#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm -> climb to 1 m -> fly the planner's setpoints -> land -> disarm.

    commander/set_pose (planner, FCU frame) --> [MISSION only] --> mavros

Replaces setpoint_buffer.py, which owned the takeoff AND relayed the planner
for the rest of the flight. Both cannot be the same node: the climb has to
finish before the planner may command anything, and the buffer could not tell
whether a planner was even alive. Two writers on mavros/setpoint_raw/local is
two people flying the aircraft, so startup refuses if the buffer is running.

STATES.  WAIT (idle until ~start) -> STREAM (setpoints, then OFFBOARD + ARM) ->
CLIMB (ramp to the planner's altitude, settle) -> MISSION (forward the planner)
-> LAND -> DISARM -> DONE.  Failure sinks: HOLD (freeze here, from planner
silence / a bad setpoint / the fence / a landing that never confirmed) and
PILOT (the RC took OFFBOARD away -- stop publishing and stay stopped).

TRANSITIONS COME FROM /mavros/state. guidance_library's FlightModes.arm() and
.offboard() read `if self.armService(True): return True` -- a service response
object, truthy whether or not .success is set -- so both return True even when
the FCU refused. Nothing here believes them.

LANDING is path_generation.py's: our own constant-rate descent in OFFBOARD,
then disarm. Not AUTO.LAND, so the profile is one we chose. Three changes --
x/y held by position instead of a zero velocity command that drifts with
estimator bias; the descent distance measured from the ground height recorded
at arming rather than assumed; and touchdown confirmed from extended_state
before the disarm. Unconfirmed means HOLD: a disarm in the air is the one
thing this must never do on a guess.

    ROS_NAMESPACE=rogx2 python3 mission_node.py
    rosservice call /rogx2/mission_node/start        # nothing arms before this
    rosservice call /rogx2/mission_node/land         # bail out, any time
"""

import copy
import sys
import threading

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State, ExtendedState
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from guidance_library import Controller

ON_GROUND = ExtendedState.LANDED_STATE_ON_GROUND


def yaw_of(pose):
    q = pose.pose.orientation
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class MissionNode(object):

    def __init__(self):
        rospy.init_node("mission_node")
        self.ctl = Controller()
        self.lock = threading.Lock()
        P = rospy.get_param

        self.rate_hz = P("~pub_rate", 20.0)
        self.z_takeoff = P("~takeoff_height", 1.0)
        self.v_climb = P("~takeoff_speed", 0.3)
        self.z_tol = P("~hover_tol", 0.15)
        self.settle = P("~hover_settle", 3.0)

        self.sp_topic = P("~sp_topic", "commander/set_pose")
        self.sp_timeout = P("~sp_timeout", 0.5)
        self.arrive_hold = P("~arrive_hold", 1.0)
        self.mission_timeout = P("~mission_timeout", 120.0)
        self.land_on_arrive = P("~land_on_arrive", True)

        self.v_land = P("~landing_speed", 0.4)
        # How far BELOW the recorded ground the setpoint may go. PX4's land
        # detector needs the thrust to fall off, which happens only once the
        # setpoint pushes into the floor. Bounded: it is the one deliberately
        # infeasible command in this file.
        self.land_push = P("~land_push", 0.20)
        self.land_timeout = P("~land_timeout", 10.0)

        self.max_step = P("~max_step", 1.2)     # setpoint leash, metres
        self.fence_r = P("~fence_r", 6.0)       # from the arming point
        self.pose_timeout = P("~pose_timeout", 0.5)

        self.pose = self.mav = self.sp = None
        self.landed = None
        self.t_pose = self.t_sp = 0.0
        self.t_arrived = None
        self.armed_once = self.offboard_once = False
        self.start_req = P("~auto_start", False)
        self.land_req = False
        self.cmd = self.home = self.frozen = None
        self.z_want = self.z0 = 0.0
        self.t_level = self.t_touch = None
        self.state, self.t_state = "WAIT", rospy.get_time()

        rospy.Subscriber("mavros/local_position/pose", PoseStamped,
                         self._cb("pose", "t_pose"), queue_size=1)
        rospy.Subscriber("mavros/state", State, self._cb("mav"), queue_size=1)
        rospy.Subscriber("mavros/extended_state", ExtendedState,
                         self._cb_landed, queue_size=1)
        rospy.Subscriber(self.sp_topic, PositionTarget,
                         self._cb("sp", "t_sp"), queue_size=1)
        rospy.Subscriber(P("~arrived_topic", "/goal_arrive_tf"), Bool,
                         self._cb_arrived, queue_size=1)

        self.pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget,
                                   queue_size=10)
        self.pub_state = rospy.Publisher("~state", String, queue_size=1,
                                         latch=True)
        self.arming = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode = rospy.ServiceProxy("mavros/set_mode", SetMode)
        rospy.Service("~start", Trigger, self._srv("start_req", ("WAIT",)))
        rospy.Service("~land", Trigger,
                      self._srv("land_req", ("STREAM", "CLIMB", "MISSION",
                                             "HOLD")))
        rospy.loginfo("[mission] climb %.2f m @ %.2f m/s, land @ %.2f m/s",
                      self.z_takeoff, self.v_climb, self.v_land)

    # ---------------------------------------------------------------- inputs

    def _cb(self, field, stamp=None):
        def f(msg):
            with self.lock:
                setattr(self, field, msg)
                if stamp:
                    setattr(self, stamp, rospy.get_time())
        return f

    def _cb_landed(self, msg):
        with self.lock:
            self.landed = msg.landed_state

    def _cb_arrived(self, msg):
        # /goal_arrive_tf is LATCHED and the planner republishes True every
        # tick, so this arms on the flag being up rather than on its rising
        # edge -- an edge test would never re-arm after MISSION entry clears
        # it, and the vehicle would hover over the goal for ever.
        with self.lock:
            self.t_arrived = (self.t_arrived or rospy.get_time()) \
                if msg.data else None

    def _srv(self, flag, states):
        def f(_req):
            if self.state not in states:
                return TriggerResponse(False, "not allowed in %s" % self.state)
            setattr(self, flag, True)
            return TriggerResponse(True, "ok")
        return f

    # --------------------------------------------------------------- helpers

    def _enter(self, state, why=""):
        rospy.loginfo("[mission] %s -> %s %s", self.state, state, why)
        self.state, self.t_state = state, rospy.get_time()
        self.pub_state.publish(String(data=state))

    @property
    def elapsed(self):
        return rospy.get_time() - self.t_state

    def _every(self, hz):
        """True once every 1/hz seconds within the current state."""
        n = max(int(self.rate_hz / hz), 1)
        return int(self.elapsed * self.rate_hz) % n == 0

    def _send(self, x, y, z, yaw):
        self.cmd = self.ctl.construct_target(x, y, z, yaw)
        self.pub.publish(self.cmd)

    def _freeze(self, pose, why):
        """Stop where we are. Every failure that is not a landing ends here."""
        p = pose.pose.position
        self.frozen = (p.x, p.y, p.z, yaw_of(pose))
        self._enter("HOLD", why)

    def _forward(self, sp, pose):
        """Bound a planner setpoint, or None to refuse it."""
        p = sp.position
        if not all(np.isfinite([p.x, p.y, p.z])):
            rospy.logerr_throttle(1.0, "[mission] non-finite setpoint")
            return None
        # A COPY: self.sp still points at the subscriber's message, and the
        # planner can publish slower than this loop runs, so clamping in place
        # would re-clamp the same setpoint each tick and walk it back toward
        # the vehicle until it stalls.
        out = copy.deepcopy(sp)
        px, py = pose.pose.position.x, pose.pose.position.y
        d = float(np.hypot(p.x - px, p.y - py))
        if d > self.max_step:
            f = self.max_step / d
            rospy.logwarn_throttle(1.0, "[mission] setpoint %.2f m away", d)
            out.position.x, out.position.y = px + (p.x - px) * f, \
                py + (p.y - py) * f
        # The planner is fixed-altitude by construction, so a z that wanders is
        # a symptom rather than a command.
        out.position.z = float(np.clip(p.z, self.z_want - 0.5,
                                       self.z_want + 0.5))
        out.header.stamp = rospy.Time.now()
        return out

    # ------------------------------------------------------------------- run

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        self.z_want = self.z_takeoff
        self.pub_state.publish(String(data=self.state))

        while not rospy.is_shutdown():
            with self.lock:
                pose, mav, sp = self.pose, self.mav, self.sp
                pose_age = rospy.get_time() - self.t_pose
                sp_age = rospy.get_time() - self.t_sp
                arrived_for = (rospy.get_time() - self.t_arrived
                               if self.t_arrived else 0.0)
            if mav is not None and mav.armed:
                self.armed_once = True

            # Disarm is tested BEFORE the mode, because PX4 commonly leaves
            # OFFBOARD in the same breath as disarming and that would read as
            # an RC takeover on what is actually a completed landing. And it is
            # gated on having been ARMED, not on having been in OFFBOARD:
            # OFFBOARD is accepted before the arm request in STREAM.
            if mav is not None and self.state not in ("WAIT", "DONE", "PILOT"):
                if self.armed_once and not mav.armed:
                    self._enter("DONE", "-- disarmed")
                elif mav.mode == "OFFBOARD":
                    self.offboard_once = True
                elif self.offboard_once:
                    self._enter("PILOT", "-- RC took over (mode=%s)" % mav.mode)

            if self.state in ("DONE", "PILOT"):
                rospy.loginfo_throttle(10.0, "[mission] %s -- not publishing",
                                       self.state)
            elif pose is None or mav is None:
                rospy.logwarn_throttle(2.0, "[mission] waiting for mavros")
            elif pose_age > self.pose_timeout and self.state != "WAIT":
                # Keep the stream alive -- dropping it trips PX4's offboard
                # failsafe -- but integrate nothing new on a stale estimate.
                rospy.logerr_throttle(1.0, "[mission] pose stale %.2f s",
                                      pose_age)
                if self.cmd is not None:
                    self.cmd.header.stamp = rospy.Time.now()
                    self.pub.publish(self.cmd)
            else:
                if self.land_req and self.state in (
                        "STREAM", "CLIMB", "MISSION", "HOLD"):
                    self._begin_land(pose, "-- ~land requested")
                elif self.state in ("CLIMB", "MISSION") and \
                        np.hypot(pose.pose.position.x - self.home[0],
                                 pose.pose.position.y - self.home[1]) \
                        > self.fence_r:
                    self._freeze(pose, "-- geofence")
                getattr(self, "_" + self.state.lower())(pose, mav, sp, sp_age,
                                                        arrived_for)
            rate.sleep()

    # ----------------------------------------------------------- the states

    def _wait(self, pose, mav, sp, sp_age, _arr):
        if not mav.connected:
            rospy.logwarn_throttle(2.0, "[mission] no FCU link")
        elif not self.start_req:
            rospy.loginfo_throttle(5.0, "[mission] READY -- rosservice call "
                                        "%s/start", rospy.get_name())
        elif sp_age > self.sp_timeout:
            rospy.logwarn_throttle(2.0, "[mission] no planner on %s -- is it "
                                        "running with _dry_run:=false?",
                                   self.sp_topic)
        else:
            p = pose.pose.position
            self.home = (p.x, p.y, p.z, yaw_of(pose))
            rospy.loginfo("[mission] home (%.2f, %.2f, %.2f)", p.x, p.y, p.z)
            self._enter("STREAM")

    def _stream(self, pose, mav, sp, sp_age, _arr):
        self._send(*self.home)
        if self.elapsed < 1.0:
            return          # PX4 accepts OFFBOARD only once setpoints stream
        if mav.mode != "OFFBOARD":
            if self._every(1.0):
                self._request(self.set_mode, dict(custom_mode="OFFBOARD"),
                              "OFFBOARD")
        elif not mav.armed:
            if self._every(1.0):
                self._request(self.arming, (True,), "ARM")
        else:
            # The altitude target comes from the PLANNER, not from our own
            # takeoff_height. It publishes z0 in vicon/world through the
            # world->FCU alignment, and the Vicon floor is not the EKF2 origin
            # (~0.14 m apart here), so climbing to our own number would leave a
            # vertical step to take at MISSION entry.
            self.z_want = (float(sp.position.z) if sp_age <= self.sp_timeout
                           else self.home[2] + self.z_takeoff)
            self.z0 = pose.pose.position.z
            self._enter("CLIMB", "-- armed in OFFBOARD, to z=%.2f"
                        % self.z_want)

    def _request(self, srv, args, what):
        rospy.loginfo_throttle(2.0, "[mission] requesting %s", what)
        try:
            if isinstance(args, dict):
                srv(**args)
            else:
                srv(*args)
        except rospy.ServiceException as e:
            rospy.logerr_throttle(2.0, "[mission] %s: %s", what, e)

    def _climb(self, pose, mav, sp, sp_age, _arr):
        dt = abs(self.z_want - self.z0) / max(self.v_climb, 1e-3)
        f = 1.0 if dt < 1e-3 else min(self.elapsed / dt, 1.0)
        self._send(self.home[0], self.home[1],
                   self.z0 + f * (self.z_want - self.z0), self.home[3])

        if f < 1.0 or abs(pose.pose.position.z - self.z_want) > self.z_tol:
            self.t_level = None
            return
        self.t_level = self.t_level or rospy.get_time()
        if rospy.get_time() - self.t_level < self.settle:
            return
        if sp_age > self.sp_timeout:
            rospy.logwarn_throttle(2.0, "[mission] hovering, no planner")
            return
        with self.lock:
            self.t_arrived = None       # ignore a latch from before takeoff
        self._enter("MISSION", "-- settled at %.2f m" % self.z_want)

    def _mission(self, pose, mav, sp, sp_age, arrived_for):
        if sp_age > self.sp_timeout:
            return self._freeze(pose, "-- planner silent %.2f s" % sp_age)
        out = self._forward(sp, pose)
        if out is None:
            return self._freeze(pose, "-- unusable setpoint")
        self.cmd = out
        self.pub.publish(out)

        if arrived_for >= self.arrive_hold:
            if self.land_on_arrive:
                self._begin_land(pose, "-- GOAL REACHED")
            else:
                self._freeze(pose, "-- GOAL REACHED (~land_on_arrive false)")
        elif self.elapsed > self.mission_timeout:
            self._freeze(pose, "-- mission_timeout")

    def _begin_land(self, pose, why):
        self.land_req = False
        p = pose.pose.position
        self.frozen = (p.x, p.y, p.z, yaw_of(pose))
        self.t_touch = max(p.z - self.home[2], 0.0) / max(self.v_land, 1e-3)
        self._enter("LAND", "%s from z=%.2f, nominal %.1f s"
                    % (why, p.z, self.t_touch))

    def _land(self, pose, mav, sp, sp_age, _arr):
        x, y, z0, yaw = self.frozen
        self._send(x, y, max(z0 - self.v_land * self.elapsed,
                             self.home[2] - self.land_push), yaw)
        with self.lock:
            landed = self.landed
        if landed is None:
            rospy.logwarn_throttle(2.0, "[mission] no extended_state -- "
                                        "touchdown from height and clock")
        down = (landed == ON_GROUND) if landed is not None else (
            pose.pose.position.z - self.home[2] < 0.12
            and self.elapsed > self.t_touch)
        if down:
            self._enter("DISARM", "-- touchdown")
        elif self.elapsed > self.t_touch + self.land_timeout:
            rospy.logerr("[mission] no touchdown after %.1f s -- take it with "
                         "the RC", self.elapsed)
            self._freeze(pose, "-- landing did not confirm")

    def _disarm(self, pose, mav, sp, sp_age, _arr):
        # Keep the stream alive between attempts, or PX4 drops OFFBOARD and
        # hands the aircraft to a failsafe instead.
        self._send(self.frozen[0], self.frozen[1],
                   self.home[2] - self.land_push, self.frozen[3])
        if self._every(2.0):
            self._request(self.arming, (False,), "DISARM")
        if self.elapsed > 8.0:
            rospy.logerr_throttle(2.0, "[mission] still armed -- disarm on "
                                       "the RC")

    def _hold(self, pose, mav, sp, sp_age, _arr):
        self._send(*self.frozen)
        rospy.loginfo_throttle(5.0, "[mission] HOLD at (%.2f, %.2f) -- "
                                    "%s/land", self.frozen[0], self.frozen[1],
                               rospy.get_name())

    def _done(self, *a):
        pass

    def _pilot(self, *a):
        pass


def main():
    node = MissionNode()
    try:
        import rosnode
        clash = [n for n in rosnode.get_node_names() if "setpoint_buffer" in n]
    except Exception:
        clash = []
    if clash:
        rospy.logfatal("[mission] %s also publishes mavros/setpoint_raw/local"
                       " -- kill it first", ", ".join(clash))
        sys.exit(1)
    node.run()


if __name__ == "__main__":
    main()
