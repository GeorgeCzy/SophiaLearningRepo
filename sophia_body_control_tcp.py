#!/usr/bin/env python3
"""
TCP JSON -> HR ROS body actuators bridge (STRICT CLAMPING)

Client sends: {"index": <int>, "value": [x,y,z]}
Server replies: {"code": 0, "result": {...}} or {"code": nonzero, "error": "..."}
"""

import json
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any

import rospy
from hr_msgs.msg import TargetPosture
from hr_msgs.srv import SetActuatorsControl, SetActuatorsControlRequest
import math


# ----------------------------
# Safety configuration
# ----------------------------

GLOBAL_SCALE = 1.0   # extra safety: shrink all motions

def deg(x):
    return x * math.pi / 180.0

# Very conservative default limits (radians).
DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
    # =========================
    # RIGHT ARM
    # =========================
    "RightShoulderPitch": (deg(-145), deg(35)),
    "RightShoulderRoll":  (deg(-4),   deg(101)),
    "RightShoulderYaw":   (deg(-66),  deg(83)),
    "RightElbowPitch":    (deg(-127), deg(119)),
    "RightElbowYaw":      (deg(-123), deg(123)),

    # =========================
    # RIGHT HAND
    # =========================
    "RightWristPitch":   (deg(-35),  deg(35)),
    "RightWristRoll":    (deg(-35),  deg(35)),
    "RightThumbRoll":    (deg(-31),  deg(22)),
    "RightThumbFinger":  (deg(-75),  deg(44)),
    "RightIndexFinger":  (deg(-123), deg(18)),
    "RightMiddleFinger": (deg(-18),  deg(132)),
    "RightRingFinger":   (deg(-18),  deg(136)),
    "RightPinkyFinger":  (deg(-75),  deg(4)),
    # =========================
    # LEFT ARM
    # =========================
    "LeftShoulderPitch": (deg(-35),  deg(145)),
    "LeftShoulderRoll":  (deg(-101), deg(4)),
    "LeftShoulderYaw":   (deg(-83),  deg(66)),
    "LeftElbowPitch":    (deg(-119), deg(127)),
    "LeftElbowYaw":      (deg(-123), deg(123)),

    # =========================
    # LEFT HAND
    # =========================
    "LeftWristPitch":   (deg(-35),  deg(35)),
    "LeftWristRoll":    (deg(-35),  deg(35)),
    "LeftThumbRoll":    (deg(-31),  deg(22)),
    "LeftThumbFinger":  (deg(-44),  deg(75)),
    "LeftIndexFinger":  (deg(-18),  deg(123)),
    "LeftMiddleFinger": (deg(-18),  deg(132)),
    "LeftRingFinger":   (deg(-18),  deg(136)),
    "LeftPinkyFinger":  (deg(-4),   deg(75)),
}


SIGN: Dict[str, float] = { # if the moving direction doesnt match, set the para = 1.0, -1.0 is default
  "RightIndexFinger": -1.0,
  "RightMiddleFinger": 1.0,
  "RightRingFinger": 1.0,
  "RightPinkyFinger": -1.0,
  "LeftIndexFinger": -1.0,
  "LeftMidlleFinger": -1.0,
  "LeftRingFinger": -1.0,
  "LeftPinkyFinger": -1.0,
  "LeftElbowYaw": 1.0,
  "RightShoulderYaw": 1.0, 
  "RightShoulderPitch": 1.0,
  "LeftElbowPitch": 1.0,
  "RightElbowPitch": 1.0,
}


# If a command targets an actuator not in LIMITS -> reject (strict!)
LIMITS = DEFAULT_LIMITS


def clamp(actuator: str, v: float) -> float:
    lo, hi = LIMITS[actuator]
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

OFFSET = { #used for customize
        "RightShoulderRoll": deg(45.0), # used for A-pose setup
        "RightElbowPitch": deg(-127.0), # used for A-pose setup
        "LeftShoulderRoll": deg(-45.0), # used for A-pose setup
        "LeftElbowPitch": deg(127.0), # used for A-pose setup
        # if using Sophia's default pose(motors=0), comment out everything above
}

GAIN = { # used for matching the amount of displacement between web-end pose and real robot
    "RightShoulderPitch": 0.95,
    "LeftShoulderPitch": 0.95,
    "RightElbowPitch": 1.75,
    "LeftElbowPitch": 1.75,
    "LeftShoulderYaw": 3.0,
    "RightShoulderRoll": 1.5,
    "LeftShoulderRoll": 1.5,
}

def send_t_pose(pose_pub, limit, repeats=10, dt=0.1):
    """
    set all actuators to 0
    """
    from hr_msgs.msg import TargetPosture
    import rospy
    
    names = []
    values = []
    for name, (lo, hi) in limit.items():
        v = OFFSET.get(name, 0.0) * float(GLOBAL_SCALE)
        if v < lo:
          v = lo
        elif v > hi:
          v = hi
        names.append(name)
        values.append(v)
    msg = TargetPosture()
    msg.names = names
    msg.values = values
    for _ in range(repeats):
        pose_pub.publish(msg)
        rospy.sleep(dt)

# ----------------------------
# Mapping from SMPL "index" to robot actuators
# This is based on your smpl_visualizer UI meanings.
# ----------------------------

@dataclass(frozen=True)
class ActuatorCmd:
    actuator: str
    extractor: Callable[[List[float]], float]   # takes [x,y,z] -> scalar command


def _need_vec3(v: Any) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError("value must be a list/tuple of 3 floats: [x,y,z]")
    return [float(v[0]), float(v[1]), float(v[2])]


# IMPORTANT:
# Your visualizer sends axis-angle (x,y,z). We'll interpret components as follows:
# - For 16/17 (shoulder pitch & roll): use x as pitch, z as roll.
# - For 18/19 (shoulder yaw & elbow pitch): use x as yaw, y as elbow pitch.
# - For 20/21 (elbow yaw): your to_axisangle uses x-axis, so use x.
# - Fingers: your to_axisangle uses z (often negative), so use z.
#
# If directions feel inverted, flip sign here (safest place to adjust).

INDEX_MAP: Dict[int, List[ActuatorCmd]] = {
    # Left shoulder pitch & roll
    16: [
        ActuatorCmd("LeftShoulderPitch", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("LeftShoulderRoll",  lambda v: _need_vec3(v)[2]),
    ],
    # Right shoulder pitch & roll
    17: [
        ActuatorCmd("RightShoulderPitch", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("RightShoulderRoll",  lambda v: _need_vec3(v)[2]),
    ],
    # Left shoulder yaw & left elbow pitch
    18: [
        ActuatorCmd("LeftShoulderYaw",  lambda v: _need_vec3(v)[0]),
        ActuatorCmd("LeftElbowPitch",   lambda v: _need_vec3(v)[1]),
    ],
    # Right shoulder yaw & right elbow pitch
    19: [
        ActuatorCmd("RightShoulderYaw", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("RightElbowPitch",  lambda v: _need_vec3(v)[1]),
    ],
    # Left elbow yaw
    20: [
        ActuatorCmd("LeftElbowYaw", lambda v: _need_vec3(v)[0]),
    ],
    # Right elbow yaw (you used a slider, but we still receive vec3 after to_axisangle)
    21: [
        ActuatorCmd("RightElbowYaw", lambda v: _need_vec3(v)[0]),
    ],

    # Left hand fingers (your indices)
    25: [ActuatorCmd("LeftIndexFinger",  lambda v: _need_vec3(v)[2])],
    28: [ActuatorCmd("LeftMiddleFinger", lambda v: _need_vec3(v)[2])],
    31: [ActuatorCmd("LeftPinkyFinger",  lambda v: _need_vec3(v)[2])],
    34: [ActuatorCmd("LeftRingFinger",   lambda v: _need_vec3(v)[2])],

    # Left thumb roll & thumb finger
    37: [
        ActuatorCmd("LeftThumbRoll",   lambda v: _need_vec3(v)[0]),
        ActuatorCmd("LeftThumbFinger", lambda v: _need_vec3(v)[2]),
    ],

    # Right thumb roll & thumb finer
    52: [
        ActuatorCmd("RightThumbRoll",   lambda v: _need_vec3(v)[0]),
        ActuatorCmd("RightThumbFinger", lambda v: _need_vec3(v)[2]),
    ],

    # Right hand fingers
    40: [ActuatorCmd("RightIndexFinger",  lambda v: _need_vec3(v)[2])],
    43: [ActuatorCmd("RightMiddleFinger", lambda v: _need_vec3(v)[2])],
    46: [ActuatorCmd("RightPinkyFinger",  lambda v: _need_vec3(v)[2])],
    49: [ActuatorCmd("RightRingFinger",   lambda v: _need_vec3(v)[2])],
}


# ----------------------------
# Server implementation
# ----------------------------

class BodyBridgeServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5005):
        rospy.loginfo("[BodyBridge] waiting for /hr/actuators/set_control ...")
        rospy.wait_for_service("/hr/actuators/set_control")

        self.pose_pub = rospy.Publisher("/hr/actuators/pose", TargetPosture, queue_size=1)
        self.set_control = rospy.ServiceProxy("/hr/actuators/set_control", SetActuatorsControl)

        # Put all actuators we may touch into MANUAL mode
        self._set_manual_for_whitelist()

        # reset robot to defualt position
        send_t_pose(self.pose_pub, LIMITS)
        rospy.loginfo("[BodyBridge] sent startup T-pose")

        # TCP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        rospy.loginfo(f"[BodyBridge] listening on {host}:{port}")

    def _set_manual_for_whitelist(self):
        whitelist = sorted(LIMITS.keys())
        req = SetActuatorsControlRequest()
        req.control = SetActuatorsControlRequest.CONTROL_MANUAL
        req.actuators = whitelist
        self.set_control(req)
        rospy.loginfo(f"[BodyBridge] set MANUAL control for {len(whitelist)} actuators")

    def serve_forever(self):
      try:
        while not rospy.is_shutdown():
            conn, addr = self.sock.accept()
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
      except KeyboardInterrupt: rospy.loginfo("Ctrl+C received, shutting down")
      finally:
        try:
          self.sock.close()
        except Exception:
          pass

    def _handle(self, conn: socket.socket, addr):
        try:
            raw = conn.recv(4096)
            if not raw:
                return
            req = json.loads(raw.decode("utf-8"))

            if not isinstance(req, dict) or "index" not in req or "value" not in req:
                self._send(conn, code=1, error="request must be dict with keys: index, value")
                return

            idx = int(req["index"])
            value = req["value"]

            if idx not in INDEX_MAP:
                self._send(conn, code=2, error=f"index {idx} not allowed (no mapping)")
                return

            cmds = INDEX_MAP[idx]
            names: List[str] = []
            vals: List[float] = []

            for c in cmds:
                if c.actuator not in LIMITS:
                    self._send(conn, code=3, error=f"no limits for actuator {c.actuator}")
                    return

                delta = float(c.extractor(value)) * GLOBAL_SCALE
                delta = delta * float(SIGN.get(c.actuator, -1.0)) # used magic number to match the moving direction of fingers
                # reason for this might be some inconsistency in to_axisangle in smpl_visualizer.py
                delta *= float(GAIN.get(c.actuator, 1.0)) # magnify rotation for certain motors
                # v = clamp(c.actuator, v)
                base_bias = float(OFFSET.get(c.actuator, 0.0))
                v = clamp(c.actuator, base_bias + delta)
                names.append(c.actuator)
                vals.append(v)

            # Publish to robot
            msg = TargetPosture()
            msg.names = names
            msg.values = vals
            self.pose_pub.publish(msg)

            self._send(conn, code=0, result={"index": idx, "sent": dict(zip(names, vals))})

        except Exception as e:
            self._send(conn, code=99, error=str(e))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _send(self, conn: socket.socket, code: int, result=None, error: str = ""):
        resp = {"code": code}
        if code == 0:
            resp["result"] = result
        else:
            resp["error"] = error
        conn.sendall(json.dumps(resp).encode("utf-8"))


if __name__ == "__main__":
    rospy.init_node("sophia_body_bridge_server", anonymous=True)
    server = BodyBridgeServer(host="0.0.0.0", port=5005)
    server.serve_forever()
