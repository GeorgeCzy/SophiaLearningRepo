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


# ----------------------------
# Safety configuration
# ----------------------------

GLOBAL_SCALE = 0.35   # extra safety: shrink all motions

# Very conservative default limits (radians).
# You SHOULD refine these based on robot documentation / calibration.
DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
    # Left arm
    "LeftShoulderPitch": (-0.25, 0.25),
    "LeftShoulderRoll":  (-0.25, 0.25),
    "LeftShoulderYaw":   (-0.25, 0.25),
    "LeftElbowPitch":    (-0.40, 0.40),
    "LeftElbowYaw":      (-0.30, 0.30),

    # Right arm
    "RightShoulderPitch": (-0.25, 0.25),
    "RightShoulderRoll":  (-0.25, 0.25),
    "RightShoulderYaw":   (-0.25, 0.25),
    "RightElbowPitch":    (-0.40, 0.40),
    "RightElbowYaw":      (-0.30, 0.30),

    # Hands/fingers (conservative)
    "LeftIndexFinger":  (-0.30, 0.05),
    "LeftMiddleFinger": (-0.30, 0.05),
    "LeftRingFinger":   (-0.30, 0.05),
    "LeftPinkyFinger":  (-0.30, 0.05),
    "LeftThumbFinger":  (-0.30, 0.30),
    "LeftThumbRoll":    (-0.20, 0.20),

    "RightIndexFinger":  (-0.30, 0.05),
    "RightMiddleFinger": (-0.30, 0.05),
    "RightRingFinger":   (-0.30, 0.05),
    "RightPinkyFinger":  (-0.30, 0.05),
    "RightThumbFinger":  (-0.30, 0.30),
    "RightThumbRoll":    (-0.20, 0.20),
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
        while not rospy.is_shutdown():
            conn, addr = self.sock.accept()
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

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

                v = float(c.extractor(value)) * GLOBAL_SCALE
                v = clamp(c.actuator, v)
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
