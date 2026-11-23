#!/usr/bin/env python3
"""Continuous loop course: walk up a 5 deg ramp -> get pushed over on the platform
-> StandUp policy gets it back up -> walk down the steps -> loop around -> repeat.

Policies: walking = policies/walk_flat_10k.onnx (trained from scratch, 10k PPO iterations) + Pollen's
alpha_stand.onnx as the get-up policy. infer_policy
switches to the StandUp session whenever |vel_cmd| <= 0.05, so recovery = zero cmd.
Any fall (pushed or not) triggers recovery; walking resumes once it is upright.
If the get-up policy deadlocks (e.g. head pinned under the trunk, actuators saturated,
nothing moving) for 3 s, the joints are driven to the default pose for 0.5 s, then it retries.

  --headless SECONDS   run without a window and print events (for testing)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
# Pollen microduck_rl checkout (scripts/fetch_upstream.sh puts it here); override with MICRODUCK_RL.
REPO = Path(os.environ.get("MICRODUCK_RL", ROOT / "upstream/microduck_rl"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from infer_policy import BAM_KP_FW, PolicyInference, load_bam_model, load_mujoco_with_bam  # noqa: E402
from play_standup_flat import to_sitting  # noqa: E402

SCENE = REPO / "src/mjlab_microduck/robot/microduck/scene_ramp_steps_demo.xml"

# Rectangle course (x, y). Leg 0->1 runs over ramp (x .4-1.0), platform (1.0-2.0), steps (2.0-2.6).
WAYPOINTS = [(-0.3, 0.0), (3.1, 0.0), (3.1, 1.2), (-0.3, 1.2)]
WP_RADIUS = 0.15
VX = 0.3
HEADING_K = 2.0
MAX_WZ = 0.8
TURN_IN_PLACE = math.radians(50)  # heading error above this -> slow down to turn

PUSH_X = (1.30, 1.45)        # push while on the platform, outbound leg only
PUSH_FORCE = tuple(float(v) for v in os.environ.get("PUSH_FORCE", "8,0,0").split(","))  # N, sideways on the trunk
PUSH_S = 0.15

FALL_UPRIGHT = 0.6
RECOVERED_UPRIGHT = 0.85
RECOVER_HOLD_S = 1.0
STUCK_S = 3.0       # still down after this long in RECOVER -> unstick
UNSTICK_S = 0.5     # drive joints to the default pose (policy off), then retry
CONTROL_DT = 0.02
DECIMATION = 4


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Course:
    def __init__(self, model, data, policy, bam_ctrl):
        self.m, self.d, self.p, self.bam = model, data, policy, bam_ctrl
        self.trunk = model.body("trunk_base").id
        fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.adr = int(model.jnt_qposadr[fj])
        self.step = 0
        self.state = "SIT"
        self.state_steps = 0
        self.wp = 1
        self.laps = 0
        self.pushed_this_lap = False
        self.push_left = 0
        self.upright_steps = 0
        self.falls = 0
        self.unstick_left = 0
        self.unsticks = 0
        self.sit_targets = to_sitting(model, data, policy, bam_ctrl)
        policy.set_vel_cmd(0.0, 0.0, 0.0)

    def log(self, msg):
        print(f"\n[t={self.step * CONTROL_DT:6.1f}s] {msg}", flush=True)

    def set_state(self, s):
        self.state, self.state_steps = s, 0

    def tick(self):
        d, p, a = self.d, self.p, self.adr
        x, y = float(d.qpos[a]), float(d.qpos[a + 1])
        upright = float(d.xmat[self.trunk].reshape(3, 3)[2, 2])

        if self.state == "SIT":
            p.set_position_targets(self.sit_targets)
            if self.state_steps * CONTROL_DT >= 1.5:
                self.log("STAND UP (StandUp policy)")
                self.set_state("STANDUP")
        elif self.state == "STANDUP":
            if self.state_steps * CONTROL_DT >= 3.0:
                self.log("WALK")
                self.set_state("WALK")
        elif self.state == "WALK":
            if upright < FALL_UPRIGHT:
                self.falls += 1
                self.log(f"FALLEN (upright {upright:+.2f}) -> StandUp policy")
                p.set_vel_cmd(0.0, 0.0, 0.0)
                self.upright_steps = 0
                self.set_state("RECOVER")
            else:
                tx, ty = WAYPOINTS[self.wp]
                if math.hypot(tx - x, ty - y) < WP_RADIUS:
                    self.wp = (self.wp + 1) % len(WAYPOINTS)
                    if self.wp == 1:
                        self.laps += 1
                        self.pushed_this_lap = False
                        self.log(f"LAP {self.laps} done -> again")
                    tx, ty = WAYPOINTS[self.wp]
                err = wrap(math.atan2(ty - y, tx - x) - yaw_of(d.qpos[a + 3 : a + 7]))
                wz = float(np.clip(HEADING_K * err, -MAX_WZ, MAX_WZ))
                vx = VX if abs(err) < TURN_IN_PLACE else 0.1
                p.vel_cmd = np.array([vx, 0.0, wz], dtype=np.float32)
                p._update_policy_session()
                p._update_command()
                if self.wp == 1 and not self.pushed_this_lap and PUSH_X[0] < x < PUSH_X[1]:
                    self.pushed_this_lap = True
                    self.push_left = int(PUSH_S / CONTROL_DT)
                    self.log("PUSH on the platform")
        elif self.state == "RECOVER":
            self.upright_steps = self.upright_steps + 1 if upright > RECOVERED_UPRIGHT else 0
            if self.unstick_left > 0:
                self.unstick_left -= 1
                if self.unstick_left == 0:
                    p.last_action[:] = 0.0
                    self.state_steps = 0
            elif upright < RECOVERED_UPRIGHT and self.state_steps * CONTROL_DT >= STUCK_S:
                self.unsticks += 1
                self.unstick_left = int(UNSTICK_S / CONTROL_DT)
                self.log(f"STUCK (upright {upright:+.2f}) -> reset limbs to default pose, retry")
            if self.upright_steps * CONTROL_DT >= RECOVER_HOLD_S:
                self.log(f"RECOVERED (upright {upright:+.2f}) -> walk")
                self.set_state("WALK")

        d.xfrc_applied[self.trunk, :3] = PUSH_FORCE if self.push_left > 0 else (0.0, 0.0, 0.0)
        self.push_left = max(0, self.push_left - 1)

        if self.state == "RECOVER" and self.unstick_left > 0:
            p.set_position_targets(p.default_pose.copy())
        elif self.state != "SIT":
            p.apply_action(p.infer())
        for _ in range(DECIMATION):
            if self.bam is not None:
                self.bam.update()
            mujoco.mj_step(self.m, d)
        self.step += 1
        self.state_steps += 1
        return x, y, float(d.qpos[a + 2]), upright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=float, default=0.0)
    args = ap.parse_args()
    standup_onnx = ROOT / "upstream/policies/alpha_stand.onnx"

    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        str(SCENE), bam_model, timestep=0.005, vin_drop_gain=0.1, vin_min=6.0
    )
    policy = PolicyInference(
        model, data,
        walking_onnx_path=str(ROOT / "policies/walk_flat_10k.onnx"),
        standing_onnx_path=str(standup_onnx),
        new_cmd_obs=True, bam_ctrl=bam_ctrl, use_projected_gravity=True,
    )
    course = Course(model, data, policy, bam_ctrl)
    print(f"=== LOOP COURSE: ramp up -> push/fall -> stand up -> steps down -> loop ===")

    if args.headless:
        for _ in range(int(args.headless / CONTROL_DT)):
            x, y, z, up = course.tick()
            if course.step % 50 == 0:
                print(f"  {course.state:8s} wp={course.wp} xy=({x:+.2f},{y:+.2f}) z={z*1000:5.1f}mm up={up:+.2f}")
        print(f"\nlaps={course.laps} falls={course.falls} unsticks={course.unsticks}")
        return

    print("Close window to quit.")
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = course.trunk
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.3, -25, 135
        t0 = time.perf_counter()
        while viewer.is_running():
            x, y, z, up = course.tick()
            viewer.sync()
            if course.step % 10 == 0:
                print(f"\r{course.state:8s} lap={course.laps} falls={course.falls} unstick={course.unsticks} xy=({x:+.2f},{y:+.2f}) "
                      f"z={z*1000:5.1f}mm up={up:+.2f}   ", end="", flush=True)
            lag = t0 + course.step * CONTROL_DT - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        print()


if __name__ == "__main__":
    main()
