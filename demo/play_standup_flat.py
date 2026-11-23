#!/usr/bin/env python3
"""Show StandUp-Flat: sit keyframe → stand (the task's main skill)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Pollen microduck_rl checkout (scripts/fetch_upstream.sh puts it here); override with MICRODUCK_RL.
REPO = Path(os.environ.get("MICRODUCK_RL", ROOT / "upstream/microduck_rl"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from infer_policy import (  # noqa: E402
    BAM_KP_FW,
    PolicyInference,
    load_bam_model,
    load_mujoco_with_bam,
)
from mjlab_microduck.tasks.microduck_standup_env_cfg import (  # noqa: E402
    SIT_Z,
    SITTING_JOINT_OVERRIDES,
)

ONNX = Path(__file__).resolve().parent / "standup_flat" / "policy.onnx"
SCENE = REPO / "src/mjlab_microduck/robot/microduck/scene.xml"


def to_sitting(model, data, policy, bam_ctrl):
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = int(model.jnt_qposadr[free_id])
    data.qpos[adr : adr + 3] = [0.0, 0.0, SIT_Z]
    data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    # Sitting joint overrides are articulation indices → qpos[7 + idx]
    for idx, ang in SITTING_JOINT_OVERRIDES.items():
        data.qpos[7 + idx] = ang
    data.qvel[:] = 0.0
    sit_targets = np.array([data.qpos[qi] for qi in policy.joint_qpos_indices], dtype=np.float32)
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(sit_targets)
    policy.last_action[:] = 0.0
    mujoco.mj_forward(model, data)
    return sit_targets


def main():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        str(SCENE), bam_model, timestep=0.005, vin_drop_gain=0.1, vin_min=6.0
    )
    policy = PolicyInference(
        model,
        data,
        standing_onnx_path=str(ONNX),
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
        use_projected_gravity=True,
    )
    sit_targets = to_sitting(model, data, policy, bam_ctrl)

    print("=== STANDUP FLAT (model_16000) — sit → stand ===")
    print("Holds sitting ~2s, then stands. Re-sits every 8s. Close window to quit.")
    trunk = model.body("trunk_base").id
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = int(model.jnt_qposadr[free_id])
    control_dt = 0.02
    decimation = 4
    hold_steps = int(2.0 / control_dt)
    cycle_steps = int(8.0 / control_dt)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = trunk
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.9, -20, 135
        t0 = time.perf_counter()
        step = 0
        while viewer.is_running():
            phase = step % cycle_steps
            if phase == 0:
                sit_targets = to_sitting(model, data, policy, bam_ctrl)
                if step:
                    print("\n[re-sit]")
                else:
                    print("[sitting hold]")
            if phase < hold_steps:
                policy.set_position_targets(sit_targets)
            else:
                if phase == hold_steps:
                    print("\n[standing up]")
                action = policy.infer()
                policy.apply_action(action)
            for _ in range(decimation):
                if bam_ctrl is not None:
                    bam_ctrl.update()
                mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            if step % 25 == 0:
                z = float(data.qpos[adr + 2])
                upright = float(data.xmat[trunk].reshape(3, 3)[2, 2])
                mode = "HOLD" if phase < hold_steps else "ACT "
                print(
                    f"\r{mode} z={z * 1000:6.1f} mm  upright={upright:+.2f}  t={step * control_dt:5.1f}s   ",
                    end="",
                    flush=True,
                )
            lag = t0 + step * control_dt - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        print()


if __name__ == "__main__":
    main()
