#!/usr/bin/env python3
"""Render the loop course (demo/play_terrain_loop.py) offscreen to a GIF.

    cd upstream/microduck_rl && PYTHONPATH=scripts:. .venv/bin/python ../../demo/record_gif.py out.gif 45
"""
import sys
from pathlib import Path

import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_terrain_loop as L  # noqa: E402

out, secs = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
bam_model = L.load_bam_model(L.BAM_KP_FW, vin=7.4, max_current=None)
model, data, bam, _ = L.load_mujoco_with_bam(str(L.SCENE), bam_model, timestep=0.005, vin_drop_gain=0.1, vin_min=6.0)
pol = L.PolicyInference(model, data, walking_onnx_path=str(L.ROOT / "policies/walk_flat_10k.onnx"),
                        standing_onnx_path=str(L.ROOT / "upstream/policies/alpha_stand.onnx"),
                        new_cmd_obs=True, bam_ctrl=bam, use_projected_gravity=True)
course = L.Course(model, data, pol, bam)
model.vis.global_.offwidth = max(model.vis.global_.offwidth, 800)
model.vis.global_.offheight = max(model.vis.global_.offheight, 500)
r = mujoco.Renderer(model, 400, 640)
cam = mujoco.MjvCamera()
cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, course.trunk
cam.distance, cam.elevation, cam.azimuth = 1.3, -25, 135
frames = []
for k in range(int(secs / L.CONTROL_DT)):
    course.tick()
    if k % 3 == 0:
        r.update_scene(data, cam)
        frames.append(Image.fromarray(r.render()))
print("laps", course.laps, "falls", course.falls, "unsticks", course.unsticks)
frames = [f.resize((480, 300)).convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
frames[0].save(out, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
