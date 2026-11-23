"""C++ stepper vs an independent Python loop using Pollen's BAM classes (fixed lag, N=1).

Voltage and torque come from bam.actuator; the friction budget is transcribed from
bam/mjlab.py:_compute_friction_budget using bam's Model parameters. Offline test only.
"""
import sys, types
import numpy as np, mujoco

sys.modules.setdefault("colorama", types.SimpleNamespace(Fore=types.SimpleNamespace(YELLOW="", RED="", GREEN="", BLUE="", CYAN="", RESET=""),
    Back=types.SimpleNamespace(), Style=types.SimpleNamespace(RESET_ALL="", BRIGHT="")))
sys.path.insert(0, "upstream/deps/bam")
import bam.model as bm

from duckmac._batch import Batch
from duckmac.scene import HOME, JOINTS, build_scene

LAG, VIN, GAIN, FS, VIN_MIN = 4, 7.5, 0.1, 1.05, 6.0


def budget(M, tm, te, s):
    fl = M.friction_base.value + s * M.friction_stribeck.value
    fl = fl + abs(te * M.load_friction_external.value - tm * M.load_friction_motor.value)
    fl = fl + s * abs(te * M.load_friction_external_stribeck.value - tm * M.load_friction_motor_stribeck.value)
    q = M.load_friction_external_quad.value * abs(te) ** 2 if abs(tm) > abs(te) else M.load_friction_motor_quad.value * abs(tm) ** 2
    return (fl + s * q) * FS


def run(limit):
    sc = build_scene(0, flat=True)
    m = sc.model
    M = bm.load_model("upstream/deps/bam/bam/params/xl330/m6.json")
    act = M.actuator
    act.kp = 200.0
    if not limit:
        act.max_current = None
    d = mujoco.MjData(m)
    b = Batch(m._address, 1, 1, 0, limit)
    b.set_fixed_lag(LAG)
    A = b.arrays()
    ix = b.indices()
    sq, sd, lq = np.array(ix["servo_qadr"]), np.array(ix["servo_dadr"]), np.array(ix["backlash_qadr"])
    home = np.array([HOME[j] for j in JOINTS])
    q0 = m.qpos0.copy()
    q0[:3] = sc.terrain_origins[2, 2] + [0, 0, 0.125]
    q0[sq] = home
    A["qpos"][0] = q0
    A["vin_env"][0], A["vin_gain"][0], A["friction_scale"][0] = VIN, GAIN, FS
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    b.forward_sense()

    def stash():
        fr = np.zeros(m.nv)
        for i in range(d.nefc):
            if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF:
                fr[d.efc_id[i]] += d.efc_force[i]
        return (-d.qfrc_bias + d.qfrc_constraint - fr)[sd].copy(), d.qfrc_actuator[sd].copy()

    hist, ptau = [], np.zeros(14)
    worst = dict(qpos=0.0, ctrl=0.0, fl=0.0)
    for t in range(150):
        tgt = home + 0.15 * np.sin(0.2 * t + np.arange(14))
        A["targets"][0] = tgt
        te_tm = stash()
        for k in range(4):
            if k:
                te_tm = stash()
            te, tm = te_tm
            hist.append(tgt.copy())
            delayed = hist[-1 - min(LAG, len(hist) - 1)]
            vin = max(VIN - GAIN * np.abs(ptau).sum(), VIN_MIN)
            act.vin = vin
            vel = d.qvel[sd].copy()
            pos = d.qpos[sq] + d.qpos[lq]
            ctrl = np.zeros(14)
            for j in range(14):
                volts = act.compute_control(delayed[j], pos[j], vel[j], 0.005)
                ctrl[j] = act.compute_torque(volts, True, pos[j], vel[j])
                s = np.exp(-((abs(vel[j]) / M.dtheta_stribeck.value) ** M.alpha.value))
                m.dof_frictionloss[sd[j]] = budget(M, tm[j], te[j], s)
                m.dof_damping[sd[j]] = M.friction_viscous.value
            d.ctrl[:] = ctrl
            ptau = ctrl
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        b.step()
        b.forward_sense()
        worst["qpos"] = max(worst["qpos"], np.abs(A["qpos"][0] - d.qpos).max())
        worst["ctrl"] = max(worst["ctrl"], np.abs(A["ctrl"][0] - d.ctrl).max())
        worst["fl"] = max(worst["fl"], np.abs(A["dof_frictionloss"][0] - m.dof_frictionloss).max())
    return worst, ctrl, float(np.abs(m.dof_frictionloss[sd]).max())


if __name__ == "__main__":
    for lim in (True, False):
        w, c, f = run(lim)
        print(f"limiter={lim} max diff over 150 control steps: {w}  (|ctrl| max {np.abs(c).max():.3f}, fl max {f:.4f})")
