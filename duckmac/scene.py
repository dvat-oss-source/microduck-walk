"""Build the Mjlab-Velocity2-Rough-Backlash-MicroDuck scene as a plain MuJoCo model.

Every edit here replicates a specific mjlab / microduck code path (cited inline, see
SPEC.md for aliases). The result is a single compiled mjModel; per-env domain
randomization is applied later by the batched stepper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from duckmac.vendor.terrains import heightfield_terrains as hf
from duckmac.vendor.terrains import primitive_terrains as pt
from duckmac.vendor.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "upstream/rl_cb8561f6/src/mjlab_microduck/robot/microduck/robot_walk_backlash.xml"
MJLAB_SCENE_XML = ROOT / "upstream/deps/src/mjlab/scene/scene.xml"

# Actuator / joint order (XML actuator order == ONNX joint_names metadata).
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
# RC:73-97 HOME_FRAME (BACKLASH_HOME_FRAME pins backlash joints at 0).
HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}

# BAM m6 XL330 (M6 json). Only values needed at spec-edit time.
BAM_KT = 0.36601349688984386
BAM_R = 2.8113923539223227
BAM_ARMATURE = 0.0018077432831600838
VIN_RANGE = (6.5, 8.2)  # RC:126
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)  # BAMJ:221-232
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

# Terrain: what actually ran (CFG:2188-2317; SPEC §2 "terrain discrepancy").
TERRAIN_ROWS = 5
TERRAIN_COLS = 5
TERRAIN_SIZE = (8.0, 8.0)


def terrain_cfg(seed: int | None, flat: bool = False) -> TerrainGeneratorCfg:
    cfg = TerrainGeneratorCfg(
        seed=seed,
        curriculum=False,
        size=TERRAIN_SIZE,
        border_width=20.0,
        border_height=1.0,
        num_rows=TERRAIN_ROWS,
        num_cols=TERRAIN_COLS,
        difficulty_range=(0.0, 1.0),
        add_lights=False,
        sub_terrains={
            "flat": pt.BoxFlatTerrainCfg(proportion=0.25),
            "pyramid_stairs": pt.BoxPyramidStairsTerrainCfg(
                proportion=0.25, step_height_range=(0.0, 0.015), step_width=0.15,
                platform_width=2.0, border_width=1.0, holes=False,
            ),
            "random_grid": pt.BoxRandomGridTerrainCfg(
                proportion=0.30, grid_width=0.45, grid_height_range=(0.0, 0.010),
                platform_width=1.5, border_width=0.25, holes=False,
                merge_similar_heights=False, height_merge_threshold=0.05,
                max_merge_distance=3,
            ),
            "pyramid_slope": hf.HfPyramidSlopedTerrainCfg(
                proportion=0.20, slope_range=(0.03, 0.10), platform_width=2.0,
                inverted=False, border_width=0.0, horizontal_scale=0.1,
                vertical_scale=0.001, base_thickness_ratio=1.0,
            ),
        },
    )
    if flat:  # diagnostics only: same grid and contact settings, every tile flat
        cfg.sub_terrains = {"flat": cfg.sub_terrains["flat"]}
    return cfg


# ── mjlab utils replicated verbatim (ML utils/string.py) ─────────────────────
def _filter_exp(exprs, names):
    pats = [re.compile(e) for e in exprs]
    return tuple(n for n in names if any(p.match(n) for p in pats))


def _resolve_expr(pattern_map, names, default_val=None):
    pats = [(re.compile(p), v) for p, v in pattern_map.items()]
    out = []
    for n in names:
        for p, v in pats:
            if p.match(n):
                out.append(v)
                break
        else:
            out.append(default_val)
    return tuple(out)


def _resolve_field(field, names, default_val=None):
    if isinstance(field, dict):
        return _resolve_expr(field, names, default_val)
    return tuple([field] * len(names))


def _apply_full_collision(spec: mujoco.MjSpec) -> None:
    """RC:99-104 FULL_COLLISION through ML utils/spec_config.py:263-301 (CollisionCfg.edit_spec).

    Faithfully includes mjlab's behaviour for unnamed geoms: `other_geoms` is a *set*
    of names, so all unnamed geoms collapse to '' and `spec.geom('')` disables only one
    of them. The XML's unnamed `self_collision_only` geoms (contype=conaffinity=2) stay
    collidable with each other.
    """
    geom_names_expr = (".*_collision",)
    condim = {r"^(left|right)_foot_collision$": 3, ".*_collision": 1}
    priority = {r"^(left|right)_foot_collision$": 1}
    friction = {r"^(left|right)_foot_collision$": (1.0,)}
    all_names = tuple(g.name for g in spec.geoms)
    subset = _filter_exp(geom_names_expr, all_names)
    r_condim = _resolve_field(condim, subset, 3)
    r_contype = _resolve_field(1, subset, 1)
    r_conaff = _resolve_field(1, subset, 1)
    r_prio = _resolve_field(priority, subset, 0)
    r_fric = _resolve_field(friction, subset, None)
    for i, name in enumerate(subset):
        g = spec.geom(name)
        g.condim = r_condim[i]
        g.contype = r_contype[i]
        g.conaffinity = r_conaff[i]
        g.priority = r_prio[i]
        if r_fric[i] is not None:
            for k, v in enumerate(r_fric[i]):
                g.friction[k] = v
    for name in set(all_names).difference(subset):
        g = spec.geom(name)
        g.contype = 0
        g.conaffinity = 0


def _apply_bam_edit_spec(spec: mujoco.MjSpec, target_names: list[str]) -> None:
    """BAMJ:234-293 BamActuator.edit_spec (all 14 targets already have XML actuators)."""
    force_limit = max(VIN_RANGE) * BAM_KT / BAM_R
    tset = set(target_names)
    converted = set()
    for act in spec.actuators:
        tgt = act.target
        tname = tgt.name if hasattr(tgt, "name") else (str(tgt) if tgt else None)
        if tname in tset:
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-force_limit, force_limit)
            act.gear = [1.0, 0, 0, 0, 0, 0]
            for j in spec.joints:
                if j.name == tname:
                    j.armature = float(BAM_ARMATURE)
                    j.damping = np.zeros((3, 1))
                    j.frictionloss = 0.0
                    j.solref_friction = STIFF_SOLREF_FRICTION
                    j.solimp_friction = STIFF_SOLIMP_FRICTION
                    break
            converted.add(tname)
    missing = tset - converted
    if missing:
        raise RuntimeError(f"BAM edit_spec: targets without XML actuator: {missing}")


def robot_spec(xml=None) -> mujoco.MjSpec:
    """Entity build for MICRODUCK_WALK_BACKLASH_ROBOT_CFG (ML entity/entity.py:143-174)."""
    spec = mujoco.MjSpec.from_file(str(xml or XML))
    joints = spec.joints
    if joints and joints[0].type == mujoco.mjtJoint.mjJNT_FREE and not joints[0].name:
        joints[0].name = "floating_base_joint"
    _apply_full_collision(spec)
    non_free = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    targets = [n for n in non_free if re.match(r"^(?!passive_).*", n)]
    _apply_bam_edit_spec(spec, targets)
    return spec


@dataclass
class Scene:
    model: mujoco.MjModel
    terrain_origins: np.ndarray  # (rows, cols, 3)
    terrain_seed: int


def build_scene(terrain_seed: int, flat: bool = False, soften: bool = True, robot_xml=None) -> Scene:
    """ML scene/scene.py:60-68: scene.xml → terrain → entities → sensors → spec_fn."""
    scene = mujoco.MjSpec.from_file(str(MJLAB_SCENE_XML))
    scene.stat.extent = 2.0  # CFG scene.extent

    # Terrain entity (ML terrains/terrain_entity.py:_build_spec). Env-origin /
    # terrain-origin / flat-patch marker sites are visual only and omitted.
    # Pollen's run used seed=None (random layout); we fix the seed for reproducibility.
    gen = TerrainGenerator(terrain_cfg(seed=terrain_seed, flat=flat))
    tspec = mujoco.MjSpec()
    gen.compile(tspec)
    scene.attach(tspec, prefix="", frame=scene.worldbody.add_frame())

    rspec = robot_spec(robot_xml)
    scene.attach(rspec, prefix="robot/", frame=scene.worldbody.add_frame())

    # Sensors (CFG:2096-2186; ML sensor/contact_sensor.py:541-602).
    # feet_ground_contact: geoms × fields (found, force), netforce, 1 slot, vs body 'terrain'.
    for prim in ("left_foot_collision", "right_foot_collision"):
        for field, bit in (("found", 0), ("force", 1)):
            scene.add_sensor(
                name=f"feet_ground_contact_{prim}_{field}",
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=f"robot/{prim}",
                reftype=mujoco.mjtObj.mjOBJ_BODY, refname="terrain",
                intprm=[1 << bit, 3, 1],
            )
    # self_collision: subtree(trunk_base) vs subtree(trunk_base), found, reduce none.
    scene.add_sensor(
        name="self_collision_trunk_base_found",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY, objname="robot/trunk_base",
        reftype=mujoco.mjtObj.mjOBJ_XBODY, refname="robot/trunk_base",
        intprm=[1 << 0, 0, 1],
    )

    # spec_fn = _soften_terrain_contacts (V:151-169).
    for g in scene.body("terrain").geoms if soften else ():
        g.solref = [0.04, 1.0]
        g.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]

    model = scene.compile()

    # MujocoCfg.apply (ML sim/sim.py:113-138; CFG:2338-2365).
    o = model.opt
    o.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.timestep = 0.005
    o.impratio = 1.0
    o.gravity[:] = (0.0, 0.0, -9.81)
    o.iterations = 30
    o.tolerance = 1e-8
    o.ls_iterations = 50
    o.ls_tolerance = 0.01
    o.ccd_iterations = 50
    return Scene(model=model, terrain_origins=gen.terrain_origins.copy(), terrain_seed=terrain_seed)


if __name__ == "__main__":
    import time
    t = time.time()
    s = build_scene(terrain_seed=0)
    m = s.model
    print(f"built in {time.time()-t:.1f}s  nq {m.nq} nv {m.nv} nu {m.nu} nbody {m.nbody} "
          f"ngeom {m.ngeom} nsensor {m.nsensor} nsensordata {m.nsensordata} nhfield {m.nhfield}")
    print("model bytes", mujoco.mj_sizeModel(m))
    coll = [(i, m.geom(i).name, int(m.geom_contype[i]), int(m.geom_conaffinity[i]), m.body(m.geom_bodyid[i]).name)
            for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and m.geom_bodyid[i] != m.body("terrain").id]
    print("robot collidable geoms:", coll)
    print("terrain origins z:", np.round(s.terrain_origins[..., 2], 4).tolist())
