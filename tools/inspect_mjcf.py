import sys, mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(sys.argv[1])
print("mujoco", mujoco.__version__, "nq nv nu nbody ngeom nsensor", m.nq, m.nv, m.nu, m.nbody, m.ngeom, m.nsensor)
print("opt dt", m.opt.timestep, "integrator", m.opt.integrator, "cone", m.opt.cone, "solver", m.opt.solver, "iters", m.opt.iterations, m.opt.ls_iterations)
print("total mass", m.body_subtreemass[1])
for j in range(m.njnt):
    print("J", j, m.joint(j).name, "type", m.jnt_type[j], "range", np.round(m.jnt_range[j],4), "lim", m.jnt_limited[j], "arm", round(float(m.dof_armature[m.jnt_dofadr[j]]),5), "damp", round(float(m.dof_damping[m.jnt_dofadr[j]]),5), "fl", round(float(m.dof_frictionloss[m.jnt_dofadr[j]]),5))
for a in range(m.nu):
    print("A", a, m.actuator(a).name, "->", m.joint(m.actuator_trnid[a,0]).name, "gain", m.actuator_gainprm[a,0], "bias", m.actuator_biasprm[a,:3], "frc", m.actuator_forcerange[a])
for s in range(m.nsensor):
    print("S", s, m.sensor(s).name, "type", m.sensor_type[s], "obj", m.sensor_objid[s], "dim", m.sensor_dim[s])
for g in range(m.ngeom):
    n = m.geom(g).name
    if n: print("G", g, n, "body", m.body(m.geom_bodyid[g]).name, "type", m.geom_type[g], "ct/ca", m.geom_contype[g], m.geom_conaffinity[g], "condim", m.geom_condim[g], "fr", m.geom_friction[g])
print("sites", [m.site(i).name for i in range(m.nsite)])
print("keys", [m.key(i).name for i in range(m.nkey)])
