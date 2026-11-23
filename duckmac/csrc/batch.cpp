// Multithreaded batched MuJoCo stepper for the microduck velocity task.
//
// Replicates, per physics substep and per env, the mjlab 1.3.0 step order
// (SPEC §1) with the BAM m6 XL330 actuator + backlash encoder (SPEC §3).
// Only physics-side work lives here; managers (rewards, commands, obs, DR
// sampling) are numpy code in duckmac/env.py.
//
// Design: per-env state (qpos, qvel, qacc_warmstart, ctrl, time) and per-env
// model "patches" (fields mjlab expands per world) are stored here; each worker
// thread owns one mjModel+mjData, applies the env's patch, loads its state,
// steps, and saves back. Per-env RNG streams make results independent of thread
// count and scheduling.

#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int J = 14;          // servo joints / actuators
constexpr int HIST = 7;        // actuator delay buffer length (max_lag 6 + 1), ML actuator.py:270-302
constexpr int LAG_MIN = 3;     // RC:130-131
constexpr int LAG_MAX = 6;
constexpr int DECIM = 4;       // CFG decimation

// BAM m6 XL330 (deps/bam/bam/params/xl330/m6.json; XL:79-99; RC:120-131).
constexpr double KT = 0.36601349688984386;
constexpr double RES = 2.8113923539223227;
constexpr double F_BASE = 0.004771183165566;
constexpr double F_STRIBECK = 0.004676345799486616;
constexpr double LF_MOTOR = 0.2667860954283698;
constexpr double LF_EXT = 8.515871897059342e-06;
constexpr double LF_MOTOR_S = 1.0722918395099123e-05;
constexpr double LF_EXT_S = 0.08077928978935671;
constexpr double LF_MOTOR_Q = 0.009972471242139415;
constexpr double LF_EXT_Q = 0.004902565732332559;
constexpr double DTHETA_S = 2.890372094130307;
constexpr double ALPHA_S = 8.683259907618984;
constexpr double F_VISCOUS = 0.005359668274599504;
constexpr double KP_FW = 200.0;
constexpr double ERROR_GAIN = (4096.0 / (2.0 * M_PI)) / (256.0 * 885.0);
constexpr double MAX_CURRENT = 1.75;
constexpr double MAX_PWM = 1.0;
constexpr double VIN_MIN = 6.0;

// splitmix64: one independent stream per env.
struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  int randint(int lo, int hi_incl) {  // uniform in [lo, hi_incl]
    uint64_t span = (uint64_t)(hi_incl - lo + 1);
    return lo + (int)(next() % span);
  }
};

class Pool {
 public:
  explicit Pool(int n) : n_(n) {
    for (int i = 0; i < n; ++i) {
      th_.emplace_back([this, i] {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        uint64_t seen = 0;
        for (;;) {
          std::function<void(int)> job;
          {
            std::unique_lock<std::mutex> l(mu_);
            cv_.wait(l, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            job = job_;
          }
          job(i);
          {
            std::lock_guard<std::mutex> l(mu_);
            if (--pending_ == 0) done_.notify_all();
          }
        }
      });
    }
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  void run(const std::function<void(int)>& f) {
    {
      std::lock_guard<std::mutex> l(mu_);
      job_ = f;
      pending_ = n_;
      ++gen_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> l(mu_);
    done_.wait(l, [&] { return pending_ == 0; });
  }
  int size() const { return n_; }

 private:
  int n_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  std::function<void(int)> job_;
  uint64_t gen_ = 0;
  int pending_ = 0;
  bool stop_ = false;
};

// Static grid over the group-0 static geoms (terrain) for straight-down rays.
// Per-geom intersection uses MuJoCo's own mju_rayGeom / mj_rayHfield, so the
// result equals mj_ray restricted to the same geoms (validated in tests).
struct RayGrid {
  double x0 = 0, y0 = 0, cell = 0.25;
  int nx = 0, ny = 0;
  std::vector<int> start, ids;
  std::vector<int> geoms;

  void build(const mjModel* m, const mjData* d) {
    std::vector<std::array<double, 4>> box;  // xmin, xmax, ymin, ymax
    double gxmin = 1e30, gxmax = -1e30, gymin = 1e30, gymax = -1e30;
    for (int g = 0; g < m->ngeom; ++g) {
      if (m->geom_group[g] != 0) continue;
      if (m->geom_rgba[4 * g + 3] == 0) continue;
      int b = m->geom_bodyid[g];
      if (m->body_weldid[b] != 0) {
        throw std::runtime_error("group-0 geom on a non-static body: " + std::to_string(g));
      }
      int t = m->geom_type[g];
      const double* R = d->geom_xmat + 9 * g;
      const double* p = d->geom_xpos + 3 * g;
      double hx, hy;
      if (t == mjGEOM_BOX) {
        const double* s = m->geom_size + 3 * g;
        hx = std::fabs(R[0]) * s[0] + std::fabs(R[1]) * s[1] + std::fabs(R[2]) * s[2];
        hy = std::fabs(R[3]) * s[0] + std::fabs(R[4]) * s[1] + std::fabs(R[5]) * s[2];
      } else if (t == mjGEOM_HFIELD) {
        const double* s = m->hfield_size + 4 * m->geom_dataid[g];
        double sz = std::max(s[2], s[3]);
        hx = std::fabs(R[0]) * s[0] + std::fabs(R[1]) * s[1] + std::fabs(R[2]) * sz;
        hy = std::fabs(R[3]) * s[0] + std::fabs(R[4]) * s[1] + std::fabs(R[5]) * sz;
      } else {
        throw std::runtime_error("unsupported terrain geom type " + std::to_string(t));
      }
      double e = 1e-6;
      box.push_back({p[0] - hx - e, p[0] + hx + e, p[1] - hy - e, p[1] + hy + e});
      geoms.push_back(g);
      gxmin = std::min(gxmin, box.back()[0]);
      gxmax = std::max(gxmax, box.back()[1]);
      gymin = std::min(gymin, box.back()[2]);
      gymax = std::max(gymax, box.back()[3]);
    }
    x0 = gxmin;
    y0 = gymin;
    nx = (int)std::ceil((gxmax - gxmin) / cell) + 1;
    ny = (int)std::ceil((gymax - gymin) / cell) + 1;
    std::vector<std::vector<int>> cells((size_t)nx * ny);
    for (size_t k = 0; k < geoms.size(); ++k) {
      int ix0 = std::max(0, (int)std::floor((box[k][0] - x0) / cell));
      int ix1 = std::min(nx - 1, (int)std::floor((box[k][1] - x0) / cell));
      int iy0 = std::max(0, (int)std::floor((box[k][2] - y0) / cell));
      int iy1 = std::min(ny - 1, (int)std::floor((box[k][3] - y0) / cell));
      for (int ix = ix0; ix <= ix1; ++ix)
        for (int iy = iy0; iy <= iy1; ++iy) cells[(size_t)ix * ny + iy].push_back(geoms[k]);
    }
    start.assign((size_t)nx * ny + 1, 0);
    for (size_t c = 0; c < cells.size(); ++c) start[c + 1] = start[c] + (int)cells[c].size();
    ids.reserve(start.back());
    for (auto& c : cells) ids.insert(ids.end(), c.begin(), c.end());
  }

  // Ray from pnt straight down (vec = (0,0,-1)). Returns distance or -1.
  double cast_down(const mjModel* m, const mjData* d, const double pnt[3], double normal[3]) const {
    const double vec[3] = {0, 0, -1};
    int ix = (int)std::floor((pnt[0] - x0) / cell);
    int iy = (int)std::floor((pnt[1] - y0) / cell);
    normal[0] = normal[1] = normal[2] = 0;
    if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) return -1;
    size_t c = (size_t)ix * ny + iy;
    double best = -1;
    double nrm[3];
    for (int k = start[c]; k < start[c + 1]; ++k) {
      int g = ids[k];
      double x;
      if (m->geom_type[g] == mjGEOM_HFIELD) {
        x = mj_rayHfield(m, d, g, pnt, vec, nrm);
      } else {
        x = mju_rayGeom(d->geom_xpos + 3 * g, d->geom_xmat + 9 * g, m->geom_size + 3 * g, pnt, vec,
                        m->geom_type[g], nrm);
      }
      if (x >= 0 && (best < 0 || x < best)) {
        best = x;
        normal[0] = nrm[0];
        normal[1] = nrm[1];
        normal[2] = nrm[2];
      }
    }
    return best;
  }
};

}  // namespace

class Batch {
 public:
  Batch(uintptr_t model_addr, int num_envs, int num_threads, uint64_t seed, bool current_limit)
      : N_(num_envs), current_limit_(current_limit) {
    const mjModel* src = reinterpret_cast<const mjModel*>(model_addr);
    base_ = mj_copyModel(nullptr, src);
    if (!base_) throw std::runtime_error("mj_copyModel failed");
    nq_ = base_->nq;
    nv_ = base_->nv;
    nb_ = base_->nbody;
    nu_ = base_->nu;
    if (nu_ != J) throw std::runtime_error("expected 14 actuators");
    resolve_indices();

    // Static terrain ray grid (forward kinematics at qpos0; terrain is world-fixed).
    {
      mjData* d = mj_makeData(base_);
      mj_forward(base_, d);
      grid_.build(base_, d);
      mj_deleteData(d);
    }
    scratch_model_ = mj_copyModel(nullptr, base_);
    scratch_data_ = mj_makeData(scratch_model_);

    auto z = [&](std::vector<double>& v, size_t n) { v.assign(n, 0.0); };
    z(qpos_, (size_t)N_ * nq_);
    z(qvel_, (size_t)N_ * nv_);
    z(qacc_ws_, (size_t)N_ * nv_);
    z(ctrl_, (size_t)N_ * nu_);
    z(time_, N_);
    // Model patches, initialized from the compiled (BAM-edited) model.
    auto rep = [&](std::vector<double>& v, const double* p, int n) {
      v.resize((size_t)N_ * n);
      for (int e = 0; e < N_; ++e) std::memcpy(&v[(size_t)e * n], p, sizeof(double) * n);
    };
    rep(p_fl_, base_->dof_frictionloss, nv_);
    rep(p_damp_, base_->dof_damping, nv_);
    rep(p_arm_, base_->dof_armature, nv_);
    rep(p_dinvw_, base_->dof_invweight0, nv_);
    rep(p_mass_, base_->body_mass, nb_);
    rep(p_ipos_, base_->body_ipos, 3 * nb_);
    rep(p_inertia_, base_->body_inertia, 3 * nb_);
    rep(p_iquat_, base_->body_iquat, 4 * nb_);
    rep(p_stm_, base_->body_subtreemass, nb_);
    rep(p_binvw_, base_->body_invweight0, 2 * nb_);
    p_gfric_.resize((size_t)N_ * 6);
    for (int e = 0; e < N_; ++e)
      for (int f = 0; f < 2; ++f)
        std::memcpy(&p_gfric_[(size_t)e * 6 + 3 * f], base_->geom_friction + 3 * foot_geom_[f],
                    sizeof(double) * 3);
    // Initial state = qpos0 (mjwarp put_data from a fresh MjData).
    for (int e = 0; e < N_; ++e) std::memcpy(&qpos_[(size_t)e * nq_], base_->qpos0, sizeof(double) * nq_);

    // BAM per-env.
    z(vin_env_, N_);
    z(vin_gain_, N_);
    fric_scale_.assign(N_, 1.0);
    z(prev_tau_, (size_t)N_ * J);
    z(hist_, (size_t)N_ * HIST * J);
    hist_len_.assign(N_, 0);
    hist_ptr_.assign(N_, 0);
    // Previous-solve stash used by substep 0 (from the end-of-step forward).
    z(st_bias_, (size_t)N_ * J);
    z(st_constr_, (size_t)N_ * J);
    z(st_act_, (size_t)N_ * J);
    z(st_fric_, (size_t)N_ * J);

    rng_.resize(N_);
    for (int e = 0; e < N_; ++e) {
      Rng r{seed * 0x100000001B3ull + (uint64_t)e * 0x9E3779B97F4A7C15ull + 12345};
      r.next();
      rng_[e] = r;
    }

    // Outputs.
    alloc_out(outS_);
    alloc_out(outF_);
    z(found_sub_, (size_t)N_ * DECIM * 2);
    z(heights_, (size_t)N_ * 2);
    z(nan_flag_, N_);
    z(targets_, (size_t)N_ * J);
    set_threads(num_threads);
  }

  ~Batch() {
    free_threads();
    mj_deleteData(scratch_data_);
    mj_deleteModel(scratch_model_);
    mj_deleteModel(base_);
  }

  void set_threads(int n) {
    if (n < 1) throw std::runtime_error("threads >= 1");
    free_threads();
    for (int i = 0; i < n; ++i) {
      mjModel* m = mj_copyModel(nullptr, base_);
      tm_.push_back(m);
      td_.push_back(mj_makeData(m));
    }
    pool_ = new Pool(n);
  }
  int num_threads() const { return pool_ ? pool_->size() : 0; }
  void set_fixed_lag(int lag) { fixed_lag_ = lag; }  // tests only; -1 = random

  // One control step of physics for all envs: DECIM substeps with BAM.
  // targets_ (N,14) must hold default_pos + action (joint-space targets).
  void step() {
    py::gil_scoped_release nogil;
    parallel([&](int tid, int e) { step_env(tm_[tid], td_[tid], e); });
  }

  // mj_forward for all envs + stash BAM prev-solve terms + derived outputs + rays.
  void forward_sense() {
    py::gil_scoped_release nogil;
    parallel([&](int tid, int e) { forward_env(tm_[tid], td_[tid], e); });
  }

  // mjlab calls scene.write_data_to_sim() after _reset_idx whenever any env reset
  // (ML envs/manager_based_rl_env.py:436-440, 364): one extra BAM compute for ALL
  // envs on the post-step / post-reset state. Reset envs have joint targets
  // cleared to 0 (ML entity/data.py:216-225) and zeroed solver outputs
  // (mjwarp reset_data); others use the last mj_step's solve. This appends to the
  // actuator delay buffers and writes ctrl / dof_frictionloss / prev torque.
  void bam_extra(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> reset_mask) {
    auto r = reset_mask.unchecked<1>();
    if (r.shape(0) != N_) throw std::runtime_error("reset_mask shape");
    std::vector<uint8_t> mask(N_);
    for (int e = 0; e < N_; ++e) mask[e] = r(e);
    py::gil_scoped_release nogil;
    parallel([&](int, int e) {
      static const double zeros[J] = {0};
      const bool rs = mask[e] != 0;
      const double* tgt = rs ? zeros : &targets_[(size_t)e * J];
      const double* b = rs ? zeros : &st_bias_[(size_t)e * J];
      const double* c = rs ? zeros : &st_constr_[(size_t)e * J];
      const double* a = rs ? zeros : &st_act_[(size_t)e * J];
      const double* f = rs ? zeros : &st_fric_[(size_t)e * J];
      bam_compute(&qpos_[(size_t)e * nq_], &qvel_[(size_t)e * nv_], &ctrl_[(size_t)e * nu_],
                  &p_fl_[(size_t)e * nv_], &p_damp_[(size_t)e * nv_], e, tgt, b, c, a, f);
    });
  }

  // Actuator reset (BAMJ:382-390; delay buffer reset ML delay_buffer.py reset()).
  void reset_actuators(py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids) {
    auto r = ids.unchecked<1>();
    for (ssize_t k = 0; k < r.shape(0); ++k) {
      int e = (int)r(k);
      check_env(e);
      std::fill(&prev_tau_[(size_t)e * J], &prev_tau_[(size_t)e * J] + J, 0.0);
      std::fill(&hist_[(size_t)e * HIST * J], &hist_[(size_t)e * HIST * J] + HIST * J, 0.0);
      hist_len_[e] = 0;
      hist_ptr_[e] = 0;
    }
  }

  // mj_setConst on the env's inertial/armature patch; writes back only the derived
  // fields mjlab expands per world (ML managers/event_manager.py:53-65).
  void recompute_const(py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids) {
    auto r = ids.unchecked<1>();
    mjModel* m = scratch_model_;
    for (ssize_t k = 0; k < r.shape(0); ++k) {
      int e = (int)r(k);
      check_env(e);
      apply_patch(m, e);
      mj_setConst(m, scratch_data_);
      std::memcpy(&p_stm_[(size_t)e * nb_], m->body_subtreemass, sizeof(double) * nb_);
      std::memcpy(&p_dinvw_[(size_t)e * nv_], m->dof_invweight0, sizeof(double) * nv_);
      std::memcpy(&p_binvw_[(size_t)e * 2 * nb_], m->body_invweight0, sizeof(double) * 2 * nb_);
    }
  }

  py::tuple ray_down(py::array_t<double, py::array::c_style | py::array::forcecast> pts) {
    auto p = pts.unchecked<2>();
    ssize_t n = p.shape(0);
    py::array_t<double> dist(n), nrm({n, (ssize_t)3});
    auto dm = dist.mutable_unchecked<1>();
    auto nm = nrm.mutable_unchecked<2>();
    mjData* d = mj_makeData(base_);
    mj_forward(base_, d);
    for (ssize_t i = 0; i < n; ++i) {
      double q[3] = {p(i, 0), p(i, 1), p(i, 2)}, nn[3];
      dm(i) = grid_.cast_down(base_, d, q, nn);
      for (int k = 0; k < 3; ++k) nm(i, k) = nn[k];
    }
    mj_deleteData(d);
    return py::make_tuple(dist, nrm);
  }

  py::dict arrays(py::object self) {
    py::dict out;
    auto v = [&](const char* name, std::vector<double>& buf, std::vector<ssize_t> shape) {
      out[name] = py::array_t<double>(shape, buf.data(), self);
    };
    ssize_t N = N_;
    v("qpos", qpos_, {N, nq_});
    v("qvel", qvel_, {N, nv_});
    v("qacc_warmstart", qacc_ws_, {N, nv_});
    v("ctrl", ctrl_, {N, nu_});
    v("time", time_, {N});
    v("dof_frictionloss", p_fl_, {N, nv_});
    v("dof_damping", p_damp_, {N, nv_});
    v("dof_armature", p_arm_, {N, nv_});
    v("dof_invweight0", p_dinvw_, {N, nv_});
    v("body_mass", p_mass_, {N, nb_});
    v("body_ipos", p_ipos_, {N, nb_, 3});
    v("body_inertia", p_inertia_, {N, nb_, 3});
    v("body_iquat", p_iquat_, {N, nb_, 4});
    v("body_subtreemass", p_stm_, {N, nb_});
    v("body_invweight0", p_binvw_, {N, nb_, 2});
    v("foot_friction", p_gfric_, {N, 2, 3});
    v("vin_env", vin_env_, {N});
    v("vin_gain", vin_gain_, {N});
    v("friction_scale", fric_scale_, {N});
    v("prev_motor_torque", prev_tau_, {N, J});
    v("targets", targets_, {N, J});
    v("found_sub", found_sub_, {N, DECIM, 2});
    v("heights", heights_, {N, 2});
    v("nan_flag", nan_flag_, {N});
    for (int s = 0; s < 2; ++s) {
      Out& o = s == 0 ? outS_ : outF_;
      std::string pre = s == 0 ? "S_" : "F_";
      auto w = [&](const char* name, std::vector<double>& buf, std::vector<ssize_t> shape) {
        out[py::str(pre + name)] = py::array_t<double>(shape, buf.data(), self);
      };
      w("root_pos", o.root_pos, {N, 3});
      w("root_quat", o.root_quat, {N, 4});
      w("root_linvel_w", o.root_lin, {N, 3});
      w("root_angvel_w", o.root_ang, {N, 3});
      w("foot_linvel_w", o.foot_lin, {N, 2, 3});
      w("foot_pos", o.foot_pos, {N, 2, 3});
      w("angmom", o.angmom, {N, 3});
      w("gyro", o.gyro, {N, 3});
      w("feet_found", o.feet_found, {N, 2});
      w("feet_force", o.feet_force, {N, 2, 3});
      w("self_found", o.self_found, {N});
    }
    return out;
  }

  py::dict indices() const {
    py::dict d;
    d["servo_qadr"] = std::vector<int>(servo_q_, servo_q_ + J);
    d["servo_dadr"] = std::vector<int>(servo_d_, servo_d_ + J);
    d["backlash_qadr"] = std::vector<int>(bl_q_, bl_q_ + J);
    d["backlash_dadr"] = std::vector<int>(bl_d_, bl_d_ + J);
    d["trunk_body"] = trunk_;
    d["foot_geoms"] = std::vector<int>(foot_geom_, foot_geom_ + 2);
    d["foot_sites"] = std::vector<int>(foot_site_, foot_site_ + 2);
    d["grid_cells"] = grid_.nx * grid_.ny;
    d["grid_geoms"] = (int)grid_.geoms.size();
    return d;
  }

 private:
  struct Out {
    std::vector<double> root_pos, root_quat, root_lin, root_ang, foot_lin, foot_pos, angmom, gyro, feet_found,
        feet_force, self_found;
  };

  void alloc_out(Out& o) {
    size_t N = N_;
    o.root_pos.assign(N * 3, 0);
    o.root_quat.assign(N * 4, 0);
    o.root_lin.assign(N * 3, 0);
    o.root_ang.assign(N * 3, 0);
    o.foot_lin.assign(N * 6, 0);
    o.foot_pos.assign(N * 6, 0);
    o.angmom.assign(N * 3, 0);
    o.gyro.assign(N * 3, 0);
    o.feet_found.assign(N * 2, 0);
    o.feet_force.assign(N * 6, 0);
    o.self_found.assign(N, 0);
  }

  void check_env(int e) const {
    if (e < 0 || e >= N_) throw std::out_of_range("env id");
  }

  int id(mjtObj t, const std::string& name) const {
    int i = mj_name2id(base_, t, name.c_str());
    if (i < 0) throw std::runtime_error("name not found: " + name);
    return i;
  }

  void resolve_indices() {
    for (int a = 0; a < J; ++a) {
      if (base_->actuator_trntype[a] != mjTRN_JOINT) throw std::runtime_error("non-joint actuator");
      int jnt = base_->actuator_trnid[2 * a];
      std::string jn = mj_id2name(base_, mjOBJ_JOINT, jnt);
      std::string short_name = jn.substr(jn.find('/') + 1);
      int bl = id(mjOBJ_JOINT, "robot/passive_" + short_name + "_backlash");
      servo_q_[a] = base_->jnt_qposadr[jnt];
      servo_d_[a] = base_->jnt_dofadr[jnt];
      bl_q_[a] = base_->jnt_qposadr[bl];
      bl_d_[a] = base_->jnt_dofadr[bl];
    }
    trunk_ = id(mjOBJ_BODY, "robot/trunk_base");
    foot_geom_[0] = id(mjOBJ_GEOM, "robot/left_foot_collision");
    foot_geom_[1] = id(mjOBJ_GEOM, "robot/right_foot_collision");
    foot_site_[0] = id(mjOBJ_SITE, "robot/left_foot");
    foot_site_[1] = id(mjOBJ_SITE, "robot/right_foot");
    auto sadr = [&](const std::string& n) { return base_->sensor_adr[id(mjOBJ_SENSOR, n)]; };
    gyro_adr_ = sadr("robot/imu_ang_vel");
    angmom_adr_ = sadr("robot/root_angmom");
    found_adr_[0] = sadr("feet_ground_contact_left_foot_collision_found");
    found_adr_[1] = sadr("feet_ground_contact_right_foot_collision_found");
    force_adr_[0] = sadr("feet_ground_contact_left_foot_collision_force");
    force_adr_[1] = sadr("feet_ground_contact_right_foot_collision_force");
    self_adr_ = sadr("self_collision_trunk_base_found");
  }

  void free_threads() {
    delete pool_;
    pool_ = nullptr;
    for (auto d : td_) mj_deleteData(d);
    for (auto m : tm_) mj_deleteModel(m);
    td_.clear();
    tm_.clear();
  }

  void parallel(const std::function<void(int, int)>& f) {
    std::atomic<int> next{0};
    const int chunk = 4;
    pool_->run([&](int tid) {
      for (;;) {
        int s = next.fetch_add(chunk);
        if (s >= N_) break;
        int t = std::min(N_, s + chunk);
        for (int e = s; e < t; ++e) f(tid, e);
      }
    });
  }

  void apply_patch(mjModel* m, int e) const {
    std::memcpy(m->dof_frictionloss, &p_fl_[(size_t)e * nv_], sizeof(double) * nv_);
    std::memcpy(m->dof_damping, &p_damp_[(size_t)e * nv_], sizeof(double) * nv_);
    std::memcpy(m->dof_armature, &p_arm_[(size_t)e * nv_], sizeof(double) * nv_);
    std::memcpy(m->dof_invweight0, &p_dinvw_[(size_t)e * nv_], sizeof(double) * nv_);
    std::memcpy(m->body_mass, &p_mass_[(size_t)e * nb_], sizeof(double) * nb_);
    std::memcpy(m->body_ipos, &p_ipos_[(size_t)e * 3 * nb_], sizeof(double) * 3 * nb_);
    std::memcpy(m->body_inertia, &p_inertia_[(size_t)e * 3 * nb_], sizeof(double) * 3 * nb_);
    std::memcpy(m->body_iquat, &p_iquat_[(size_t)e * 4 * nb_], sizeof(double) * 4 * nb_);
    std::memcpy(m->body_subtreemass, &p_stm_[(size_t)e * nb_], sizeof(double) * nb_);
    std::memcpy(m->body_invweight0, &p_binvw_[(size_t)e * 2 * nb_], sizeof(double) * 2 * nb_);
    for (int f = 0; f < 2; ++f)
      std::memcpy(m->geom_friction + 3 * foot_geom_[f], &p_gfric_[(size_t)e * 6 + 3 * f], sizeof(double) * 3);
  }

  void load_state(mjData* d, int e) const {
    std::memcpy(d->qpos, &qpos_[(size_t)e * nq_], sizeof(double) * nq_);
    std::memcpy(d->qvel, &qvel_[(size_t)e * nv_], sizeof(double) * nv_);
    std::memcpy(d->qacc_warmstart, &qacc_ws_[(size_t)e * nv_], sizeof(double) * nv_);
    std::memcpy(d->ctrl, &ctrl_[(size_t)e * nu_], sizeof(double) * nu_);
    d->time = time_[e];
    mju_zero(d->qfrc_applied, nv_);
    mju_zero(d->xfrc_applied, 6 * nb_);
  }

  void save_state(const mjData* d, int e) {
    std::memcpy(&qpos_[(size_t)e * nq_], d->qpos, sizeof(double) * nq_);
    std::memcpy(&qvel_[(size_t)e * nv_], d->qvel, sizeof(double) * nv_);
    std::memcpy(&qacc_ws_[(size_t)e * nv_], d->qacc_warmstart, sizeof(double) * nv_);
    std::memcpy(&ctrl_[(size_t)e * nu_], d->ctrl, sizeof(double) * nu_);
    time_[e] = d->time;
  }

  // Scatter of DOF-friction constraint forces at the servo DOFs (BAMJ:541-574).
  void servo_friction_force(const mjModel* m, const mjData* d, double out[J]) const {
    std::fill(out, out + J, 0.0);
    // dof -> servo index (only 14 entries; linear search is cheap).
    for (int i = 0; i < d->nefc; ++i) {
      if (d->efc_type[i] != mjCNSTR_FRICTION_DOF) continue;
      int dof = d->efc_id[i];
      for (int j = 0; j < J; ++j)
        if (servo_d_[j] == dof) {
          out[j] += d->efc_force[i];
          break;
        }
    }
  }

  void stash_prev_solve(const mjModel* m, const mjData* d, double* bias, double* constr, double* act,
                        double* fric) const {
    for (int j = 0; j < J; ++j) {
      bias[j] = d->qfrc_bias[servo_d_[j]];
      constr[j] = d->qfrc_constraint[servo_d_[j]];
      act[j] = d->qfrc_actuator[servo_d_[j]];
    }
    servo_friction_force(m, d, fric);
  }

  // BAM compute() for one env at one substep (BAMJ:576-673, ACT:28-104, BAMA:220-292).
  // qpos/qvel/ctrl/fl/damp are full-length (nq/nv/nu/nv) arrays of either an mjData /
  // mjModel or the per-env state/patch buffers. tgt = undelayed joint target (14).
  void bam_compute(const double* qpos, const double* qvel, double* ctrl, double* fl_out, double* damp_out, int e,
                   const double* tgt, const double* bias, const double* constr, const double* act,
                   const double* fric) {
    // apply_delay: append the (undelayed) joint target, then read the lagged one.
    double* h = &hist_[(size_t)e * HIST * J];
    int& len = hist_len_[e];
    int& ptr = hist_ptr_[e];
    if (len == 0) {  // first append after reset: backfill every slot
      for (int k = 0; k < HIST; ++k) std::memcpy(h + k * J, tgt, sizeof(double) * J);
      ptr = 0;
    } else {
      ptr = (ptr + 1) % HIST;
      std::memcpy(h + ptr * J, tgt, sizeof(double) * J);
    }
    len = std::min(len + 1, HIST);
    int lag = fixed_lag_ >= 0 ? fixed_lag_ : rng_[e].randint(LAG_MIN, LAG_MAX);
    lag = std::max(0, std::min(lag, len - 1));
    const double* delayed = h + ((ptr - lag + HIST) % HIST) * J;

    double* ptau = &prev_tau_[(size_t)e * J];
    double load = 0;
    for (int j = 0; j < J; ++j) load += std::fabs(ptau[j]);
    double vin = vin_env_[e] - vin_gain_[e] * load;
    vin = std::max(vin, VIN_MIN);
    const double fs = fric_scale_[e];
    for (int j = 0; j < J; ++j) {
      const double pos = qpos[servo_q_[j]] + qpos[bl_q_[j]];  // encoder through backlash (ACT:64-104)
      const double vel = qvel[servo_d_[j]];                    // motor-side velocity
      double duty = (delayed[j] - pos) * KP_FW * ERROR_GAIN;
      if (current_limit_) {
        const double span = RES * MAX_CURRENT / vin;
        const double center = KT * vel / vin;
        duty = std::min(std::max(duty, center - span), center + span);
      }
      duty = std::min(std::max(duty, -MAX_PWM), MAX_PWM);
      const double volts = vin * duty;
      const double tau = KT * volts / RES - KT * KT * vel / RES;

      const double ext = -bias[j] + constr[j] - fric[j];
      const double tm = act[j];
      const double s = std::exp(-std::pow(std::fabs(vel) / DTHETA_S, ALPHA_S));
      double fl = F_BASE + s * F_STRIBECK;
      fl += std::fabs(ext * LF_EXT - tm * LF_MOTOR);
      fl += s * std::fabs(ext * LF_EXT_S - tm * LF_MOTOR_S);
      const double quad = (std::fabs(tm) > std::fabs(ext)) ? LF_EXT_Q * ext * ext : LF_MOTOR_Q * tm * tm;
      fl += s * quad;
      fl *= fs;
      fl_out[servo_d_[j]] = fl;
      damp_out[servo_d_[j]] = F_VISCOUS;
      ctrl[j] = tau;
      ptau[j] = tau;
    }
  }

  static bool bad_state(const mjData* d) {
    return d->warning[mjWARN_BADQPOS].number > 0 || d->warning[mjWARN_BADQVEL].number > 0 ||
           d->warning[mjWARN_BADQACC].number > 0 || d->warning[mjWARN_BADCTRL].number > 0;
  }

  void extract(const mjModel* m, const mjData* d, int e, Out& o) const {
    const int b = trunk_;
    const double* xp = d->xpos + 3 * b;
    const double* com = d->subtree_com + 3 * b;  // ML entity/data.py:252-260
    const double* cv = d->cvel + 6 * b;
    double off[3] = {com[0] - xp[0], com[1] - xp[1], com[2] - xp[2]}, cr[3];
    mju_cross(cr, cv, off);
    for (int k = 0; k < 3; ++k) {
      o.root_pos[3 * e + k] = xp[k];
      o.root_lin[3 * e + k] = cv[3 + k] - cr[k];
      o.root_ang[3 * e + k] = cv[k];
    }
    for (int k = 0; k < 4; ++k) o.root_quat[4 * e + k] = d->xquat[4 * b + k];
    for (int f = 0; f < 2; ++f) {
      int s = foot_site_[f];
      const double* sp = d->site_xpos + 3 * s;
      const double* scv = d->cvel + 6 * m->site_bodyid[s];
      double so[3] = {com[0] - sp[0], com[1] - sp[1], com[2] - sp[2]}, sc[3];
      mju_cross(sc, scv, so);
      for (int k = 0; k < 3; ++k) {
        o.foot_lin[6 * e + 3 * f + k] = scv[3 + k] - sc[k];
        o.foot_pos[6 * e + 3 * f + k] = sp[k];
        o.feet_force[6 * e + 3 * f + k] = d->sensordata[force_adr_[f] + k];
      }
      o.feet_found[2 * e + f] = d->sensordata[found_adr_[f]];
    }
    for (int k = 0; k < 3; ++k) {
      o.angmom[3 * e + k] = d->sensordata[angmom_adr_ + k];
      o.gyro[3 * e + k] = d->sensordata[gyro_adr_ + k];
    }
    o.self_found[e] = d->sensordata[self_adr_];
  }

  void step_env(mjModel* m, mjData* d, int e) {
    apply_patch(m, e);
    load_state(d, e);
    nan_flag_[e] = 0;
    double bias[J], constr[J], act[J], fric[J];
    for (int k = 0; k < DECIM; ++k) {
      const double* tgt = &targets_[(size_t)e * J];
      if (k == 0) {
        bam_compute(d->qpos, d->qvel, d->ctrl, m->dof_frictionloss, m->dof_damping, e, tgt,
                    &st_bias_[(size_t)e * J], &st_constr_[(size_t)e * J], &st_act_[(size_t)e * J],
                    &st_fric_[(size_t)e * J]);
      } else {
        stash_prev_solve(m, d, bias, constr, act, fric);
        bam_compute(d->qpos, d->qvel, d->ctrl, m->dof_frictionloss, m->dof_damping, e, tgt, bias, constr, act,
                    fric);
      }
      for (int w = 0; w < mjNWARNING; ++w) d->warning[w].number = 0;
      mj_step(m, d);
      if (bad_state(d)) {
        nan_flag_[e] = 1;
      }
      for (int f = 0; f < 2; ++f) found_sub_[((size_t)e * DECIM + k) * 2 + f] = d->sensordata[found_adr_[f]];
    }
    extract(m, d, e, outS_);
    // Keep the last solve's terms: the post-reset extra BAM pass reads them.
    stash_prev_solve(m, d, &st_bias_[(size_t)e * J], &st_constr_[(size_t)e * J], &st_act_[(size_t)e * J],
                     &st_fric_[(size_t)e * J]);
    save_state(d, e);
    std::memcpy(&p_fl_[(size_t)e * nv_], m->dof_frictionloss, sizeof(double) * nv_);
    std::memcpy(&p_damp_[(size_t)e * nv_], m->dof_damping, sizeof(double) * nv_);
    if (nan_flag_[e]) poison(e);
  }

  // MuJoCo C auto-resets diverged data (mj_checkPos/Vel/Acc); MJWarp does not and
  // the NaN reaches mjlab's nan_state termination. Reproduce that by poisoning.
  void poison(int e) {
    const double nan = std::nan("");
    std::fill(&qpos_[(size_t)e * nq_], &qpos_[(size_t)e * nq_] + nq_, nan);
    std::fill(&qvel_[(size_t)e * nv_], &qvel_[(size_t)e * nv_] + nv_, nan);
    for (int k = 0; k < 3; ++k) outS_.root_pos[3 * e + k] = nan;
  }

  void forward_env(mjModel* m, mjData* d, int e) {
    apply_patch(m, e);
    load_state(d, e);
    mj_forward(m, d);
    stash_prev_solve(m, d, &st_bias_[(size_t)e * J], &st_constr_[(size_t)e * J], &st_act_[(size_t)e * J],
                     &st_fric_[(size_t)e * J]);
    extract(m, d, e, outF_);
    save_state(d, e);
    // foot_height_scan (CFG:2150-2186; ML sensor/raycast_sensor.py:691-807,
    // sensor/terrain_height_sensor.py:57-90): per foot, center + ring(r=0.04, n=2),
    // yaw-aligned, straight down, max_distance 1, reduction min.
    const double radius = 0.04, max_dist = 1.0;
    for (int f = 0; f < 2; ++f) {
      int s = foot_site_[f];
      const double* sp = d->site_xpos + 3 * s;
      const double* R = d->site_xmat + 9 * s;
      // yaw extraction (raycast_sensor.py:828-874): x axis = first column.
      double xx = R[0], xy = R[3];
      double nrm = std::sqrt(xx * xx + xy * xy);
      if (nrm < 0.1) {  // singular: use y axis rotated -90 deg
        double yx = R[1], yy = R[4];
        double yn = std::max(std::sqrt(yx * yx + yy * yy), 1e-6);
        xx = yy / yn;
        xy = -yx / yn;
        nrm = 1.0;
      }
      nrm = std::max(nrm, 1e-6);
      xx /= nrm;
      xy /= nrm;
      const double offs[3][2] = {{0.0, 0.0}, {radius, 0.0}, {radius * std::cos(M_PI), radius * std::sin(M_PI)}};
      double h[3];
      bool miss[3];
      for (int r = 0; r < 3; ++r) {
        double p[3] = {sp[0] + xx * offs[r][0] - xy * offs[r][1], sp[1] + xy * offs[r][0] + xx * offs[r][1],
                       sp[2]};
        double n3[3];
        double dist = grid_.cast_down(m, d, p, n3);
        if (dist > max_dist) dist = -1;
        miss[r] = dist < 0;
        if (miss[r]) {
          h[r] = 0;
        } else {
          double hit_z = p[2] - dist;
          h[r] = sp[2] - hit_z;
          if (n3[2] < 0) h[r] = 0;  // backface
        }
      }
      bool all_miss = miss[0] && miss[1] && miss[2];
      double fallback = std::min(std::max(sp[2], 0.0), max_dist);
      double best = 1e300;
      for (int r = 0; r < 3; ++r) {
        double v = miss[r] ? (all_miss ? fallback : max_dist) : h[r];
        best = std::min(best, v);
      }
      heights_[2 * e + f] = best;
    }
  }

  int N_, nq_, nv_, nb_, nu_;
  bool current_limit_;
  int fixed_lag_ = -1;
  mjModel* base_ = nullptr;
  mjModel* scratch_model_ = nullptr;
  mjData* scratch_data_ = nullptr;
  std::vector<mjModel*> tm_;
  std::vector<mjData*> td_;
  Pool* pool_ = nullptr;
  RayGrid grid_;

  int servo_q_[J], servo_d_[J], bl_q_[J], bl_d_[J];
  int trunk_, foot_geom_[2], foot_site_[2];
  int gyro_adr_, angmom_adr_, found_adr_[2], force_adr_[2], self_adr_;

  std::vector<double> qpos_, qvel_, qacc_ws_, ctrl_, time_;
  std::vector<double> p_fl_, p_damp_, p_arm_, p_dinvw_, p_mass_, p_ipos_, p_inertia_, p_iquat_, p_stm_, p_binvw_,
      p_gfric_;
  std::vector<double> vin_env_, vin_gain_, fric_scale_, prev_tau_, hist_;
  std::vector<int> hist_len_, hist_ptr_;
  std::vector<double> st_bias_, st_constr_, st_act_, st_fric_;
  std::vector<Rng> rng_;
  Out outS_, outF_;
  std::vector<double> found_sub_, heights_, nan_flag_, targets_;
};

PYBIND11_MODULE(_batch, mod) {
  py::class_<Batch>(mod, "Batch")
      .def(py::init<uintptr_t, int, int, uint64_t, bool>(), py::arg("model_addr"), py::arg("num_envs"),
           py::arg("num_threads"), py::arg("seed"), py::arg("current_limit"))
      .def("set_threads", &Batch::set_threads)
      .def("set_fixed_lag", &Batch::set_fixed_lag)
      .def_property_readonly("num_threads", &Batch::num_threads)
      .def("step", &Batch::step)
      .def("forward_sense", &Batch::forward_sense)
      .def("reset_actuators", &Batch::reset_actuators)
      .def("bam_extra", &Batch::bam_extra)
      .def("recompute_const", &Batch::recompute_const)
      .def("ray_down", &Batch::ray_down)
      .def("indices", &Batch::indices)
      .def("arrays", [](py::object self) { return self.cast<Batch&>().arrays(self); });
}
