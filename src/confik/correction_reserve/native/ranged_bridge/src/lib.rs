//! Ownership-safe FFI over pinned official RangedIK objectives and kinematics.
//!
//! The native upstream wrapper does not expose status or accepted motion history.
//! This bridge uses its exact PANOC defaults, but a per-call hard Rectangle and
//! explicit actual accepted/held command history. No candidate is committed here.

use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector6};
use optimization_engine::{constraints::Rectangle, panoc::*, *};
use optimization_engine::core::ExitStatus;
use relaxed_ik_lib::groove::objective_master::ObjectiveMaster;
use relaxed_ik_lib::groove::vars::RelaxedIKVars;
use relaxed_ik_lib::spacetime::robot::Robot;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

const MAX_ITER: usize = 100;
const PANOC_TOLERANCE: f64 = 0.0005;

thread_local! {
    static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn error(message: &str) {
    ERROR.with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
}

#[no_mangle]
pub extern "C" fn crik_ranged_error() -> *const c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}

pub struct RangedBridge {
    vars: RelaxedIKVars,
    objective: ObjectiveMaster,
    cache: PANOCCache,
}

#[repr(C)]
pub struct SolveStats {
    pub code: c_int,
    pub iterations: u64,
    pub objective_calls: u64,
    pub gradient_calls: u64,
    pub setup_ns: u64,
    pub solve_ns: u64,
    pub cost: f64,
    pub fpr_norm: f64,
}

/// `initial`, `urdf`, and link strings are borrowed only for this call.
#[no_mangle]
pub unsafe extern "C" fn crik_ranged_new(
    urdf: *const c_char, base: *const c_char, end: *const c_char,
    n: usize, initial: *const f64, collision_objective: c_int,
) -> *mut RangedBridge {
    let result = catch_unwind(AssertUnwindSafe(|| {
        assert!(!urdf.is_null() && !base.is_null() && !end.is_null() && !initial.is_null());
        let robot = Robot::from_urdf(
            CStr::from_ptr(urdf).to_str().unwrap(),
            &[CStr::from_ptr(base).to_str().unwrap().to_owned()],
            &[CStr::from_ptr(end).to_str().unwrap().to_owned()],
        );
        assert_eq!(robot.num_dofs, n, "official and common joint counts differ");
        let q = std::slice::from_raw_parts(initial, n).to_vec();
        assert!(q.iter().all(|x| x.is_finite()));
        let pose = robot.get_ee_pos_and_quat_immutable(&q);
        let positions = vec![pose[0].0];
        let quaternions = vec![pose[0].1];
        assert!(collision_objective == 0 || collision_objective == 1);
        let mut objective = ObjectiveMaster::relaxed_ik(&robot.chain_lengths);
        // The pinned official constructor appends SelfCollision terms after all
        // pose/joint-limit/motion/manipulability terms. Disabling that exact tail
        // avoids evaluating out-of-contract collision penalties (rather than
        // just multiplying them by zero). Every retained weight is unchanged.
        let retained = 6 * robot.num_chains + robot.num_dofs + 4;
        let collision_count: usize = robot.chain_lengths.iter()
            .map(|length| (length-1)*(length-2)/2).sum();
        assert_eq!(objective.objectives.len(), retained + collision_count);
        assert_eq!(objective.weight_priors.len(), retained + collision_count);
        if collision_objective == 0 {
            objective.objectives.truncate(retained);
            objective.weight_priors.truncate(retained);
        }
        let vars = RelaxedIKVars {
            robot, init_state: q.clone(), xopt: q.clone(), prev_state: q.clone(),
            prev_state2: q.clone(), prev_state3: q,
            goal_positions: positions.clone(), goal_quats: quaternions.clone(),
            tolerances: vec![Vector6::zeros()],
            init_ee_positions: positions, init_ee_quats: quaternions,
        };
        Box::into_raw(Box::new(RangedBridge {
            vars, objective, cache: PANOCCache::new(n, 1e-14, 10),
        }))
    }));
    match result {
        Ok(ptr) => ptr,
        Err(_) => { error("RangedIK constructor failed; inspect URDF and chain"); std::ptr::null_mut() }
    }
}

/// Actual active weights, for an implementation check without any optimization.
#[no_mangle]
pub unsafe extern "C" fn crik_ranged_weights(
    ptr: *mut RangedBridge, capacity: usize, out: *mut f64,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        assert!(!ptr.is_null() && !out.is_null());
        let bridge = &*ptr;
        let weights = &bridge.objective.weight_priors;
        assert!(capacity >= weights.len());
        std::slice::from_raw_parts_mut(out, capacity)[..weights.len()].copy_from_slice(weights);
        weights.len() as c_int
    }));
    match result {
        Ok(count) => count,
        Err(_) => { error("RangedIK active weight query failed"); -1 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn crik_ranged_free(ptr: *mut RangedBridge) {
    if !ptr.is_null() { drop(Box::from_raw(ptr)); }
}

#[no_mangle]
pub unsafe extern "C" fn crik_ranged_reset(ptr: *mut RangedBridge, n: usize, q: *const f64) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        assert!(!ptr.is_null() && !q.is_null());
        let bridge = &mut *ptr;
        assert_eq!(n, bridge.vars.robot.num_dofs);
        bridge.vars.reset(std::slice::from_raw_parts(q, n).to_vec());
        bridge.cache = PANOCCache::new(n, 1e-14, 10);
    }));
    if result.is_ok() { 0 } else { error("RangedIK reset failed"); -1 }
}

#[no_mangle]
pub unsafe extern "C" fn crik_ranged_fk(
    ptr: *mut RangedBridge, n: usize, q: *const f64, out: *mut f64,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        assert!(!ptr.is_null() && !q.is_null() && !out.is_null());
        let bridge = &*ptr;
        assert_eq!(n, bridge.vars.robot.num_dofs);
        let pose = bridge.vars.robot.get_ee_pos_and_quat_immutable(std::slice::from_raw_parts(q, n));
        let destination = std::slice::from_raw_parts_mut(out, 7);
        destination[..3].copy_from_slice(pose[0].0.as_slice());
        destination[3..].copy_from_slice(pose[0].1.quaternion().coords.as_slice());
    }));
    if result.is_ok() { 0 } else { error("RangedIK FK failed"); -1 }
}

/// Probe the six official unweighted Cartesian loss terms without a solve.
#[no_mangle]
pub unsafe extern "C" fn crik_ranged_components(
    ptr: *mut RangedBridge, n: usize, q: *const f64, position: *const f64,
    quaternion: *const f64, tolerances: *const f64, out: *mut f64,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        assert!(!ptr.is_null() && !q.is_null() && !out.is_null());
        assert!(!position.is_null() && !quaternion.is_null() && !tolerances.is_null());
        let bridge = &mut *ptr;
        assert_eq!(n, bridge.vars.robot.num_dofs);
        let p = std::slice::from_raw_parts(position, 3);
        let quat = std::slice::from_raw_parts(quaternion, 4);
        bridge.vars.goal_positions[0] = Vector3::from_row_slice(p);
        bridge.vars.goal_quats[0] = UnitQuaternion::from_quaternion(Quaternion::new(quat[3],quat[0],quat[1],quat[2]));
        bridge.vars.tolerances[0] = Vector6::from_row_slice(std::slice::from_raw_parts(tolerances, 6));
        let joints = std::slice::from_raw_parts(q, n);
        let frames = bridge.vars.robot.get_frames_immutable(joints);
        let destination = std::slice::from_raw_parts_mut(out, 6);
        for i in 0..6 {
            destination[i] = bridge.objective.objectives[i].call(joints, &bridge.vars, &frames);
        }
    }));
    if result.is_ok() { 0 } else { error("RangedIK component probe failed"); -1 }
}

/// Explicit history is [q_(t-1), q_(t-2), q_(t-3), q_(t-4)] including held frames.
/// Returned q is caller-owned, avoiding the upstream wrapper's leaked Vec.
#[no_mangle]
pub unsafe extern "C" fn crik_ranged_solve(
    ptr: *mut RangedBridge, n: usize, position: *const f64, quaternion: *const f64,
    tolerances: *const f64, history: *const f64, lower: *const f64, upper: *const f64,
    max_duration_ns: u64, out: *mut f64, stats: *mut SolveStats,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let start = Instant::now();
        assert!(!ptr.is_null() && !position.is_null() && !quaternion.is_null());
        assert!(!tolerances.is_null() && !history.is_null() && !lower.is_null());
        assert!(!upper.is_null() && !out.is_null() && !stats.is_null());
        let bridge = &mut *ptr;
        assert_eq!(n, bridge.vars.robot.num_dofs);
        let p = std::slice::from_raw_parts(position, 3);
        let quat = std::slice::from_raw_parts(quaternion, 4);
        let tol = std::slice::from_raw_parts(tolerances, 6);
        let hist = std::slice::from_raw_parts(history, 4*n);
        let lb = std::slice::from_raw_parts(lower, n);
        let ub = std::slice::from_raw_parts(upper, n);
        let q = std::slice::from_raw_parts_mut(out, n);
        assert!(lb.iter().zip(ub).all(|(a,b)| a.is_finite() && b.is_finite() && a <= b));
        bridge.vars.goal_positions[0] = Vector3::new(p[0], p[1], p[2]);
        bridge.vars.goal_quats[0] = UnitQuaternion::from_quaternion(Quaternion::new(quat[3],quat[0],quat[1],quat[2]));
        bridge.vars.tolerances[0] = Vector6::from_row_slice(tol);
        bridge.vars.xopt.copy_from_slice(&hist[..n]);
        bridge.vars.prev_state.copy_from_slice(&hist[n..2*n]);
        bridge.vars.prev_state2.copy_from_slice(&hist[2*n..3*n]);
        bridge.vars.prev_state3.copy_from_slice(&hist[3*n..4*n]);
        for i in 0..n { q[i] = hist[i].max(lb[i]).min(ub[i]); }

        let objective_calls = Cell::new(0u64);
        let gradient_calls = Cell::new(0u64);
        let vars = &bridge.vars;
        let objective = &bridge.objective;
        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            gradient_calls.set(gradient_calls.get()+1);
            let (_, derivative) = objective.gradient(u, vars);
            grad.copy_from_slice(&derivative);
            Ok(())
        };
        let f = |u: &[f64], cost: &mut f64| -> Result<(), SolverError> {
            objective_calls.set(objective_calls.get()+1);
            *cost = objective.call(u, vars);
            Ok(())
        };
        // Static URDF limits remain in the official objective. Only this hard
        // Rectangle is narrowed to the actual previous command's rate interval.
        let bounds = Rectangle::new(Some(lb), Some(ub));
        let problem = Problem::new(&bounds, df, f);
        let mut optimizer = PANOCOptimizer::new(problem, &mut bridge.cache)
            .with_max_iter(MAX_ITER).with_tolerance(PANOC_TOLERANCE);
        if max_duration_ns > 0 {
            optimizer = optimizer.with_max_duration(std::time::Duration::from_nanos(max_duration_ns));
        }
        let setup_ns = start.elapsed().as_nanos() as u64;
        let solve_start = Instant::now();
        let status = optimizer.solve(q);
        let solve_ns = solve_start.elapsed().as_nanos() as u64;
        let destination = &mut *stats;
        destination.objective_calls = objective_calls.get();
        destination.gradient_calls = gradient_calls.get();
        destination.setup_ns = setup_ns;
        destination.solve_ns = solve_ns;
        match status {
            Ok(s) => {
                destination.code = match s.exit_status() {
                    ExitStatus::Converged => 0,
                    ExitStatus::NotConvergedIterations => 1,
                    ExitStatus::NotConvergedOutOfTime => 2,
                };
                destination.iterations = s.iterations() as u64;
                destination.cost = s.cost_value();
                destination.fpr_norm = s.norm_fpr();
            },
            Err(e) => {
                destination.code = -1;
                destination.iterations = 0;
                destination.cost = f64::NAN;
                destination.fpr_norm = f64::NAN;
                error(&format!("PANOC error: {:?}", e));
            },
        }
        destination.code
    }));
    match result {
        Ok(code) => code,
        Err(_) => { error("RangedIK solve bridge failed"); -999 }
    }
}
