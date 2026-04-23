from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

from multi_target_env import MultiTargetEnv

# ----------------------
# Continuous-time dynamics and Jacobians
# ----------------------

def f_cv_cont(x, omega=0):
    # state: [px,py,vx,vy]
    px, py, vx, vy = x
    return np.array([vx, vy, 0.0, 0.0])


def Fx_cv_cont(x, omega=0):
    Fx = np.zeros((4,4))
    Fx[0,2] = 1.0
    Fx[1,3] = 1.0
    return Fx

@staticmethod
def f_ct(x):
    # state: [px,py,vx,vy,omega]
    px, py, vx, vy, omega = x
    return np.array([vx, vy, -omega * vy, omega * vx, 0.0])


def Fx_ct(x):
    px, py, vx, vy, omega = x
    Fx = np.zeros((5,5))
    Fx[0,2] = 1.0
    Fx[1,3] = 1.0
    Fx[2,3] = -omega
    Fx[2,4] = -vy
    Fx[3,2] = omega
    Fx[3,4] = vx
    return Fx


def f_ct_linear(x, omega):
    # state: [px,py,vx,vy,omega]
    px, py, vx, vy = x
    return np.array([vx, vy, -omega * vy, omega * vx])


def Fx_ct_linear(x, omega):
    px, py, vx, vy = x
    Fx = np.zeros((4,4))
    Fx[0,2] = 1.0
    Fx[1,3] = 1.0
    Fx[2,3] = -omega
    Fx[3,2] = omega
    return Fx

#  propagate (state + STM) 
def propagate_state_and_stm(t0, t1, x0, Phi0_vec, f_dyn, Fx_dyn, n, omega=0,
                            rtol=1e-6, atol=1e-8):

    def dyn_with_stm(t, z):
        x = z[:n]
        Phi = z[n:].reshape((n, n))
        
        dx = f_dyn(x, omega)
        Fx = Fx_dyn(x, omega)
        dPhi = Fx @ Phi
        return np.concatenate([dx, dPhi.reshape(-1)])

    z0 = np.concatenate([x0, Phi0_vec])
    sol = solve_ivp(dyn_with_stm, (t0, t1), z0,
                    rtol=rtol, atol=atol, dense_output=False)

    if not sol.success:
        raise RuntimeError("STM/state propagation failed.")

    z_end = sol.y[:, -1]
    return z_end[:n], z_end[n:].reshape((n, n))

# ----------------------
# Batch estimator for one target (Batch LS)
# ----------------------
def batch_estimate_single_target(
    t_obs, y_obs,
    x0_ref,
    P0,
    R,
    model,
    omega=0,
    f_cv_cont=f_cv_cont, Fx_cv_cont=Fx_cv_cont,
    f_ct_cont=f_ct_linear, Fx_ct_cont=Fx_ct_linear,
    rtol=1e-6,
    atol=1e-8,
    obsFunc=None
):
    """
    Batch estimator, with CV/CT selection.

    model: "CV" or "CT"
    f_dyn / Fx_dyn chosen accordingly
    """

    # choose either linear or rotational model
    n = 4
    if model.upper() == "CV":
        
        f_dyn = lambda x, omega=None: f_cv_cont(x)
        Fx_dyn = lambda x, omega=None: Fx_cv_cont(x)
    elif model.upper() == "CT":

        f_dyn = lambda x, omega: f_ct_cont(x, omega)
        Fx_dyn = lambda x, omega: Fx_ct_cont(x, omega)
    else:
        raise ValueError("Unknown model: choose 'CV' or 'CT'")

    # extract time steps and observations
    t_obs = np.asarray(t_obs)
    y_obs = np.asarray(y_obs)
    L = len(t_obs)

    Xref = np.zeros((n, L))
    resids = np.zeros((2, L))

    # Build batch matrices
    P0 = P0[:4,:4]
    Cinv = np.linalg.inv(P0)
    Rinv = np.linalg.inv(R)

    Lambda = Cinv.copy()
    Nvec = np.zeros(n)

    # initial STM & state
    Phi = np.eye(n)
    x = x0_ref.copy()

    Xref[:, 0] = x[:4]

    # build Λ and N
    for k in range(L):

        if k > 0:
            x, Phi = propagate_state_and_stm(
                t_obs[k - 1], t_obs[k],
                x, Phi.reshape(-1),
                f_dyn, Fx_dyn, n, omega,
                rtol, atol
            )

        Xref[:, k] = x[:4]

        # measurement model
        Htil, Gk = obsFunc(x)
        yk = y_obs[k] - Gk
        resids[:, k] = yk

        # measurement Jacobian wrt initial state
        Hk = Htil @ Phi

        # batch accumulation
        Lambda += Hk.T @ Rinv @ Hk
        Nvec += Hk.T @ Rinv @ yk

    # Solve normal equations Λ x = N
    try:
        x0_correction = np.linalg.solve(Lambda, Nvec)
    except np.linalg.LinAlgError:
        x0_correction = np.linalg.pinv(Lambda) @ Nvec

    x0_new = x0_ref[:4] + x0_correction

    # posterior covariance of initial state
    try:
        P0_post = np.linalg.inv(Lambda)
    except np.linalg.LinAlgError:
        P0_post = np.linalg.pinv(Lambda)

    # propagate state and stm
    Xest = np.zeros((n, L))
    P_mat = np.zeros((n, n, L))

    Phi = np.eye(n)
    x = x0_new.copy()

    Xest[:, 0] = x
    P_mat[:, :, 0] = P0_post

    for k in range(1, L):
        x, Phi = propagate_state_and_stm(
            t_obs[k - 1], t_obs[k],
            x, Phi.reshape(-1),
            f_dyn, Fx_dyn, n,
            rtol, atol
        )
        Xest[:, k] = x
        P_mat[:, :, k] = Phi @ P0_post @ Phi.T

    return {
        "model": model,
        "x0_new": x0_new,
        "P0_post": P0_post,
        "Xref": Xref,
        "Xest": Xest,
        "P_mat": P_mat,
        "resids": resids,
        "Lambda": Lambda,
        "Nvec": Nvec,
        "omega": omega
    }

# Residuals for CV: params = [px0, py0, vx0, vy0]
def residuals_cv(params, t_obs, y_obs, R_sqrt_inv=None):
    x0 = np.asarray(params)
    # Integrate CV dynamics
    states = integrate_and_sample(f_cv_cont, x0, t_obs)  # shape (L,4)
    thetas = []
    rs = []
    for s in states:
        H, Gk = MultiTargetEnv.extract_measurement_bearingRange(s)
        thetas.append(Gk[0])
        rs.append(Gk[1])
    thetas = np.array(thetas)
    rs = np.array(rs)

    # angle residuals wrapped
    th_res = angle_diff(thetas, y_obs[:,0])
    r_res = rs - y_obs[:,1]

    # Interleave [th0, r0, th1, r1, ...]
    res = np.empty(2 * len(t_obs))
    res[0::2] = th_res
    res[1::2] = r_res

    if R_sqrt_inv is not None:
        # whiten residuals if provided measurement covariance sqrt-inverse
        # R_sqrt_inv is 2x2 such that R_sqrt_inv @ residual_2 = whitened residual
        res_whitened = np.empty_like(res)
        for i in range(len(t_obs)):
            r2 = res[2*i:2*i+2]
            w2 = R_sqrt_inv @ r2
            res_whitened[2*i:2*i+2] = w2
        return res_whitened
    return res


# CT (no-heading) dynamics (state: [px,py,vx,vy,omega], n=5)
def f_ct(x):
    px, py, vx, vy, omega = x
    return np.array([vx, vy, -omega * vy, omega * vx, 0.0])

# ------------------------
# helper: measurement prediction for a state vector (4D px,py,vx,vy)
def predict_measurement_from_state(x4):
    H, Gk = MultiTargetEnv.extract_measurement_bearingRange(x4)
    return Gk

# ------------------------
# helper: angle residual wrap to [-pi, pi]
def angle_diff(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))

# ------------------------
# integrate dynamics once for a candidate initial state and sample at t_obs
def integrate_and_sample(f_dyn, x0, t_obs, rtol=1e-9, atol=1e-9):
    # integrate from t0 = t_obs[0] to t_obs[-1] and evaluate at t_obs
    t0 = float(t_obs[0])
    tspan = (t0, float(t_obs[-1]))
    # shift times to absolute by integrating from t0; inside solver we return states at absolute times
    def fun(tt, zz):
        return f_dyn(zz)

    # Use solve_ivp with t_eval = t_obs for direct sampling
    sol = solve_ivp(fun, tspan, x0, t_eval=np.asarray(t_obs), rtol=rtol, atol=atol)
    if not sol.success:
        # fallback: try dense output and sample
        sol = solve_ivp(fun, tspan, x0, dense_output=True, rtol=rtol, atol=atol)
        states = sol.sol(t_obs).T
    else:
        states = sol.y.T  # shape (len(t_obs), n)
    return states
# ------------------------
# Residuals for CT: params = [px0, py0, vx0, vy0, omega]
def residuals_ct(params, t_obs, y_obs, R_sqrt_inv=None):
    x0 = np.asarray(params)
    states = integrate_and_sample(f_ct, x0, t_obs)  # shape (L,5)
    thetas = []
    rs = []
    for s in states:
        # pass only first 4 entries to measurement function
        H, Gk = MultiTargetEnv.extract_measurement_bearingRange(s[:4])
        thetas.append(Gk[0])
        rs.append(Gk[1])
    thetas = np.array(thetas)
    rs = np.array(rs)

    th_res = angle_diff(thetas, y_obs[:,0])
    r_res = rs - y_obs[:,1]

    res = np.empty(2 * len(t_obs))
    res[0::2] = th_res
    res[1::2] = r_res

    if R_sqrt_inv is not None:
        res_whitened = np.empty_like(res)
        for i in range(len(t_obs)):
            r2 = res[2*i:2*i+2]
            w2 = R_sqrt_inv @ r2
            res_whitened[2*i:2*i+2] = w2
        return res_whitened
    return res

# fit functions that call least_squares
def fit_initial_state_cv(t_obs, y_obs, R=None, x0_guess=None, bounds=None, verbose=0):
    """
    Fit initial CV state [px0,py0,vx0,vy0] from measurements y_obs at times t_obs.
    """
    if x0_guess is None:
        # rough guess from first measurement (range+bearing -> px,py) and zero velocity
        th0 = y_obs[0,0]; r0 = y_obs[0,1]
        px0 = r0 * np.cos(th0); py0 = r0 * np.sin(th0)
        x0_guess = np.array([px0, py0, 0.0, 0.0])
    if R is not None:
        # compute whitening: R = cov, get inv sqrt via eigen or cholesky
        try:
            L = np.linalg.cholesky(R)
            R_sqrt_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # fallback using sqrt of diagonal
            R_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(R)))
    else:
        R_sqrt_inv = None

    res = least_squares(residuals_cv, x0_guess, args=(t_obs, y_obs, R_sqrt_inv), bounds=bounds or (-np.inf, np.inf), verbose=verbose)
    return res.x, res

def fit_initial_state_ct(t_obs, y_obs, R=None, x0_guess=None, bounds=None, verbose=0):
    """
    Fit initial CT state [px0,py0,vx0,vy0,omega] from measurements.
    """
    if x0_guess is None:
        th0 = y_obs[0,0]; r0 = y_obs[0,1]
        px0 = r0 * np.cos(th0); py0 = r0 * np.sin(th0)
        x0_guess = np.array([px0, py0, 0.5, 0.0, 0.0])  # moderate speed guess, omega=0
    if R is not None:
        try:
            L = np.linalg.cholesky(R)
            R_sqrt_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            R_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(R)))
    else:
        R_sqrt_inv = None

    # sensible bounds: keep omega in a reasonable range to help identifiability
    if bounds is None:
        # px,py unbounded, vx,vy within [-50*space, 50*space] (you can tune), omega in [-2*pi,2*pi]
        lower = [-np.inf, -np.inf, -np.inf, -np.inf, -2*np.pi]
        upper = [ np.inf,  np.inf,  np.inf,  np.inf,  2*np.pi]
        bounds = (lower, upper)

    res = least_squares(residuals_ct, x0_guess, args=(t_obs, y_obs, R_sqrt_inv), bounds=bounds, verbose=verbose)
    return res.x, res

# ----------------------
# Wrapper to estimate all targets from tracks
# ----------------------
def estimate_all_targets_from_tracks(tracks, env, R=None, **batch_kwargs):
    """
    tracks: output of extract_tracks_from_log
    R: measurement noise matrix
    """

    if R is None:
        sigma_theta = np.deg2rad(1/3600)       # 1arcsec
        sigma_range = 0.0001
        R = np.diag([sigma_theta**2, sigma_range**2])

    estimates = {}

    for tgt_id, track in tracks.items():

        # --- Extract data from track ---
        timesteps = [obs["t"] for obs in track]
        #states = [obs["state"] for obs in track]

        # Skip this target if there are no timesteps/measurements
        if not timesteps:   # or: if len(timesteps) == 0:
            continue

        # --- Generate truth states ---
        truth_states = generate_truth_states(timesteps, tgt_id, env)

        if len(timesteps) == 0:
            continue

        # --- Convert ground-truth states → measurements ---
        y_obs = []
        for x in truth_states:
            H, Gk = MultiTargetEnv.extract_measurement_bearingRange(x)
            y_true = Gk
            # Add measurement noise from R
            noise = np.random.multivariate_normal(mean=np.zeros(2), cov=R)
            y_noisy = y_true + noise

            y_obs.append(y_noisy)
        y_obs = np.array(y_obs)

        # --------------------------------------------
        # 1) Fit initial state under CV model
        # --------------------------------------------
        x0_cv, res_cv = fit_initial_state_cv(timesteps, y_obs, R=R, verbose=0)
        cost_cv = 2 * res_cv.cost

        # --------------------------------------------
        # 2) Fit initial state under CT model
        # --------------------------------------------
        x0_ct, res_ct = fit_initial_state_ct(timesteps, y_obs, R=R, verbose=0)
        cost_ct = 2 * res_ct.cost

        # --------------------------------------------
        # 3) Pick best model
        # --------------------------------------------
        if env.motion_model[tgt_id] == "L": #if (cost_cv < cost_ct) or (abs(x0_ct[4]) < 1e-10):      # assume linear motion if turn rate is too small
            chosen_model = "CV"
            x0_ref = truth_states[0]
            best_cost = cost_cv
        else:
            chosen_model = "CT"
            #x0_ref = x0_ct
            x0_ref = truth_states[0]
            #x0_ref = np.append(x0_ref, env.motion_params[tgt_id])
            best_cost = cost_ct

        print(f"Target {tgt_id}: chosen {chosen_model} with cost {best_cost}")

        # --------------------------------------------
        # 4) Build matching P0 (dimension = len(x0_ref))
        # --------------------------------------------
        dim = len(x0_ref)
        P0 = np.eye(dim) * 0.05
        P0[:2, :2] = np.eye(2) * 0.1

        # --------------------------------------------
        # 5) Run batch estimator with chosen model
        # --------------------------------------------
        out = batch_estimate_single_target(
            timesteps,
            y_obs,
            x0_ref,
            P0,
            R,
            model=chosen_model,
            omega=env.motion_params[tgt_id],
            **batch_kwargs
        )

        estimates[tgt_id] = out

    return estimates

def generate_truth_states(t_vec, tgt_id, env, H_fcn):
    """
    Generate truth trajectory for a given target and measurement values.
    Returns truth_states with shape (T, n), measurements and observation matrix.
    """

    # State dimension
    n = env.targets[0]["x"].shape[0]

    # Allocate output
    truth_states = np.zeros((len(t_vec), n))
    measurements = np.zeros((len(t_vec), 2))        
    H_mod = []

    # Initial state at t=0
    if tgt_id>=env.init_n_targets:
        x0 = env.unknown_targets[tgt_id - env.init_n_targets]["x"].copy()
    else:
        x0 = env.targets[tgt_id]["x"].copy()

    Phi0 = np.eye(n).reshape(-1)

    # --- Propagate from t=0 → t_vec[0] ---
    t_first = t_vec[0]
    if t_first > 0:
        if env.motion_model[tgt_id] == "T":
            x0, _ = propagate_state_and_stm(
                0.0, t_first, x0, Phi0,
                f_ct, Fx_ct, n
            )
        else:
            x0, _ = propagate_state_and_stm(
                0.0, t_first, x0, Phi0,
                f_cv_cont, Fx_cv_cont, n
            )

    # Store initial propagated truth
    truth_states[0] = x0[:4]
    H, Gk = H_fcn(x0[:4])
    measurements[0,:] = Gk
    H_mod.append(H)

    # --- Propagate through all times ---
    for k in range(len(t_vec)-1):
        t0 = t_vec[k]
        t1 = t_vec[k+1]

        if env.motion_model[tgt_id] == "T":
            x_next, _ = propagate_state_and_stm(
                t0, t1, x0, Phi0,
                f_ct_linear, Fx_ct_linear, n, env.motion_params[tgt_id]
            )
        else:
            x_next, _ = propagate_state_and_stm(
                t0, t1, x0, Phi0,
                f_cv_cont, Fx_cv_cont, n
            )

        truth_states[k+1] = x_next
        H, Gk = H_fcn(x_next[:4])
        measurements[k+1,:] = Gk
        H_mod.append(H)


        x0 = x_next

    return truth_states, measurements, H_mod

# -------------------------------------------------------------------------
# 1. Extract states, covariances, and compute errors + sigma bounds
# -------------------------------------------------------------------------
def process_estimates(t_vec, estimates, env):
    """
    estimates: output of estimate_all_targets_from_tracks

    Returns:
        truth_by_target[tgt_id] → array [T, 4 or 5]
        est_by_target[tgt_id]   → array [T, 4 or 5]
        cov_by_target[tgt_id]   → array [T, dim, dim]
        errors_by_target[tgt_id] → array [T, dim]
        sigma_by_target[tgt_id]  → array [T, dim]
    """

    truth_by_target = {}
    est_by_target = {}
    cov_by_target = {}
    errors_by_target = {}
    sigma_by_target = {}
    t_by_target = {}

    for tgt_id, est in estimates.items():

        # Extract timestamps
        #t_vec = np.array(est["timesteps"])

        # --- Generate truth states ---
        truth_states, measurements, H_mod = generate_truth_states(t_vec, tgt_id, env)
        truth_by_target[tgt_id] = truth_states

        # --- Extract estimates ---
        est_states = np.array(estimates[tgt_id]["Xest"]).T   # shape (T, n)
        est_covs   = np.array(estimates[tgt_id]["P_mat"])

        # If covariances are stored as (n, n, T), reorder to (T, n, n)
        if est_covs.ndim == 3 and est_covs.shape[2] == len(t_vec):
            est_covs = np.transpose(est_covs, (2, 0, 1))

        est_by_target[tgt_id] = est_states
        cov_by_target[tgt_id] = est_covs

        # --- Compute errors & sigma bounds ---
        errors = est_states - truth_states             # (T, n)
        sigma  = np.sqrt(np.array([np.diag(P) for P in est_covs]))

        errors_by_target[tgt_id] = errors
        sigma_by_target[tgt_id] = sigma
        t_by_target[tgt_id] = t_vec

    return (
        truth_by_target,
        est_by_target,
        cov_by_target,
        errors_by_target,
        sigma_by_target,
        t_by_target,
    )
# -------------------------------------------------------------------------
# 2. Plot error trajectories + ±3σ envelopes
# -------------------------------------------------------------------------
def plot_errors_and_sigmas(errors, sigmas, t, target_id):
    """
    errors_by_target: dict[tgt_id] -> [T, dim]
    sigma_by_target: dict[tgt_id] -> [T, dim]
    """

    labels = ["x", "y", "xdot", "ydot"]
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    for i in range(4):
        axs[i].scatter(t, errors[:, i], label="error")
        axs[i].plot(t, 3*sigmas[:, i], linestyle="--", label="+3σ")
        axs[i].plot(t, -3*sigmas[:, i], linestyle="--", label="-3σ")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-1].set_xlabel("time")
    fig.suptitle(f"Target {target_id}: Errors and 3σ Bounds")
    axs[0].legend()
    plt.tight_layout()
    plt.show()
