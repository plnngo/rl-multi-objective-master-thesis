import numpy as np
import random
import math
from multi_target_env import MultiTargetEnv, compute_fov_prob_single


def select_best_action_pFOV(env, dt=None, fov=4):
    """
    Returns the target ID that yields the highest probability to be inside FOV after measurement update
    when selecting it as the next action.

    Parameters
    ----------
    env : object
        Environment-like object containing targets, models, parameters, rng, etc.
    dt : float or None
        Time step.

    Returns
    -------
    best_target_id : int
        ID of the target with highest information gain.
    best_ig : float
        The corresponding information gain value.
    best_update : dict
        Contains the updated state and covariance (x, P) for the best action.
    """
    dt = dt if dt is not None else env.dt

    best_target_id = None
    best_ig = -float('inf')
    highest_prob = -float('inf')
    best_update = None 

    obsFunc = MultiTargetEnv.extract_measurement_XY

    # Pre-compute propagations for all targets
    propagated = {}
    for tgt in env.targets:
        idx = tgt['id']
        model = env.motion_model[idx]
        param = env.motion_params[idx]
        x_pred, P_pred = MultiTargetEnv.propagate_target_2D(
            tgt['x'], tgt['P'], tgt.get('Q', env.Q0),
            dt=dt,
            rng=env.rng,
            motion_model=model,
            motion_param=param
        )
        propagated[idx] = (x_pred, P_pred)

    # Pre-sum all predicted probs
    total_pred_prob = sum(
        compute_fov_prob_single(fov, x_p, P_p)
        for x_p, P_p in propagated.values()
    )

    # Main loop: choose one target as the sensing action 
    for tgt in env.targets:
        idx = tgt['id']
        x_pred, P_pred = propagated[idx]
        x_upd, P_upd = MultiTargetEnv.ekf_update(x_pred, P_pred, env.R, obsFunc)
       
        # Swap out this target's predicted prob for its updated prob
        prob_this_pred = compute_fov_prob_single(fov, x_pred, P_pred)
        prob_this_upd  = compute_fov_prob_single(fov, x_upd,  P_upd)
        prob = total_pred_prob - prob_this_pred + prob_this_upd
        """ print(f"probUpdated={prob:.10f}, "
            f"prob_diff={(prob_this_pred - prob_this_upd):.10f}, "
            f"pred={prob_this_pred:.10f}, "
            f"upd={prob_this_upd:.10f}") """

        # Keep the best
        if prob > highest_prob:
            highest_prob = prob
            best_targets = [(idx, x_upd, P_upd)]

        elif math.isclose(prob, highest_prob, rel_tol=1e-12, abs_tol=1e-15):
            best_targets.append((idx, x_upd, P_upd))
        
    # randomly choose among the tied best targets
    best_target_id, x_best, P_best = random.choice(best_targets)

    best_update = {"x": x_best, "P": P_best}

    return best_target_id, best_ig, best_update

def select_best_action_sumTrace(env, dt=None, fov=4):
    """
    Returns the target ID that yields the lowest total sum of covariance trace FOV after measurement update
    when selecting it as the next action.

    Parameters
    ----------
    env : object
        Environment-like object containing targets, models, parameters, rng, etc.
    dt : float or None
        Time step.

    Returns
    -------
    best_target_id : int
        ID of the target with lowest total covariance trace.
    best_ig : float
        The corresponding information gain value.
    best_update : dict
        Contains the updated state and covariance (x, P) for the best action.
    """
    dt = dt if dt is not None else env.dt

    best_target_id = None
    best_ig = -float('inf')
    highest_prob = -float('inf')
    lowest_trace = float('inf')
    best_update = None
    #fov = env.fov_size 

    obsFunc = MultiTargetEnv.extract_measurement_XY

    # Pre-compute propagations for all targets
    propagated = {}
    trace = 0
    for tgt in env.targets:
        idx = tgt['id']
        model = env.motion_model[idx]
        param = env.motion_params[idx]
        x_pred, P_pred = MultiTargetEnv.propagate_target_2D(
            tgt['x'], tgt['P'], tgt.get('Q', env.Q0),
            dt=dt,
            rng=env.rng,
            motion_model=model,
            motion_param=param
        )
        propagated[idx] = (x_pred, P_pred)
        trace += np.trace(P_pred)

    # Pre-sum all predicted probs
    """ total_pred_prob = sum(
        compute_fov_prob_single(fov, x_p, P_p)
        for x_p, P_p in propagated.values()
    ) """

    # Main loop: choose one target as the sensing action 
    for tgt in env.targets:
        idx = tgt['id']
        x_pred, P_pred = propagated[idx]
        x_upd, P_upd = MultiTargetEnv.ekf_update(x_pred, P_pred, env.R, obsFunc)
       
        # Swap out this target's predicted prob for its updated prob
        """ prob_this_pred = compute_fov_prob_single(fov, x_pred, P_pred)
        prob_this_upd  = compute_fov_prob_single(fov, x_upd,  P_upd) """
        #prob = total_pred_prob - prob_this_pred + prob_this_upd
        traceUpdated = trace - np.trace(P_pred) + np.trace(P_upd)
        """ print(f"traceUpdated={traceUpdated:.4f}, "
            f"trace_diff={(np.trace(P_pred) - np.trace(P_upd)):.4f}, "
            f"trace(P_pred)={np.trace(P_pred):.4f}, "
            f"trace(P_upd)={np.trace(P_upd):.4f}") """

        # Keep the best
        """ if prob > highest_prob:
            highest_prob = prob
            best_targets = [(idx, x_upd, P_upd)]

        elif math.isclose(prob, highest_prob, rel_tol=1e-12, abs_tol=1e-15):
            best_targets.append((idx, x_upd, P_upd)) """
        if traceUpdated < lowest_trace:
            lowest_trace = traceUpdated
            best_targets = [(idx, x_upd, P_upd)]

        elif math.isclose(traceUpdated, lowest_trace, rel_tol=1e-12, abs_tol=1e-15):
            best_targets.append((idx, x_upd, P_upd))

    # randomly choose among the tied best targets
    best_target_id, x_best, P_best = random.choice(best_targets)
    #print(highest_prob)
    if len(best_targets) > 1:
            print("Random")
    best_update = {"x": x_best, "P": P_best}

    return best_target_id, best_ig, best_update

def select_best_action_IG(env, dt=None):
    """
    Returns the target ID that yields the highest information gain (KL divergence)
    when selecting it as the next action.

    Parameters
    ----------
    env : object
        Environment-like object containing targets, models, parameters, rng, etc.
    dt : float or None
        Time step.

    Returns
    -------
    best_target_id : int
        ID of the target with highest information gain.
    best_ig : float
        The corresponding information gain value.
    best_update : dict
        Contains the updated state and covariance (x, P) for the best action.
    """
    dt = dt if dt is not None else env.dt

    best_target_id = None
    best_ig = -float('inf')
    best_update = None
    best_targets = []

    obsFunc = MultiTargetEnv.extract_measurement_XY

    # Loop through all possible sensing actions: choose one target
    for tgt in env.targets:
        idx = tgt['id']
        model = env.motion_model[idx]
        param = env.motion_params[idx]

        # Propagate the target forward
        x_pred, P_pred = MultiTargetEnv.propagate_target_2D(
            tgt['x'], tgt['P'], tgt.get('Q', env.Q0),
            dt=dt,
            rng=env.rng,
            motion_model=model,
            motion_param=param
        )

        # Perform the hypothetical EKF update
        x_upd, P_upd = MultiTargetEnv.ekf_update(x_pred, P_pred, env.R, obsFunc)

        # Compute information gain
        ig = MultiTargetEnv.compute_kl_divergence(x_pred, P_pred, x_upd, P_upd)

        if ig > best_ig:
            best_ig = ig
            best_targets = [(idx, x_upd, P_upd)]

        elif math.isclose(ig, best_ig, rel_tol=1e-12, abs_tol=1e-15):
            best_targets.append((idx, x_upd, P_upd))

    # Random tie-break
    best_target_id, x_best, P_best = random.choice(best_targets)
    best_update = {"x": x_best, "P": P_best}

    return best_target_id, best_ig, best_update

def select_best_macro_action(env):
    obs = env._get_obs()
    known_obs = obs[env.known_mask]
    return 1 if np.any(known_obs <= env.threshold_fov) else 0

def select_best_pointingToFind(env):
    obj = random.choice(env.unknown_targets)

    idx = obj['id']
    model = env.motion_model[idx]
    param = env.motion_params[idx]
    x_pred, P_pred = MultiTargetEnv.propagate_target_2D(
        obj['x'], obj['P'], obj.get('Q', env.Q0),
        dt=env.dt,
        rng=env.rng,
        motion_model=model,
        motion_param=param
    )

    dx = x_pred[0]
    dy = x_pred[1]

    pos = np.array([dx, dy])
    dists = np.sum((env.grid_coords - pos) ** 2, axis=1)
    grid_idx = int(np.argmin(dists))
  
    return np.array(grid_idx)