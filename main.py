import copy
import os
import re
from matplotlib.colors import to_rgba
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import KalmanFilter
from LSBatchFilter import Fx_ct_linear, Fx_cv_cont, f_ct_linear, f_cv_cont
from deterministic_tracker import select_best_action_IG, select_best_action_pFOV, select_best_action_sumTrace
from multi_target_env import MultiTargetEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from collections import defaultdict

def plot_cov_ellipse(cov, mean, ax, n_std=1.0, **kwargs):
    """Plot an ellipse representing the covariance matrix cov centered at mean."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)


def visualize_initial_positions(env):
    """Plot the initial positions of known and unknown targets."""
    fig, ax = plt.subplots()

    # Draw discretized grid
    fov_half = env.fov_size / 2.0
    for gx, gy in env.grid_coords:
        rect = Rectangle(
            (gx - fov_half, gy - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="lightgray",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    # Known targets (blue)
    for tgt in env.targets:
        ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target")
        #plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="blue", alpha=0.3)

    # Unknown targets (orange)
    for utgt in env.unknown_targets:
        ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target")
        #plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="orange", alpha=0.3)

    ax.set_xlim(-env.space_size / 2, env.space_size / 2)
    ax.set_ylim(-env.space_size / 2, env.space_size / 2)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Initial Target Positions with Grid Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    # Combine duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()


def run_random_policy_search(env, n_steps):
    """Task sensor to search in grid cells corresponding to initial unknown target positions."""
    #obs = env.reset()
    positions, covariances = [], []

    fov_half = env.fov_size / 2.0

    # Find grid cells closest to unknown targets
    search_indices = []
    for utgt in env.unknown_targets:
        pos = utgt["x"][:2]
        pos = np.array([pos[0] + 1, pos[1]])
        dists = np.linalg.norm(env.grid_coords - pos, axis=1)
        grid_idx = np.argmin(dists)
        search_indices.append(grid_idx)

    print("Grid indices for initial unknown targets:", search_indices)

    for step, grid_idx in enumerate(search_indices):
        search_pos = env.grid_coords[grid_idx]
        action = {"macro": 0, "micro_search": grid_idx}
        obs, reward, done, truncated, info = env.step(action)

        print(f"Step {step+1:02d}: Search at {search_pos}, Reward={reward:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw all grid cells
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)

        # Draw current FOV (green)
        fov_rect = Rectangle(
            (search_pos[0] - fov_half, search_pos[1] - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            lw=2,
        )
        ax.add_patch(fov_rect)

        # Plot known (blue) + unknown (orange)
        for tgt in env.targets:
            ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target" if step == 0 else "")
        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target" if step == 0 else "")

        # Draw uncertainty ellipses 
        for tgt in env.targets:
            plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="red", alpha=0.3)
        for utgt in env.unknown_targets:
            plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="red", alpha=0.3)

        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: Search at {search_pos}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        # Combine legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

        plt.show(block=True)  # <-- opens one figure per step, waits until closed

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    plt.show()
    return np.array(positions), np.array(covariances)

def run_random_policy_track(env, n_steps):
    """
    Executes only TRACK macro-actions with random valid known targets at each step.
    Visualizes the state and uncertainty after every tracking action.
    """
    positions, covariances = [], []
    figures = []  # store figure handles
    exceed_target = []

    for step in range(n_steps):
        valid_ids = np.flatnonzero(env.known_mask)
        if len(valid_ids) == 0:
            print("No known targets available for tracking at step", step)
            break

        action = int(env.rng.choice(valid_ids))
        obs, reward, done, truncated, info = env.step(action)
        if len(info["lost_target"]) > 0:
            print(f"Step {step+1:02d}: lost target")
      
        print(f"Step {step+1:02d}: TRACK target {action}, Reward={reward:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        figures.append(fig)  # keep it open!

        # (your plotting code remains unchanged)
        fov_half = env.fov_size / 2.0
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)

        for tgt in env.targets:
            if tgt["id"] == action:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="red", s=120, marker="*", label="Tracked Target")
            else:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", s=40, marker="o", label="Known Target" if step == 0 else "")
        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target" if step == 0 else "")

        for tgt in env.targets:
            plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="red" if tgt["id"] == action else "blue", alpha=0.5)
        for utgt in env.unknown_targets:
            plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="red", alpha=0.3)

        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: TRACK target {action}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    # --- Show all figures together at the end ---
    #plt.show()
    return np.array(positions), np.array(covariances)


def plot_results(env, positions, covariances):
    """Plot 2D trajectories and uncertainty ellipses for known and unknown targets."""
    fig, ax = plt.subplots()

    n_known = env.n_targets
    n_unknown = env.n_unknown_targets

    # Plot known targets (blue)
    for i in range(n_known):
        ax.plot(positions[:, i, 0], positions[:, i, 1], color="blue", label=f"Known {i}")
        for step in range(len(positions)):
            plot_cov_ellipse(
                covariances[step, i],
                positions[step, i],
                ax,
                n_std=1.0,
                edgecolor="blue",
                alpha=0.3,
            )

    # Plot unknown targets (orange)
    for j in range(n_unknown):
        idx = n_known + j
        ax.plot(positions[:, idx, 0], positions[:, idx, 1], color="orange", label=f"Unknown {j}")
        for step in range(len(positions)):
            plot_cov_ellipse(
                covariances[step, idx],
                positions[step, idx],
                ax,
                n_std=1.0,
                edgecolor="orange",
                alpha=0.3,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Target Trajectories (Known + Unknown)")
    ax.legend()
    ax.grid(True)
    plt.show()

def _unpack_cholesky(ch_pack, d):
    """
    Unpack a lower-triangular matrix L (d x d) from the packed vector ch_pack.
    Packing assumed row-by-row for lower triangle:
      for i in range(d):
          for j in range(i+1):
              append(L[i, j])
    Returns L (d x d).
    """
    L = np.zeros((d, d), dtype=float)
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            L[i, j] = ch_pack[idx]
            idx += 1
    return L

def extract_target_state_cov(target_idx, env):
    """
    Extract both the d-dimensional state vector and its corresponding dxd covariance
    matrix for a given target index from the flat observation vector `obs`.

    Parameters
    ----------
    target_idx : int
        Index of the target to extract (0-based).
    env : MultiTargetEnv
        Environment instance containing dimensional parameters.

    Returns
    -------
    x : np.ndarray
        State vector of shape (d_state,).
    P : np.ndarray
        Covariance matrix of shape (d_state, d_state).
    """
    x = None
    P = None
    # Extract state vector (first d_state elements)
    for tgt in env.targets:
        if tgt["id"] == target_idx:
            x = tgt["x"]
            P = tgt["P"]
            break

    return x, P

@staticmethod
def analyse_tracking_task(target_idx, env, confidence=0.95):
    """
    Test whether the 2D positional covariance (first two state dims) for target
    exceeds the env.fov_size.

    By default this uses the 95% confidence ellipse (chi2 = 5.991 for 2D).
    It computes the semi-major axis = sqrt(chi2 * lambda_max) where lambda_max
    is the largest eigenvalue of the 2x2 positional covariance.
    We consider the target's covariance to 'exceed' the FOV when the FULL-length
    major axis (2 * semi_major) is larger than env.fov_size.

    Return: 
    True/False. If the target is masked (mask <= 0.5) then returns False.
    x : np.ndarray
        State vector of shape (d_state,).
    P : np.ndarray
        Covariance matrix of shape (d_state, d_state).
    """
    # check mask: mask vector placed after all per-target blocks
    """ mask_offset = int(env.max_targets * env.obs_dim_per_target)
    mask_val = obs[mask_offset + int(target_idx)]
    if mask_val <= 0.5:
        return True, None, None """

    x, P = extract_target_state_cov(target_idx, env)
    if x is None:
        return False, None , None
    # positional covariance assumed to be in state dims [0,1] (x,y)
    posP = P[:2, :2]

    # handle degenerate / non-PD posP
    # force symmetric
    posP = 0.5 * (posP + posP.T)
    # small regularization if needed
    try:
        eigvals = np.linalg.eigvalsh(posP)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvals(posP).real
    # clip negative tiny eigenvalues to zero
    eigvals = np.clip(eigvals, 0.0, None)
    lambda_max = np.max(eigvals)

    # chi-square value for 2D at given confidence:
    # 0.68 -> 2.279, 0.90 -> 4.605, 0.95 -> 5.991, 0.99 -> 9.210
    # Here we use the 95% default
    if confidence == 0.95:
        chi2_val = 5.991
    elif confidence == 0.90:
        chi2_val = 4.605
    elif confidence == 0.99:
        chi2_val = 9.210
    else:
        # approximate general quantile using scipy would be best, but avoid dependency:
        # for uncommon values fallback to 95% constant
        chi2_val = 5.991

    semi_major = np.sqrt(chi2_val * lambda_max)  # semi-major axis length
    full_major = 2.0 * semi_major

    return full_major > float(env.fov_size), x, P

def evaluate_agent_track(env, model=None, n_episodes=1, random_policy=False, deterministic_policy=False, deterministic_policy_alternative = False, seed=None, maskable=False, fov=4):
    rewards = []
    exceedFOV = []
    illegal_actions = []

    # for logging the final episode
    last_episode_log = {}
    last_env = None

    for ep in trange(n_episodes, desc="Evaluating"):
        obs = env.obs
        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_env = copy.deepcopy(env)
            episode_log = {}        # create a temporary log if this is the last episode
        done = False
        total_reward = 0.0
        illegal_action = 0.0
        t = 0  # timestep counter

        while not done:
            # --- Choose action ---
            if random_policy:
                #action = env.action_space.sample()
                action = np.random.randint(0, 5, dtype=np.int64)
            elif deterministic_policy and deterministic_policy_alternative:
                action, best_ig, best_update = select_best_action_sumTrace(env, env.dt)
            elif deterministic_policy:
                action, best_ig, best_update = select_best_action_pFOV(env, env.dt, fov)
            elif deterministic_policy_alternative:
                action, best_ig, best_update = select_best_action_IG(env, env.dt)
                #action = 4
            elif maskable:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks)

            else:
                action, _ = model.predict(obs, deterministic=False)


            # --- Step environment ---
            obs, reward, done, truncated, info = env.step(action)

            if not info["action_mask"][np.asarray(action).item()]:
                illegal_action = illegal_action + 1
            total_reward += reward

            # --- Analyse targets ---
            if ep == n_episodes - 1:
                episode_log[t] = {}

            if ep == n_episodes - 1:
                for tgt in env.targets:
                    #exceed, x, P = analyse_tracking_task(tgt["id"], env, confidence=0.95)
                    x, P = extract_target_state_cov(tgt["id"], env)

                    # if this is the last episode, store everything
                    if not info["action_mask"][np.asarray(action).item()]:
                        """ if info.get("invalid_action"):
                            illegalActs += 1 """
                        continue
                    if tgt["id"] in info["target_id"]:
                        if x is None:
                            print("error")
                        episode_log[t][tgt["id"]] = {
                            "id": tgt["id"],
                            "state": x.copy(),
                            "cov": P.copy(),
                        }

            t += 1  # increment timestep

        rewards.append(total_reward)
        exceedFOV.append(env.init_n_targets-env.n_targets)
        illegal_actions.append(illegal_action)

        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_episode_log = episode_log
        else:
            obs, _ = env.reset(seed=seed)


    return rewards, exceedFOV, last_env, last_episode_log, illegal_actions

def visualize_search_pointing_heatmap(env, pointing_history):
    """
    Plot grid and overlay SEARCH actions colored by timestep.
    """
    fig, ax = plt.subplots()

    # Draw grid
    fov_half = env.fov_size / 2.0
    for gx, gy in env.grid_coords:
        rect = Rectangle(
            (gx - fov_half, gy - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="lightgray",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    if len(pointing_history) > 0:
        grid_indices, timesteps = zip(*pointing_history)
        grid_positions = np.array([env.grid_coords[i] for i in grid_indices])

        sc = ax.scatter(
            grid_positions[:, 0],
            grid_positions[:, 1],
            c=timesteps,
            cmap="viridis",
            s=60,
            alpha=0.9,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Timestep")

    ax.set_xlim(-env.space_size / 2, env.space_size / 2)
    ax.set_ylim(-env.space_size / 2, env.space_size / 2)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    #ax.set_title("Agent Pointing Over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    plt.show()

def visualize_unknown_target_heatmap(env, unknown_target_history):
    """
    Plot unknown target positions over time using a time-normalized colormap.
    """
    fig, ax = plt.subplots()

    # Optional: draw grid for reference
    fov_half = env.fov_size / 2.0
    for gx, gy in env.grid_coords:
        rect = Rectangle(
            (gx - fov_half, gy - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="lightgray",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    if len(unknown_target_history) > 0:
        data = np.array(unknown_target_history)
        xs, ys, ts = data[:, 0], data[:, 1], data[:, 2]

        sc = ax.scatter(
            xs,
            ys,
            c=ts,
            cmap="viridis",   # distinct from agent heatmap
            s=60,
            alpha=0.9,
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Timestep")

    ax.set_xlim(-env.space_size / 2, env.space_size / 2)
    ax.set_ylim(-env.space_size / 2, env.space_size / 2)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    #ax.set_title("Unknown Target Positions Over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    plt.show()

def evaluate_agent_search(env, model=None, n_episodes=100, random_policy=False, seed=None):
    """
    Evaluates an RL agent or random policy on the given environment,
    tracking episode rewards and detection counts.

    Detection count is inferred by comparing the evolution of the action mask:
    whenever a new target becomes trackable (mask bit switches from 0 to 1),
    we count it as a detection.

    Parameters:
        env: MultiTargetEnv instance
        model: trained SB3 model (PPO, DQN, etc.)
        n_episodes: number of episodes
        random_policy: if True, sample random actions instead of using the model

    Returns:
        rewards: list of total rewards per episode
        detections: list of detection counts per episode
    """
    rewards = []
    detection_count = []

    for ep in trange(n_episodes, desc="Evaluating"):

        obs, _ = env.reset(seed=int(np.random.choice(seed)))
        done = False
        total_reward = 0.0
        pointing_history = []  # list of (grid_idx, timestep)
        unknown_target_history = []  
        t = 0

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, done, truncated, info = env.step(action)
            macro, micro_search, micro_track = env.decode_action(action)
            pointing_history.append((micro_search, t))
            for utgt in env.unknown_targets:
                x, y = utgt["x"][0], utgt["x"][1]
                unknown_target_history.append((x, y, t))
            total_reward += reward
            """ # Compare action mask with previous one to detect new trackable targets
            known_targets = sum(info["action_mask"]["micro_track"])
            if known_targets>detections:
                detect_count3 = detect_count3 + (known_targets - detections)
                detections = known_targets """
            t=t+1
        if ep == n_episodes - 1:
            visualize_search_pointing_heatmap(env, pointing_history)
            visualize_unknown_target_heatmap(env, unknown_target_history)
        rewards.append(total_reward)
        detection_count.append(env.detect_counter)

    return rewards, detection_count

def computeKL():
    mean_1 = np.array([0])
    mean_2 = np.array([0])

    cov_1 = np.array([1])
    cov_2 = np.array([1])
    kl_value = MultiTargetEnv.compute_kl_divergence(mean_1, cov_1, mean_2, cov_2)
    print(kl_value)

@staticmethod
def plot_violin(results_dict, ylabel="Episode Reward"):
    """
    Plots a violin plot comparing metrics (e.g., rewards or detections) across agents.
    """
    colors = {
        "PPO": "blue",
        "DQN": "orange",
        "Random": "red",
        "Heuristic": "green",
        "MCTS": "purple",
        "Maskable PPO": "grey",
        "0.95": "blue",
        "0.5": "orange",
        "0.1": "red",
        "HeuristicHeuristic": "black" 
    }
    
    data = []
    labels = []
    for label, values in results_dict.items():
        data.extend(values)
        labels.extend([label] * len(values))

    plt.figure()
    sns.violinplot(x=labels, y=data, inner="quart", cut=0, palette=colors)
    plt.xlabel("Agent")
    plt.ylabel(ylabel)
    plt.title(f"Distribution of {ylabel}")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Create output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "evaluated")
    os.makedirs(output_dir, exist_ok=True)

    # --- sanitize ylabel for filename ---
    safe_ylabel = re.sub(r'[^a-zA-Z0-9]+', '_', ylabel).strip('_').lower()

    filename = f"violin_{safe_ylabel}.pdf"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def visualize_trained_agent(env, model, n_steps=20):
    """Run the trained agent and visualize its decisions and environment state."""
    obs, _ = env.reset()
    fov_half = env.fov_size / 2.0

    for step in range(n_steps):
        # --- Get action from trained agent ---
        if isinstance(model, MaskablePPO):
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks)
        else:
            action, _ = model.predict(obs, deterministic=False)
        macro, micro_search, micro_track = env.decode_action(action)

        # --- Apply action in environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: Reward={reward:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # --- Draw grid ---
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)
        
        # --- Visualize the agent’s chosen action ---
        if macro == 0:  # SEARCH
            search_pos = env.grid_coords[micro_search]
            fov_rect = Rectangle(
                (search_pos[0] - fov_half, search_pos[1] - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                lw=2,
                label="Search FOV",
            )
            ax.add_patch(fov_rect)
            print(f"Step {step+1:02d}: Search at {search_pos}, Reward={reward:.4f}")
        else:  # TRACK
            target_id = micro_track
            tgt = env.targets[target_id]
            ax.scatter(
                tgt["x"][0], tgt["x"][1],
                c="red", s=120, marker="*",
                label=f"Tracked target {target_id}"
            )
            print(f"Step {step+1}: TRACK target {target_id}, Reward={reward:.4f}")

        # --- Plot targets ---
        for tgt in env.targets:
            ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target")

            # Plot 2D covariance ellipse if available
            if "P" in tgt:
                P_xy = tgt["P"][:2, :2]  # take only position covariance
                plot_cov_ellipse(P_xy, tgt["x"][:2], ax, edgecolor="blue", alpha=0.4)

        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target")

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="lower left")

        #plt.show(block=True)
        plt.pause(4)
        plt.close(fig)

        if done:
            obs, _ = env.reset()

@staticmethod
def plot_detection_bar_chart(results_dict):
    """
    Plots a bar chart showing the mean number of detections (count 2) per agent.

    Parameters:
        results_dict: dict mapping agent name -> list of detection counts per episode
                      e.g. {"Random": random_detections, "PPO": ppo_detections, "DQN": dqn_detections}
    """
    colors = {
        "PPO": "orange",
        "DQN": "green",
        "Random": "red"
    }

    agents = list(results_dict.keys())
    means = [np.mean(results_dict[a]) for a in agents]
    stds = [np.std(results_dict[a]) for a in agents]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(agents, means, yerr=stds, capsize=6,
                   color=[colors[a] for a in agents], alpha=0.8)

    plt.xlabel("Agent")
    plt.ylabel("Total Number of Detections")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Add value labels on top of bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f"{mean:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_positions(positions, env=None, show_start_end=True):
    """
    Plots the target positions over time on an x-y field.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n_steps, n_targets, 2), containing x and y positions.
    env : optional
        Environment object, used to set plot limits if provided.
    show_start_end : bool, default=True
        If True, mark start and end positions for each target.
    """
    n_steps, n_targets, _ = positions.shape

    plt.figure(figsize=(8, 8))
    
    # Plot each target's trajectory
    for i in range(n_targets):
        traj = positions[:, i, :]
        plt.plot(traj[:, 0], traj[:, 1], '-', lw=2, label=f'Target {i}')
        if show_start_end:
            plt.scatter(traj[0, 0], traj[0, 1], c='green', marker='o', s=60)  # Start
            plt.scatter(traj[-1, 0], traj[-1, 1], c='red', marker='x', s=80)  # End
    
    # Add grid / field visualization
    if env is not None:
        plt.xlim(-env.space_size / 2, env.space_size / 2)
        plt.ylim(-env.space_size / 2, env.space_size / 2)
    else:
        all_x = positions[:, :, 0].flatten()
        all_y = positions[:, :, 1].flatten()
        plt.xlim(np.min(all_x) - 10, np.max(all_x) + 10)
        plt.ylim(np.min(all_y) - 10, np.max(all_y) + 10)

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Target Trajectories over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

@staticmethod
def plot_means_lost_targets(ppo, knownPPO, dqn = [], knownDQN =[], random=[], knownRandom =[], det=[], knowndet=[], labels=None):
    """
    ppo, dqn, random are lists of arrays.
    Compute:
        - global mean over all entries
        - global std over all entries
    knownPPO, knownDQN, knownRandom:
        - compute simple means (kept unchanged)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    # --- Compute statistics ---
    means = np.array([
        np.mean([np.mean(arr) for arr in ppo]),
        np.mean([np.mean(arr) for arr in dqn]),
        np.mean([np.mean(arr) for arr in random]),
        np.mean([np.mean(arr) for arr in det])
    ])

    stds = np.array([
        np.std([np.mean(arr) for arr in ppo], ddof=1),
        np.std([np.mean(arr) for arr in dqn], ddof=1),
        np.std([np.mean(arr) for arr in random], ddof=1),
        np.std([np.mean(arr) for arr in det], ddof=1)
    ])

    # --- Known means remain unchanged ---
    known_means = np.array([
        np.mean(knownPPO),
        np.mean(knownDQN),
        np.mean(knownRandom),
        np.mean(knowndet)
    ])

    # --- Plotting ---
    x = np.arange(len(labels))
    colors = ["blue", "orange", "red", "green"]
    light_colors = [to_rgba(c, alpha=0.35) for c in colors]

    plt.figure(figsize=(7, 5))

    # Baseline bars (tracking targets)
    baseline_bars = plt.bar(
        x, known_means,
        width=0.6,
        color=light_colors,
        label="Number of tracking targets",
        zorder=1
    )

    for bar, value in zip(baseline_bars, known_means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    # Main bars (lost targets, with std)
    bars = plt.bar(
        x, means,
        yerr=stds,
        capsize=8,
        color=colors,
        label="Number of lost targets",
        zorder=2
    )

    # Bar height labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.xticks(x, labels)
    plt.ylabel("Mean Lost Targets")
    plt.title("Mean Lost Targets Across Agents (Global Mean + Std)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def unpack_cholesky(L_flat, dim):
    """
    Convert a flat vector of length dim*(dim+1)/2 into a lower-triangular matrix.
    """
    L = np.zeros((dim, dim))
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            L[i, j] = L_flat[idx]
            idx += 1
    return L

def propagate_known_targets_over_episode(last_env):
    """
    Extract all known targets from last_env and propagate their states forward
    over one episode duration using the correct motion model and parameters.

    Returns:
        traj (dict):
            {
                t: {tgt_id: x_t (4D vector)},
                ...
            }
    """
    # Episode duration in steps
    T = last_env.max_steps
    dt = last_env.dt
    rng = last_env.rng

    # Known targets are those with known_mask == True
    known_ids = np.where(last_env.known_mask)[0]

    # storage: trajectories[t][id] = position vector
    traj = {}

    # Extract initial states for each known target
    # Observation encoding: consecutive blocks of obs_dim_per_target
    obs = last_env._get_obs()   # use env’s getter (better than last obs)
    d = last_env.obs_dim_per_target
    targets = {}

    for tgt_id in known_ids:
        # Extract the slice for that target
        start = tgt_id * d
        end   = start + d

        block = obs[start:end]

        # First d_state entries are the mean vector x
        x0 = block[:last_env.d_state].copy()

        # Next entries represent the Cholesky factor L packed row-wise
        chol_size = last_env.cholesky_size
        L_flat = block[last_env.d_state:last_env.d_state + chol_size]
        L = unpack_cholesky(L_flat, last_env.d_state)
        P0 = L @ L.T

        # Store the current state for propagation
        targets[tgt_id] = {
            "x": x0,
            "P": P0,
            "motion": last_env.motion_model[tgt_id],
            "param":  last_env.motion_params[tgt_id],
        }

    # ---- Propagation loop ----
    for t in range(T):
        traj[t] = {}

        for tgt_id in known_ids:
            tgt = targets[tgt_id]

            x, P = MultiTargetEnv.propagate_target_2D(
                tgt["x"], tgt["P"], last_env.Q0,
                dt=dt, rng=rng,
                motion_model=tgt["motion"],
                motion_param=tgt["param"]
            )

            # store updated state
            targets[tgt_id]["x"] = x
            targets[tgt_id]["P"] = P

            traj[t][tgt_id] = x.copy()

    return traj

def compute_state_differences_with_ekf(traj_pred, last_episode_log, last_env, R):
    """
    Apply EKF update to the predicted trajectories and compare with the
    logged 'posterior' states in last_episode_log.

    Inputs:
        traj_pred: dict[t][tgt_id] = predicted state
        last_episode_log: dict[t][tgt_id] = logged EKF-like state
        last_env: env instance (for P0, dynamics, dt)
        R: measurement noise covariance (2x2)

    Returns:
        diffs: dict[t][tgt_id] = (x_upd - x_pred)
        compared: dict[t][tgt_id] = (x_upd - x_log)
    """

    diffs = {}
    compared = {}

    for t in traj_pred.keys():
        diffs[t] = {}
        compared[t] = {}

        for tgt_id, x_pred in traj_pred[t].items():

            # We do not have P_pred stored, but environments usually propagate P similarly.
            # You should store P when generating traj_pred — if not, assume P0.
            # Here: fallback to P0
            P_pred = last_env.P0.copy()

            # EKF update
            x_upd, P_upd = MultiTargetEnv.ekf_update(
                x_pred.copy(),
                P_pred.copy(),
                R, 
                MultiTargetEnv.extract_measurement_bearingRange
            )

            # Difference: updated EKF - predicted
            diffs[t][tgt_id] = x_upd - x_pred

            # If we have a logged state, compare as well
            if t in last_episode_log and tgt_id in last_episode_log[t]:
                x_log = last_episode_log[t][tgt_id]["state"]
                compared[t][tgt_id] = x_upd - x_log

    return diffs, compared

def constant_obs_all_targets(estimates=None):

    tracks = {tid: [] for tid in estimates.keys()}

    for target_id, entry in estimates.items():

        if not isinstance(entry, dict):
            continue

        Xest = entry.get("Xest")
        P_mat = entry.get("P_mat")

        if Xest is None or P_mat is None:
            continue

        L = Xest.shape[1]

        for k in range(L):

            tracks[target_id].append({
                "t": k,
                "state": Xest[:, k],
                "cov": P_mat[:, :, k]
            })

    return tracks

def extract_tracks_from_log(last_episode_log):
    """
    Convert last_episode_log into per-target time-ordered tracks.

    Returns:
        tracks: dict[target_id] = list of dicts with
            { "t": timestep, "state": ..., "cov": ..., "exceedFOV": ... }
    """

    # Discover all target IDs present in the log
    all_target_ids = set()
    for snapshot in last_episode_log.values():
        if isinstance(snapshot, dict):
            all_target_ids.update(snapshot.keys())

    # Prepare empty track containers for discovered targets
    tracks = {tid: [] for tid in all_target_ids}

    # Iterate through sorted timesteps
    for t in sorted(last_episode_log.keys()):
        snapshot = last_episode_log[t]

        # snapshot might be {} if no target logged at this timestep
        if not isinstance(snapshot, dict):
            continue

        for target_id, entry in snapshot.items():
            # Safety check: skip malformed entries
            if not isinstance(entry, dict):
                continue

            tracks[target_id].append({
                "t": t,
                "state": entry.get("state"),
                "cov": entry.get("cov"),
                "exceedFOV": entry.get("exceedFOV", False),
            })

    return tracks

def estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc):
    errors_all_targets = []
    total_trace_cov = []
    KFstate = []
    KFcov = []
    #for tgt_id, track in tracks.items():
    for tgt_id in range(len(all_target_states)):
        #print("Plot errors for target " + str(tgt_id))

        # --- Extract data from track ---
        timesteps = [obs["t"] for obs in tracks[tgt_id]]
        #timesteps = np.arange(len(all_meas[tgt_id]))
        #if not timesteps and tgt_id >= last_env.init_n_targets:   # if not detected and initially unknown
        if tgt_id >= last_env.init_n_targets:   # if not detected and initially unknown
            continue
        all_tgt_meas = all_meas[tgt_id]
        tgt_meas = all_tgt_meas[timesteps, :]
        if tgt_id < last_env.init_n_targets:
            for tgt in last_env.targets.copy():
                    tid = tgt['id']
                    if tid == tgt_id:
                        motion = last_env.motion_model[tgt_id]
                        if motion == "L":
                            f_dyn = lambda x, omega=None: f_cv_cont(x)
                            Fx_dyn = lambda x, omega=None: Fx_cv_cont(x)
                            inputs = {
                                "Rk": R,
                                #"Q": np.eye(2) * 1e-27,
                                "Q": np.eye(2) * 0.,
                                "Po": tgt['P'],
                                #"omega": omega,
                                "f_dyn": f_dyn,
                                "Fx_dyn": Fx_dyn
                            }
                            integrationFcn = KalmanFilter.int_constant_velocity_stm
                            
                        else:
                            omega = last_env.motion_params[tgt_id]
                            f_dyn = lambda x, omega: f_ct_linear(x, omega)
                            Fx_dyn = lambda x, omega: Fx_ct_linear(x, omega)
                            inputs = {
                                "Rk": R,
                                #"Q": np.eye(2) * 5e-13,
                                "Q": np.eye(2) * 0,
                                "Po": tgt['P'],
                                "omega": omega,
                                "f_dyn": f_dyn,
                                "Fx_dyn": Fx_dyn
                            }
                            integrationFcn = KalmanFilter.int_constant_turn_stm_2D
                        break
        else:
            for tgt in last_env.unknown_targets.copy():
                tid = tgt['id']
                if tid == tgt_id:
                    motion = last_env.motion_model[tgt_id]
                    if motion == "L":
                        f_dyn = lambda x, omega=None: f_cv_cont(x)
                        Fx_dyn = lambda x, omega=None: Fx_cv_cont(x)
                        inputs = {
                            "Rk": R,
                            #"Q": np.eye(2) * 1e-27,
                            "Q": np.eye(2) * 0,
                            "Po": tgt['P'],
                            "f_dyn": f_dyn,
                            "Fx_dyn": Fx_dyn
                        }
                        integrationFcn = KalmanFilter.int_constant_velocity_stm
                        
                    else:
                        omega = last_env.motion_params[tgt_id]
                        f_dyn = lambda x, omega: f_ct_linear(x, omega)
                        Fx_dyn = lambda x, omega: Fx_ct_linear(x, omega)
                        inputs = {
                            "Rk": R,
                            #"Q": np.eye(2) * 5e-13,
                            "Q": np.eye(2) * 0,
                            "Po": tgt['P'],
                            "omega": omega,
                            "f_dyn": f_dyn,
                            "Fx_dyn": Fx_dyn
                        }
                        integrationFcn = KalmanFilter.int_constant_turn_stm_2D
                    break

        t_all, Xk_mat, P_mat, resids = KalmanFilter.ckf_predict_update(
                                        Xo_ref = tgt['x'],          # shape (4,)
                                        t_obs  = timesteps,         # shape (L,)
                                        tend   = last_env.max_steps,
                                        obs    = tgt_meas,          # shape (p, L)
                                        intfcn = integrationFcn,
                                        H_fcn  = obsFunc,
                                        inputs = inputs
                                    )
        all_tgt_states = all_target_states[tgt_id]
        tgt_states_with_velocity = all_tgt_states[t_all, :]
        tgt_states = tgt_states_with_velocity.T[:4, :]
        trace_P_pos = np.trace(P_mat[0:2, 0:2, :], axis1=0, axis2=1)
        if timesteps:
            selected_traces = trace_P_pos[timesteps[0]:]
        else:
            selected_traces = trace_P_pos
        total_trace_cov.append(selected_traces)

        """ plt.figure()
        plt.plot(t_all, three_sigma_pos, label="+3σ x")
        plt.plot(t_all, -three_sigma_pos, label="-3σ x")
        plt.scatter(t_all, pos_error, label="Positional error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show() """

        """ # Skip this target if there are no timesteps/measurements
        if not timesteps:   # or: if len(timesteps) == 0:
            continue
        else: """
            
        Xk, Pk, resids = KalmanFilter.ekf(
                        Xo_ref = tgt['x'],          # shape (4,)
                        t_obs  = timesteps,         # shape (L,)
                        obs    = tgt_meas,          # shape (p, L)
                        intfcn = integrationFcn,
                        H_fcn  = obsFunc,   
                        inputs = inputs
                    )
        all_tgt_states = all_target_states[tgt_id]
        tgt_states_with_velocity = all_tgt_states[timesteps, :]
        tgt_states = tgt_states_with_velocity.T[:4, :]
        error = tgt_states - Xk
        errors_all_targets.append(error)
        """ pos_error = np.sqrt((tgt_states[0, :] - Xk[0, :])**2 + (tgt_states[1, :] - Xk[1, :])**2)
        three_sigma_x  = 3.0 * np.sqrt(Pk[0, 0, :])
        three_sigma_y  = 3.0 * np.sqrt(Pk[1, 1, :])
        three_sigma_vx = 3.0 * np.sqrt(Pk[2, 2, :])
        three_sigma_vy = 3.0 * np.sqrt(Pk[3, 3, :])
        three_sigma_pos = 3.0 * np.sqrt(Pk[0,0,:] + Pk[1,1,:])

        t = timesteps """

        KFstate.append(Xk)
        KFcov.append(Pk)

        """ plt.figure()
        plt.plot(t, three_sigma_x, label="+3σ x")
        plt.plot(t, -three_sigma_x, label="-3σ x")
        plt.scatter(t, error[0, :], label="X error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(t, three_sigma_y, label="+3σ x")
        plt.plot(t, -three_sigma_y, label="-3σ x")
        plt.scatter(t, error[1, :], label="Y error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show() """

        """ plt.figure()
        plt.plot(t, three_sigma_pos, label="+3σ x")
        plt.plot(t, -three_sigma_pos, label="-3σ x")
        plt.scatter(t, pos_error, label="Positional error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show() """

    return errors_all_targets, total_trace_cov, KFstate, KFcov

def computeRMSEalgo(heuristic=False, random=False, model=None, env=None, n_episodes=100, sigma_theta=0, sigma_r=0, R=None, Q=None):
    error_episodes = []
    total_error_episodes = []
    for i in range(n_episodes):
        if heuristic or random:
            det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = evaluate_agent_track(env, n_episodes=1, random_policy=random, deterministic_policy=heuristic)
        elif model=="ppo":
            ppo_model = PPO.load("agents/ppo_track_trained_IEEE", env=env)
            ppo_rewards, exceedFOV_ppo, last_env, last_episode_log, illegal_actions_ppo = evaluate_agent_track(env, model=ppo_model, n_episodes=1)

        elif model=="maskableppo":
            maskppo_model = MaskablePPO.load("agents/maskableppo_track_trained_IEEE", env=env)
            maskppo_rewards, exceedFOV_maskppo, last_env, last_episode_log, illegal_actions_maskppo = evaluate_agent_track(env, model=maskppo_model, n_episodes=1, maskable=True)

        else:
            dqn_model = DQN.load("agents/dqn_track_trained_IEEE", env=env)
            dqn_rewards, exceedFOV_dqn, last_env, last_episode_log, illegal_actions_dqn = evaluate_agent_track(env, model=dqn_model, n_episodes=1)

        tracks = extract_tracks_from_log(last_episode_log)
        # Generate truth data
        timesteps = np.arange(last_env.max_steps)
        all_target_states = {}
        all_meas = {}
        for tgt_id in last_env.targets:
            tid = tgt_id["id"]
            truth_states, measurements, _ = generate_truth_states(timesteps, tid, last_env)
            all_target_states[tid] = truth_states
            measurements[:, 0] += np.random.normal(0, sigma_theta, size=len(measurements))
            measurements[:, 1] += np.random.normal(0, sigma_r, size=len(measurements))
            all_meas[tid] = measurements

        errors_all_targets, total_trace_cov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, Q)

        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodes.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)
        """ total_episode_error = 0
        for tgt_error in total_error_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            total_rmse_target = np.sqrt(np.mean(sq_err))
            total_episode_error = total_episode_error + total_rmse_target """
        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodes.append(total_rmse_all_target)
        env.reset()

    error_episodes = np.array(error_episodes)
    total_error_episodes = np.array(total_error_episodes)

    if len(error_episodes)>0:

        mean_pos_error_all_episodes = sum(error_episodes)/len(error_episodes)
        print("Mean of positional errors over all episodes " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodes], ddof=1))
    if len(total_error_episodes)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodes)/len(total_error_episodes)
        print("Mean of covariance trace over all episodes " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodes], ddof=1))
    return error_episodes, total_error_episodes


def main():

    CONFIG = {
        "search": {
            "model_class": PPO,
            "model_path": "agents/ppo_search_trained_slowTargets_obsSpace4Channels",
            "test_fn": None,
        },
        "track": {
            "model_class": MaskablePPO,
            "model_path": "agents/maskableppo_track_trained_IEEE_randomSpawn",
            "test_fn": run_random_policy_track,
        }
    }

    mode = "track"  # can also switch to "track" 

    cfg = CONFIG[mode]

    env = MultiTargetEnv(
        n_targets=5,
        n_unknown_targets=100,
        seed=None,
        mode=mode
    )

    # Optional test hook
    if cfg["test_fn"]:
        positions, covariances = cfg["test_fn"](env, n_steps=10)
        plot_positions(positions, env)

    # Load model
    model = cfg["model_class"].load(cfg["model_path"], env=env)

    visualize_trained_agent(env, model, n_steps=10)
    
 
if __name__ == "__main__":
    main()
    #computeKL()