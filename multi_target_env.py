import math
import gymnasium as gym
import numpy as np
from math import sqrt
from scipy.stats import norm
from scipy.linalg import expm
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from scipy.integrate import dblquad

class MultiTargetEnv(gym.Env):
    def __init__(self, n_targets=5, n_unknown_targets=100, space_size=100.0, d_state=4, fov_size=4.0, max_steps=100, seed=None, mode="combined"):
        super().__init__()

        # ── Target counts ──────────────────────────────────────────────────────────
        # Store both the initial counts (used for resets) and mutable runtime counts
        self.init_n_targets = n_targets
        self.init_n_unknown_target = n_unknown_targets
        self.n_targets = n_targets              # Known targets (observable from the start)
        self.n_unknown_targets = n_unknown_targets  # Hidden targets discovered during episode

        # ── Environment configuration ──────────────────────────────────────────────
        self.mode = mode                        # Operating mode: "search", "track", or "combined"
        self.fov_size = fov_size                # Side length of the square field-of-view (world units)
        self.space_size = space_size            # Side length of the square world (world units)
        self.d_state = d_state                  # State dimension per target: (x, y, vx, vy)
        self.max_steps = max_steps              # Maximum timesteps before episode termination
        self.dt = 1.0                           # Simulation timestep (seconds)
        self.threshold_fov = 0.5               # Fraction of FOV overlap required to count as "seen"

        # ── Episode diagnostics ────────────────────────────────────────────────────
        self.lost_counter = 0                   # Tracks how many targets were lost this episode
        self.detect_counter = 0                 # Tracks how many detections occurred this episode

        # ── RNG and numerical stability ────────────────────────────────────────────
        self.rng = np.random.default_rng(seed)  # Seeded RNG for reproducibility
        self.boundary = np.sqrt(2.0e-8)         # Small epsilon to avoid boundary singularities

        # ── Measurement noise covariance (R) ──────────────────────────────────────
        # R is a 2×2 diagonal matrix representing sensor noise in x and y
        sigma_theta = np.deg2rad(1.0)           # Bearing noise std dev (1 degree → radians; unused directly)
        sigma_r = 0.001                         # Range noise std dev (1 cm)
        sigma = sigma_r                         # Active noise term (range noise used for both axes)
        self.R = np.diag([sigma**2, sigma**2])  # Isotropic position noise covariance

        # ── Motion model per target ────────────────────────────────────────────────
        # Each target is independently assigned a motion model and a corresponding parameter
        self.motion_model = self.rng.choice(
            ["L", "T"],
            size=self.n_targets + self.n_unknown_targets
        )  # "L" = linear (constant velocity), "T" = coordinated turn
        self.motion_params = self.rng.uniform(
            0.05, 0.3,
            size=self.n_targets + self.n_unknown_targets
        ) / 10  # Linear targets: speed magnitude; turn targets: turn rate (rad/s)

        # ── Observation packing (Cholesky upper-triangle of covariance) ────────────
        # The covariance matrix P (d_state × d_state) is packed into its upper-triangle
        # to reduce observation dimensionality while retaining full covariance information
        self.cholesky_size = d_state * (d_state + 1) // 2  # Number of upper-triangle elements
        self.obs_dim_per_target = d_state + self.cholesky_size  # State mean + packed covariance
        self.max_targets = self.init_n_targets + self.init_n_unknown_target  # Total possible targets

        # ── Initial covariance matrices ────────────────────────────────────────────
        # P0: initial state covariance for new tracks (high uncertainty in velocity)
        self.P0 = np.eye(self.d_state) * 0.1
        self.P0[-2:, -2:] = np.eye(2) * 0.001  # Lower uncertainty on the velocity components
        # Q0: process noise covariance (models unmodelled accelerations / manoeuvres)
        self.Q0 = np.eye(self.d_state) * 1e-1

        # ── Spatial grid (discretised FOV coverage map) ────────────────────────────
        # The world is divided into a uniform grid of FOV-sized cells for coverage tracking
        n_grid = max(1, int(np.floor(self.space_size / self.fov_size)))  # Cells per axis
        self.n_grid_cells = n_grid * n_grid                              # Total cell count

        # Cell centres, offset inward by half a FOV so cells don't exceed world bounds
        x_vals = np.linspace(
            -self.space_size / 2 + self.fov_size / 2,
            self.space_size / 2 - self.fov_size / 2,
            n_grid
        )
        y_vals = np.linspace(
            -self.space_size / 2 + self.fov_size / 2,
            self.space_size / 2 - self.fov_size / 2,
            n_grid
        )
        self.grid_coords = np.array([[x, y] for x in x_vals for y in y_vals])  # (n_grid_cells, 2)
        self.visit_counts = np.zeros(self.n_grid_cells, dtype=int)              # Times each cell visited

        # ── Spatial information maps ───────────────────────────────────────────────
        self.detection_history = np.zeros((n_grid, n_grid), dtype=np.float32)  # Cumulative detection density
        self.recency_map = np.zeros((n_grid, n_grid), dtype=np.float32)        # Decay-weighted recency of visits
        self.roi_mask = self._build_roi_mask()                                  # Binary mask of regions of interest

        # ── Observation space ──────────────────────────────────────────────────────
        if mode == "search":
            # Four spatial channels stacked: e.g. visit density, recency, ROI mask, and one more
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(4, n_grid, n_grid),
                dtype=np.float32
            )

        if mode == "track":
            # Each target is represented by two scalars: filter trace and existence probability
            self.obs_dim_per_target = 2
            self.observation_space = gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.max_targets, self.obs_dim_per_target),
                dtype=np.float32
            )

        # ── Action space ───────────────────────────────────────────────────────────
        if self.mode == "search":
            self.n_actions = self.n_grid_cells          # Select which grid cell to visit next
        elif self.mode == "track":
            self.n_actions = self.max_targets           # Select which target to focus the sensor on
        else:  # "combined" — deprecated, kept for backwards compatibility
            self.n_actions = self.n_grid_cells + self.max_targets

        self.action_space = gym.spaces.Discrete(self.n_actions)

        # ── Known/unknown target mask ──────────────────────────────────────────────
        # Boolean mask indicating which slots in the target array correspond to known targets;
        # the remainder are unknown targets that may be discovered during the episode
        self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.n_targets] = True  # First n_targets slots are known at episode start

        self.reset(seed=seed)

    def _build_roi_mask(self):
        n_grid = int(np.sqrt(self.n_grid_cells))
        mask = np.zeros((n_grid, n_grid), dtype=np.float32)
        for grid_idx, search_pos in enumerate(self.grid_coords):
            x_index = grid_idx // n_grid
            y_index = grid_idx % n_grid
            if search_pos[1] > 0:
                mask[x_index, y_index] = 1.0
        return mask
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Propagate seed to the parent Gym environment

        # ── Episode counters ───────────────────────────────────────────────────────
        self.step_count = 0                                 # Current timestep within the episode
        self.n_targets = self.init_n_targets                # Restore known target count to initial value
        self.n_unknown_targets = self.init_n_unknown_target # Restore unknown target count to initial value
        self.lost_counter = 0                               # Reset lost-target tally
        self.detect_counter = 0                             # Reset detection event tally

        # ── Motion model re-sampling ───────────────────────────────────────────────
        # Re-draw motion models and parameters each episode so targets behave differently per run
        self.motion_model = self.rng.choice(
            ["L", "T"],
            size=self.n_targets + self.n_unknown_targets
        )  # "L" = linear (constant velocity), "T" = coordinated turn
        self.motion_params = self.rng.uniform(
            0.05, 0.3,
            size=self.n_targets + self.n_unknown_targets
        )  # Speed magnitude (L) or turn rate in rad/s (T)

        # ── Target initialisation ──────────────────────────────────────────────────
        # Known targets occupy indices [0, init_n_targets);
        # unknown targets are offset so their indices don't collide with known ones
        self.targets = [self._init_target(i) for i in range(self.init_n_targets)]
        self.unknown_targets = [
            self._init_unknown_target(i + self.init_n_targets)
            for i in range(self.init_n_unknown_target)
        ]

        # ── Spatial coverage maps ──────────────────────────────────────────────────
        self.visit_counts[:] = 0        # Number of times each grid cell has been visited
        self.detection_history[:] = 0.0 # Cumulative detection density per cell
        self.recency_map[:] = 0.0       # Decay-weighted recency of visits per cell

        # ── Episode reward accumulators ────────────────────────────────────────────
        # These track reward breakdowns for logging/diagnostics, not for training
        self.episode_detection_reward = 0.0  # Total reward earned from detections
        self.episode_roi_reward = 0.0        # Total reward earned from visiting ROI cells
        self.episode_n_detections = 0        # Total number of detection events this episode

        # ── Search memory ──────────────────────────────────────────────────────────
        # Used to compute step-to-step search displacement rewards or avoid revisits
        self.last_search_idx = None   # Grid index of the most recent search action
        self.prev_search_pos = None   # World-space position of the most recent search action

        # ── Known/unknown target mask ──────────────────────────────────────────────
        # Rebuild from scratch each reset; first n_targets slots are known at episode start
        self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.n_targets] = True

        # ── Initial observation ────────────────────────────────────────────────────
        self.obs = self._get_obs()
        info = {}  # Placeholder — populate with diagnostics if needed by wrappers
        return self.obs, info
    
    # Action decoding logic 
    def decode_action(self, action_int):
        """Convert flat integer into (macro, micro_search, micro_track)."""
        if self.mode == "search":
            macro = 0
            micro_search = action_int
            micro_track = None
        elif self.mode == "track":
            macro = 1
            micro_search = None
            micro_track = action_int
        else:  # combined - depreciated
            if action_int < self.n_grid_cells:
                macro = 0
                micro_search = action_int
                micro_track = None
            else:
                macro = 1
                micro_search = None
                micro_track = action_int - self.n_grid_cells
        return macro, micro_search, micro_track
    
    def encode_action(self, macro, micro_search=None, micro_track=None):
        """
        Convert hierarchical action (macro + micro) into a flat integer.
        Mirrors decode_action().
        
        Parameters:
            macro: 0 for search, 1 for track
            micro_search: index of search grid (if macro==0)
            micro_track: target index (if macro==1)
        
        Returns:
            action_int: single integer representing the action
        """
        if macro == 0:  # SEARCH
            if micro_search is None:
                raise ValueError("micro_search must be provided for macro=0")
            return micro_search
        elif macro == 1:  # TRACK
            if micro_track is None:
                raise ValueError("micro_track must be provided for macro=1")
            if self.mode == "track":
                return micro_track
            elif self.mode == "combined":
                return self.n_grid_cells + micro_track
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            raise ValueError(f"Invalid macro action: {macro}")

    def step(self, action):
        # ── Action decoding ────────────────────────────────────────────────────────
        # Decode the flat action into a macro choice (SEARCH vs TRACK) and the
        # corresponding micro-action (grid cell index or target id)
        macro, micro_search, micro_track = self.decode_action(action)
        micro = [0., 0.]
        target_id = []
        search_pos = None
        n_grid = int(np.sqrt(self.n_grid_cells))  # Grid side length (cells per axis)

        if macro == 0:  # SEARCH: agent chose to move the sensor to a grid cell
            grid_idx = micro_search
            search_pos = self.grid_coords[grid_idx]  # World-space centre of chosen cell
            micro = search_pos
        else:           # TRACK: agent chose to update a specific known target
            target_id.append(micro_track)
            micro = micro_track

        # ── Reward component initialisation ───────────────────────────────────────
        total_iG = 0.0        # Accumulated information gain (or detection bonus) this step
        lost_reward = 0.0     # Penalty contribution from lost targets (currently unused)
        prob_reward = 0.0     # Reward from in-FOV probability of the tracked target
        lost = 0              # Binary flag: tracked target considered lost (1) or not (0)
        lost_targets = []     # Targets dropped from tracking this step
        trace_reward = 0      # Net change in total covariance trace (pre- minus post-update)

        # ── Recency map decay ──────────────────────────────────────────────────────
        # Exponential decay ensures older visits contribute less to the recency signal
        self.recency_map *= 0.99

        # ── Known target propagation ───────────────────────────────────────────────
        # Advance each known target's state estimate forward by one timestep,
        # then optionally apply an EKF measurement update if this target was tracked
        for tgt in self.targets:
            idx = tgt['id']
            model = self.motion_model[idx]
            param = self.motion_params[idx]

            # Propagate state and covariance using the target's assigned motion model
            tgt['x'], tgt['P'] = MultiTargetEnv.propagate_target_2D(
                tgt['x'], tgt['P'], tgt.get('Q', self.Q0),
                dt=self.dt, rng=self.rng,
                motion_model=model, motion_param=param
            )
            trace_reward += np.trace(tgt['P'])  # Accumulate pre-update trace contribution

            if target_id and idx == micro:
                # ── EKF update for the actively tracked target ─────────────────────
                # Subtract pre-update trace, apply measurement update, add post-update
                # trace; the net difference reflects the information gained this step
                trace_reward -= np.trace(tgt['P'])
                xUpdate, PUpdate = MultiTargetEnv.ekf_update(
                    tgt['x'], tgt['P'], self.R,
                    MultiTargetEnv.extract_measurement_XY
                )
                tgt['x'], tgt['P'] = xUpdate, PUpdate
                trace_reward += np.trace(tgt['P'])

                # Probability that the target remains within the sensor boundary post-update
                prob = compute_fov_prob_single(self.boundary, tgt['x'], tgt['P'])
                prob_reward += prob
                lost = 1 - prob  # High lost value --> target likely escaped the FOV


            else:
                # ── Neglected target: check whether it has drifted out of FOV ──────
                # Use the full FOV size (not boundary) for the maintenance probability check
                probMaintainFOV = compute_fov_prob_single(self.fov_size, tgt['x'], tgt['P'])

                if probMaintainFOV < self.threshold_fov:
                    # Target covariance has grown so large it is unlikely to be in FOV;
                    # mark as lost and remove from the active tracking list
                    lost_targets.append(tgt)
                    self._remove_lost_tracking_target(idx)

        # ── Unknown target propagation ─────────────────────────────────────────────
        # Unknown targets are propagated but never updated (no measurements taken);
        # they may later be promoted to known targets via a detection event
        for utgt in self.unknown_targets:
            idx = utgt['id']
            model = self.motion_model[idx]
            param = self.motion_params[idx]

            utgt['x'], utgt['P'] = MultiTargetEnv.propagate_target_2D(
                utgt['x'], utgt['P'], utgt.get('Q', self.Q0),
                dt=self.dt, rng=self.rng,
                motion_model=model, motion_param=param
            )

        # ══ TRACK macro: early return ══════════════════════════════════════════════
        if target_id:
            obs = self._get_obs(target_id)
            self.step_count += 1

            # Guard: penalise heavily if agent tries to track an unknown target
            if self.known_mask[target_id]:
                invalid_action = False
            else:
                invalid_action = True
                info = {
                    "invalid_action": invalid_action,
                    "action_mask": self.action_masks(),
                    "lost_target": lost_targets
                }
                done = self.step_count >= self.max_steps
                return obs, -10.0, done, False, info  # Hard penalty for invalid track action

            # Reward is the negative mean covariance trace across all known targets
            # (lower trace = tighter estimates = better tracking performance)
            reward = -trace_reward / self.n_targets

            done = self.step_count >= self.max_steps
            truncated = False

            info = {
                "macro": macro,
                "micro": micro,
                "target_id": target_id,
                "reward_info_gain": total_iG,
                "action_mask": self.action_masks(),
                "lost_target": lost_targets,
                "n_known": np.sum(self.known_mask),
                "n_lost_this_step": len(lost_targets),
                "detect_counter": self.detect_counter,
                "lost_counter": self.lost_counter
            }
            self.obs = obs
            return obs, reward, done, truncated, info

        # ══ SEARCH macro ═══════════════════════════════════════════════════════════

        # ── FOV detection sweep ────────────────────────────────────────────────────
        # Check which targets (known or unknown) fall inside the square FOV
        # centred on the chosen search position
        detections = []
        fov_halfWidth = self.fov_size / 2.0
        for obj in self.targets + self.unknown_targets:
            try:
                dx = obj['x'][0] - search_pos[0]
                dy = obj['x'][1] - search_pos[1]
                if abs(dx) <= fov_halfWidth and abs(dy) <= fov_halfWidth:
                    detections.append(obj)
            except Exception as e:
                # Malformed target state — return immediately with a strong penalty
                obs = self._get_obs() if hasattr(self, "_get_obs") else self.observation_space.sample()
                info = {
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                }
                return obs, -100.0, True, False, info

        # Convert flat grid index to 2D map coordinates for history/recency updates
        x_index = grid_idx // n_grid
        y_index = grid_idx % n_grid

        # ── Detection check (Mahalanobis gating) ─────────────────────────────
        # A detection is considered "new" if it is statistically far from every
        # currently tracked target; new detections promote an unknown to known
        if len(detections) > 0:
            threshold = 3.0  # Mahalanobis distance gate (standard deviations)
            for det in detections:
                distances = [
                    mahalanobis_distance(det['x'], known['x'], known['P'])
                    for known in self.targets
                ]
                if all(d > threshold for d in distances):
                    total_iG += 10.0  # Bonus reward for discovering a new target
                    self._add_new_tracking_target(det['id'])
                    target_id.append(det['id'])
                    # Cap detection density at 1.0 to keep the map normalised
                    self.detection_history[x_index, y_index] = min(
                        self.detection_history[x_index, y_index] + 1.0, 1.0
                    )

        # ── Coverage map updates ───────────────────────────────────────────────────
        self.recency_map[x_index, y_index] = 1.0   # Mark cell as freshly visited
        self.detection_history *= 0.95              # Decay historical detections toward zero
        self.visit_counts[grid_idx] += 1            # Increment raw visit counter for this cell

        reward = total_iG

        # Cache last search position for use by _get_obs or reward shaping next step
        self.last_search_idx = grid_idx
        self.prev_search_pos = search_pos

        # ── Exploration bonus ──────────────────────────────────────────────────────
        # Small additive bonus for visiting cells in the upper half of the world;
        # acts as a soft prior to encourage coverage of the ROI
        exploration_bonus = 0
        if search_pos[1] > 0:
            exploration_bonus += 1.0
        # note: exploration_bonus is accumulated into episode stats but not added to `reward` - it was added before but this guides the agent too much to a biased behaviour

        # ── Episode statistic accumulators ────────────────────────────────────────
        self.episode_detection_reward += total_iG
        self.episode_roi_reward += exploration_bonus
        self.episode_n_detections += len(detections)

        # ── Construct return values ────────────────────────────────────────────────
        obs = self._get_obs()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False

        info = {
            "macro": macro,
            "micro": micro,
            "target_id": target_id,
            "reward_info_gain": total_iG,
            "action_mask": self.action_masks(),
            "lost_target": lost_targets,
            "episode_detection_reward": self.episode_detection_reward,
            "episode_roi_reward": self.episode_roi_reward,
            "episode_n_detections": self.episode_n_detections
        }
        self.obs = obs

        return obs, reward, done, truncated, info

    def _init_target(self, target_id,  y_range=None):
        """Initialize target with random position in left half and fixed velocity."""
        if y_range is None:
            y_low, y_high = -self.space_size/2, self.space_size/2
        else:
            y_low, y_high = y_range

        # Sample x exclusively in left half: [-space_size/2, 0)
        x0_left = self.rng.uniform(-self.space_size/2, 0)
        y0 = self.rng.uniform(y_low, y_high)

        pos0 = np.array([x0_left, y0])
        vel0 = np.array([self.motion_params[target_id], 0.0])
        x0 = np.concatenate([pos0, vel0])
        covMultiplier = [0.7, 0.75, 0.85, 1.0]
        P0 = self.P0.copy() * np.random.choice(covMultiplier)

        Q = np.eye(self.d_state) * 0.
        return {"id": target_id, "x": x0, "P": P0, "Q": Q}
    
    def _init_unknown_target(self, target_id, x_range=None, y_range=None):
        if x_range is None:
            # Left half of the field
            x_low, x_high = -self.space_size/2, 0
        else:
            x_low, x_high = x_range
        if y_range is None:
            # Upper half of the field
            y_low, y_high = 0, self.space_size/2
        else:
            y_low, y_high = y_range

        x0 = self.rng.uniform(x_low, x_high)
        y0 = self.rng.uniform(y_low, y_high)
        pos0 = np.array([x0, y0])
        vel0 = np.array([1.0, 0.0])         # velocity for CT model
        if self.motion_model[target_id] == "L":
            vel0 = np.array([self.motion_params[target_id], 0.0])  # move rightward
        x0_full = np.concatenate([pos0, vel0])
        P0 = self.P0.copy()
        Q = np.eye(self.d_state) * 0.0
        return {"id": target_id, "x": x0_full, "P": P0, "Q": Q}
    
    def action_masks(self):
        return self.known_mask

    def _get_obs(self, target_id=None):
        """Flatten all target states and covariances into one observation vector."""

        # ══ SEARCH mode observation ════════════════════════════════════════════════
        # Returns a (4, n_grid, n_grid) tensor — one spatial channel per information source
        if self.mode == "search":
            n_grid = int(np.sqrt(self.n_grid_cells))  # Grid side length (cells per axis)

            # Channel 1 — normalised visit counts
            # Log-scaled so frequently visited cells don't dominate; +1 avoids log(0)
            visit_2d = self.visit_counts.reshape(n_grid, n_grid).astype(np.float32)
            visit_map = np.log1p(visit_2d) / np.log1p(visit_2d.max() + 1)  # Range: [0, 1]

            # Channel 2 — detection history (decaying sum of past detections)
            # Set to 1.0 on detection; decayed by ×0.95 each step in step()
            detection_map = self.detection_history.copy()

            # Channel 3 — recency map (1.0 = visited last step, exponentially decays at ×0.99/step)
            recency_map = self.recency_map.copy()

            # Channel 4 — ROI mask (static binary map: 1.0 inside region of interest, 0.0 outside)
            roi_map = self.roi_mask.copy()

            # Stack channels along axis 0 → shape: (4, n_grid, n_grid)
            return np.stack([
                visit_map,      # How often has this cell been visited?
                detection_map,  # Where have detections historically occurred?
                recency_map,    # How recently was this cell visited?
                roi_map,        # Is this cell inside the region of interest?
            ], axis=0).astype(np.float32)

        # ══ TRACK mode observation ══════════════════════════════════════════════════
        # Returns a (max_targets, 2) array — one row per target slot, two features per target

        # Build a unified lookup of all targets (known + unknown) keyed by global id
        by_id = {t["id"]: t for t in self.targets}
        by_id.update({t["id"]: t for t in self.unknown_targets})

        # Sort by id so the observation rows are always in a consistent order
        all_targets = [by_id[k] for k in sorted(by_id)]

        features = []
        for tgt in all_targets:
            trace = np.trace(tgt["P"])   # Scalar uncertainty measure: sum of state variances
            p_fov = compute_fov_prob_single(self.boundary, tgt["x"], tgt["P"])  # P(target inside FOV)
            known = 1.0 if self.known_mask[tgt["id"]] else 0.0  # Mask: 0.0 zeroes out unknown targets

            # Unknown targets contribute zeros so the agent cannot exploit their hidden state;
            # known targets expose their trace and FOV probability as tracking quality signals
            features.append([
                trace * known,  # Covariance trace (0.0 if unknown)
                p_fov * known   # FOV containment probability (0.0 if unknown)
            ])

        obs = np.stack(features, axis=0)  # shape: (max_targets, 2)
        return obs.astype(np.float32)
            
    def _add_new_tracking_target(self, unknown_idx):
        """Promote unknown_targets[unknown_idx] into known targets."""
        if len(self.targets) >= self.max_targets:
            return

        # remove from unknown list
        for i, element in enumerate(self.unknown_targets):
            if element.get('id') == unknown_idx:
                new = self.unknown_targets.pop(i)
                break
        else:
            new = None   # not found
        #new = self.unknown_targets.pop(unknown_idx-self.init_n_target)
        new_target = {
            "id": unknown_idx,
            "x": new['x'].copy(),
            "P": self.P0.copy(),
            "Q": new.get('Q', self.Q0)
        }
        self.targets.append(new_target)
        self.known_mask[unknown_idx] = True
        self.n_targets += 1
        self.n_unknown_targets -= 1
        self.detect_counter += 1

    def _remove_lost_tracking_target(self, target_id):
        """Move a known target back into unknown targets when it is lost."""
    
        # Find the matching target in the known list by ID
        for i, tgt in enumerate(self.targets):
            if tgt['id'] == target_id:
                
                # Remove from known list
                removed = self.targets.pop(i)

                # Move to unknown list (keep its covariance!)
                self.unknown_targets.append({
                    "id": target_id,
                    "x": removed["x"].copy(),
                    "P": removed["P"].copy(),
                    "Q": removed.get("Q", self.Q0)
                })

                # Update mask and counters
                self.known_mask[target_id] = False
                self.n_targets -= 1
                self.n_unknown_targets += 1
                self.lost_counter += 1
                return

    @staticmethod
    def propagate_target_2D(x, P, Q, dt, rng, motion_model="L", motion_param=1.0):
        """
        Propagate a 2D target state based on its motion model.

        Parameters:
            x : np.array
                State vector [x, y, vx, vy]
            P : np.array
                Covariance matrix
            Q : np.array
                Process noise covariance
            dt : float
                Time step
            rng : np.random.Generator
                Random number generator for stochastic noise
            motion_model : str
                "L" = linear, "T" = coordinated turn
            motion_param : float
                Linear velocity for "L", turn rate for "T"

        Returns:
            x_new, P_new : propagated state and covariance
        """
        F = None
        if motion_model == "L":
            F = MultiTargetEnv.constant_velocity_F_2D(dt)
        else:
            F = MultiTargetEnv.constant_turnrate_F_2D(dt, motion_param)
        w = rng.multivariate_normal(np.zeros(len(x)), Q)
        x_next = F @ x + w
        P_next = F @ P @ F.T + Q
        return x_next, P_next
    
    @staticmethod
    def constant_velocity_F_2D(dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def constant_turnrate_F_2D(dt, omega):
        """
        Build 4x4 F for state ordering [x, y, vx, vy].
        """

        #modify omega to increase radius of targets
        #omega = omega / 10
        wdt = omega * dt
        # handle small w via stable evaluations:
        if abs(wdt) < 1e-8:
            # Use Taylor expansions directly for the needed terms:
            # s1 = sin(wdt)/omega  ~= dt * (1 - z^2/6 + z^4/120)
            # s2 = (1-cos(wdt))/omega ~= dt*(z/2 - z^3/24 + ...)
            z = wdt
            s1 = dt * (1.0 - z*z/6.0 + z**4/120.0)
            s2 = dt * (z/2.0 - z**3/24.0 + z**5/720.0)
        else:
            s1 = np.sin(wdt) / omega
            s2 = (1.0 - np.cos(wdt)) / omega

        c = np.cos(wdt)
        s = np.sin(wdt)

        F = np.array([
            [1.0, 0.0,      s1,         s2],
            [0.0, 1.0,     -s2,         s1],
            [0.0, 0.0,       c,        -s ],
            [0.0, 0.0,       s,         c ]
        ])
        return F
    
    @staticmethod
    def extract_measurement_bearingRange(x):
        # Access first two entries (x, y)
        px, py = x[:2]

        # Compute range and bearing
        theta = np.arctan2(py, px)
        r = np.sqrt(px**2 + py**2)

        # Compute observation matrix
        H11 = -py / (px**2 + py**2)
        H12 =  px / (px**2 + py**2)
        H21 =  px / r
        H22 =  py / r

        # Full Jacobian (2x4, assuming state = [x, y, vx, vy])
        H = np.array([
            [H11, H12, 0.0, 0.0],  # Bearing partials
            [H21, H22, 0.0, 0.0]   # Range partials
        ])
        Gk = np.array([theta, r])
        return H, Gk
    
    @staticmethod
    def extract_measurement_XY(x):
        posX = x[0]
        posY = x[1]
        Gk = np.array([posX, posY])

        # Full Jacobian (2x4, assuming state = [x, y, vx, vy])
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],  # x partials
            [0.0, 1.0, 0.0, 0.0]   # y partials
        ])
        return H, Gk

    
    def ekf_update(x, P, R, obsFcn):
        """
        Perform one EKF measurement update.
        Inputs:
            x: state vector (4,)
            P: covariance matrix (4x4)
            R: measurement noise covariance (2x2)
        Returns:
            x_upd, P_upd
        """
        # Predict measurement
        H, Gk = obsFcn(x)
        
        # Innovation
        y = np.zeros_like(Gk)
        
        # Innovation covariance
        S = H @ P @ H.T + R
        
        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        
        # Updated state and covariance
        x_upd = x + K @ y
        P_upd = (np.eye(len(x)) - K @ H) @ P
        
        return x_upd, P_upd

    @staticmethod
    def compute_kl_divergence(mean_p, cov_p, mean_q, cov_q):
        """
        Compute the Kullback–Leibler divergence D_KL(P || Q) between two multivariate Gaussians.

        Parameters
        ----------
        mean_p : np.ndarray
            Mean vector of distribution P (n,)
        cov_p : np.ndarray
            Covariance matrix of distribution P (n x n)
        mean_q : np.ndarray
            Mean vector of distribution Q (n,)
        cov_q : np.ndarray
            Covariance matrix of distribution Q (n x n)

        Returns
        -------
        float
            The KL divergence D_KL(P || Q)
        """
        n = mean_p.shape[0]

        # Ensure inputs are NumPy arrays
        mean_p = np.atleast_1d(mean_p)
        mean_q = np.atleast_1d(mean_q)
        cov_p = np.atleast_2d(cov_p)
        cov_q = np.atleast_2d(cov_q)

        # Compute inverses and determinants
        inv_cov_q = np.linalg.inv(cov_q)
        det_cov_p = np.linalg.det(cov_p)
        det_cov_q = np.linalg.det(cov_q)

        # Compute trace term
        trace_term = np.trace(inv_cov_q @ cov_p)

        # Mean difference
        mean_diff = mean_q - mean_p
        mean_term = mean_diff.T @ inv_cov_q @ mean_diff

        # Log determinant ratio term
        log_det_term = np.log(det_cov_q / det_cov_p)

        # Combine terms (0.5 * [log|Σq|/|Σp| - n + Tr(invΣq Σp) + (μq - μp)^T invΣq (μq - μp)])
        d_kl = 0.5 * (log_det_term - n + trace_term + mean_term)

        return float(d_kl)

    @staticmethod
    def compute_fov_prob_full(P, fov_x, fov_y=None, rtol=1e-8):
        """
        Compute probability that a 2D Gaussian state remains inside a rectangular FOV
        using full covariance integration (no independence assumption).
        This function can be used as well to approximate the FOV probability for a sensor 
        located in Earth center pointing towards geostationary satellites.

        This function should not be used for large FOV and for satellites with large
        declination angles. 

        Parameters
        ----------
        P : 2x2 numpy array
            Position covariance matrix
        fov_x : float
            Full FOV width in x
        fov_y : float, optional
            Full FOV width in y (defaults to fov_x)
        rtol : float
            Relative tolerance for quadrature

        Returns
        -------
        prob : float
            Probability of remaining inside the FOV
        """

        if fov_y is None:
            fov_y = fov_x

        hx = fov_x / 2.0
        hy = fov_y / 2.0

        # slice P to only take into account positional/angular values
        P2 = P[0:2, 0:2]
        w, _ = np.linalg.eig(P2)

        # Ensure positive definiteness if needed
        detP = np.linalg.det(P2)
        if detP <= 0:
            raise ValueError("Covariance matrix must be positive definite")

        Pinv = np.linalg.inv(P2)

        # Gaussian exponent
        def integrand(y, x):
            quad = (
                Pinv[0, 0] * x**2
                + (Pinv[0, 1] + Pinv[1, 0]) * x * y
                + Pinv[1, 1] * y**2
            )
            return math.exp(-0.5 * quad)

        atol = 1e-13

        integral = dblquad(
            integrand,
            -hx, hx,
            lambda x: -hy,
            lambda x:  hy,
            epsabs=atol,
            epsrel=rtol
        )[0]

        prob = (1.0 / (2.0 * math.pi * math.sqrt(detP))) * integral
        #print(f"{integral}; {detP}; {prob}; {w}")
        return prob

def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)

def compute_fov_prob_single(fov, x, P):
    """
    Compute FOV-probability analytically under the assumption that the integration 
    axes are independent.
    """

    #half_fov = fov*0.75 / 2.0   # radians 
    half_fov = fov / 2.0   # radians 

    prob = 1.0
    for i in range(2):
        pos_var = P[i, i]
        pos_std = sqrt(pos_var)

        # Numerical safety
        if pos_std < 1e-8:
            pos_std = 1e-8

        # Probability that target's x-pos ∈ [-half_fov, +half_fov] ---
        # Gaussian N(0, σ)
        dist = norm(loc=0.0, scale=pos_std)
        prob *= dist.cdf(half_fov) - dist.cdf(-half_fov)
    return prob


    