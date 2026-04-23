import numpy as np
from scipy.integrate import solve_ivp

from LSBatchFilter import propagate_state_and_stm

@staticmethod
def ekf(Xo_ref, t_obs, obs, intfcn, H_fcn, inputs):
    """
    Least Squares Conventional Kalman Filter (CKF)

    Parameters
    ----------
    Xo_ref : (n,) ndarray
        Reference state at t=0
    t_obs : (L,) ndarray
        Observation times
    obs : (p, L) ndarray
        Observations
    intfcn : callable
        Dynamics + STM ODE function: f(t, state, inputs)
    H_fcn : callable
        Measurement model: H_fcn(Xref, inputs, t)
    inputs : object or dict
        Must contain Rk, Q, Po

    Returns
    -------
    Xk_mat : (n, L) ndarray
        Estimated states
    P_mat : (n, n, L) ndarray
        State covariance matrices
    resids : (p, L) ndarray
        Post-fit residuals
    """

    # Dimensions
    L = obs.shape[0]
    p = obs.shape[1]
    n = Xo_ref.size

    # Inputs
    Rk = inputs["Rk"]
    Q  = inputs["Q"]
    Po_bar = inputs["Po"]
    f_dyn = inputs["f_dyn"]     
    Fx_dyn = inputs["Fx_dyn"]

    # Initialization
    xhat = np.zeros(n)
    P = Po_bar.copy()
    Xref = Xo_ref.copy()

    resids = np.zeros((p, L))
    Xk_mat = np.zeros((n, L))
    P_mat = np.zeros((n, n, L))

    # STM initialization
    phi0 = np.eye(n)

    # ODE tolerances
    rtol = 1e-6
    atol = 1e-8

    count = 0
    t_prev = 0

    # Kalman filter loop
    for k in range(L):
        tk = t_obs[k]
        Yk = obs[k,:]
        """ t = t_obs[k]
        t_prior = 0.0 if k == 0 else t_obs[k - 1]
        delta_t = t - t_prior

        Yk = obs[k,:]

        # Save priors
        Xref_prior = Xref.copy()
        xhat_prior = xhat.copy()
        P_prior = P.copy()

        # -------------------------
        # Step B: Integrate Xref & STM
        # -------------------------
        int0 = np.hstack((Xref_prior, phi0_v))

        if t == t_prior:
            xout = int0
        else:
            sol = solve_ivp(
                fun=lambda tau, y: intfcn(tau, y, inputs),
                t_span=(t_prior, t),
                y0=int0,
                rtol=ode_tol,
                atol=ode_tol
            )
            xout = sol.y[:, -1]

        Xref = xout[:n]
        phi = xout[n:].reshape((n, n)) """
        if "omega" in inputs:
            omega = inputs["omega"]
        else:
            omega = 0
        Xref, Phi = propagate_state_and_stm(
            t0=t_prev,
            t1=tk,
            x0=Xref,
            Phi0_vec=phi0.reshape(-1),
            f_dyn=f_dyn,
            Fx_dyn=Fx_dyn,
            n=n,
            omega=omega,
            rtol=rtol,
            atol=atol
        )
        """ else:
            Phi = np.eye(n) """

        # -------------------------
        # Step C: Time update
        # -------------------------
        """ Gamma = np.block([
            [(delta_t**2 / 2) * np.eye(2)],
            [delta_t * np.eye(2)]
        ]) """

        """ xbar = phi @ xhat_prior
        Pbar = phi @ P_prior @ phi.T + Gamma @ Q @ Gamma.T
        print(Q) """
        xbar = Phi @ xhat
        Pbar = Phi @ P @ Phi.T #+ Q

        # -------------------------
        # Step D: Measurement update
        # -------------------------
        Hk_til, Gk = H_fcn(Xref)
        yk = Yk - Gk

        S = Hk_til @ Pbar @ Hk_til.T + Rk
        Kk = Pbar @ Hk_til.T @ np.linalg.inv(S)

        # -------------------------
        # Step E: State & covariance update
        # -------------------------
        xhat = xbar + Kk @ (yk - Hk_til @ xbar)

        I = np.eye(n)
        P = (I - Kk @ Hk_til) @ Pbar @ (I - Kk @ Hk_til).T + Kk @ Rk @ Kk.T

        # Post-fit residuals
        Xk = Xref + xhat
        resids[:,k] = yk - Hk_til @ xhat

        # Save outputs
        Xk_mat[:, k] = Xk
        P_mat[:, :, k] = P

        # EKF
        """ if k != 0:
            if np.linalg.norm(xhat) < converged and prev_norm < np.linalg.norm(xhat) and count>5:
                #print(np.linalg.norm(xhat))
                Xref = Xk.copy()
                xhat = np.zeros(n)
                count = 0 """


        count += 1
        t_prev = tk

    return Xk_mat, P_mat, resids

def ckf_predict_update(Xo_ref, t_obs, tend, obs, intfcn, H_fcn, inputs):
    import numpy as np
    from scipy.integrate import solve_ivp

    # Dimensions
    n = Xo_ref.size
    p = obs.shape[1]

    # Inputs
    Rk = inputs["Rk"]
    Q  = inputs["Q"]
    Po = inputs["Po"]

    # Build dense time grid
    t_all = np.arange(tend)
    L_all = len(t_all)

    # Map measurements to times
    meas_dict = {t_obs[k]: obs[k, :] for k in range(len(t_obs))}

    # Initialization
    xhat = np.zeros(n)
    P = Po.copy()
    Xref = Xo_ref.copy()

    # Storage
    Xk_mat = np.zeros((n, L_all))
    P_mat = np.zeros((n, n, L_all))
    resids = np.full((p, L_all), np.nan)

    # STM initialization
    phi0 = np.eye(n).reshape(n * n)

    ode_tol = 1e-12

    # Main loop over *all* time steps
    for k in range(L_all):

        t = t_all[k]
        t_prior = t if k == 0 else t_all[k - 1]
        dt = t - t_prior

        # -------------------------
        # Propagation
        # -------------------------
        int0 = np.hstack((Xref, phi0))

        if dt > 0:
            sol = solve_ivp(
                fun=lambda tau, y: intfcn(tau, y, inputs),
                t_span=(t_prior, t),
                y0=int0,
                rtol=ode_tol,
                atol=ode_tol
            )
            xout = sol.y[:, -1]
        else:
            xout = int0

        Xref = xout[:n]
        phi = xout[n:].reshape((n, n))

        # Process noise (2D CV)
        Gamma = np.block([
            [(dt**2 / 2) * np.eye(2)],
            [dt * np.eye(2)]
        ])

        xbar = phi @ xhat
        Pbar = phi @ P @ phi.T + Gamma @ Q @ Gamma.T

        # -------------------------
        # Measurement update (if available)
        # -------------------------
        if t in meas_dict:
            Yk = meas_dict[t]

            Hk_til, Gk = H_fcn(Xref)
            #Gk = np.array([theta, r])
            yk = Yk - Gk

            S = Hk_til @ Pbar @ Hk_til.T + Rk
            Kk = Pbar @ Hk_til.T @ np.linalg.inv(S)

            xhat = xbar + Kk @ (yk - Hk_til @ xbar)
            I = np.eye(n)
            P = (I - Kk @ Hk_til) @ Pbar @ (I - Kk @ Hk_til).T + Kk @ Rk @ Kk.T

            resids[:, k] = yk - Hk_til @ xhat

        else:
            # No measurement → prediction only
            xhat = xbar
            P = Pbar

        # -------------------------
        # Save estimate
        # -------------------------
        Xk_mat[:, k] = Xref + xhat
        P_mat[:, :, k] = P

    return t_all, Xk_mat, P_mat, resids

def int_constant_velocity_stm(t, X, inputs):
    """
    Continuous-time constant velocity model with STM integration

    State ordering:
        X = [x, y, vx, vy, Phi(0,0), ..., Phi(3,3)]
    """

    n = 4

    # Continuous-time system matrix
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # Break out state and STM
    x = X[:n]
    phi = X[n:].reshape((n, n))

    # State derivative
    dx = A @ x

    # STM derivative
    dphi = A @ phi

    # Pack output
    dX = np.zeros_like(X)
    dX[:n] = dx
    dX[n:] = dphi.reshape(n * n)

    return dX

def int_constant_turn_stm_2D(t, X, inputs):
    """
    ODE for 2D constant‑turn + STM.
    inputs must contain 'omega' (turn rate).
    """
    n = 4
    x = X[:n].reshape(-1, 1)
    Phi = X[n:].reshape(n, n)

    omega = inputs["omega"]  # or inputs.omega if it's an object

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -omega],
        [0, 0, omega, 0]
    ])

    dxdt = A @ x
    dPhi_dt = A @ Phi

    dX = np.zeros(4 + 16)
    dX[:4] = dxdt.flatten()
    dX[4:] = dPhi_dt.flatten()

    return dX

def int_two_body_stm(t, X, inputs):
    """
    Continuous-time two-body dynamics with STM integration.

    State ordering:
        X = [x, y, z, vx, vy, vz, Phi(0,0), ..., Phi(5,5)]

    inputs must contain:
        inputs["mu"] = gravitational parameter
    """

    mu = inputs["mu"]
    n = 6

    # ------------------------------------------------------------------
    # Break out state and STM
    # ------------------------------------------------------------------
    x = X[:n]
    phi = X[n:].reshape((n, n))

    rx, ry, rz, vx, vy, vz = x

    r = np.sqrt(rx**2 + ry**2 + rz**2)
    r3 = r**3
    r5 = r3 * r**2

    # ------------------------------------------------------------------
    # State derivative (two-body)
    # ------------------------------------------------------------------
    dx = np.zeros(6)

    dx[0] = vx
    dx[1] = vy
    dx[2] = vz
    dx[3] = -mu * rx / r3
    dx[4] = -mu * ry / r3
    dx[5] = -mu * rz / r3

    # ------------------------------------------------------------------
    # System matrix A (Jacobian of dynamics)
    # ------------------------------------------------------------------
    A = np.zeros((6, 6))

    # Position-velocity coupling
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1

    # Gravity partials
    A41 = -mu * (1/r3 - 3*rx*rx/r5)
    A42 =  3*mu*rx*ry/r5
    A43 =  3*mu*rx*rz/r5

    A52 = -mu * (1/r3 - 3*ry*ry/r5)
    A53 =  3*mu*ry*rz/r5

    A63 = -mu * (1/r3 - 3*rz*rz/r5)

    A[3, 0] = A41
    A[3, 1] = A42
    A[3, 2] = A43

    A[4, 0] = A42
    A[4, 1] = A52
    A[4, 2] = A53

    A[5, 0] = A43
    A[5, 1] = A53
    A[5, 2] = A63

    # ------------------------------------------------------------------
    # STM derivative
    # ------------------------------------------------------------------
    dphi = A @ phi

    # ------------------------------------------------------------------
    # Pack output
    # ------------------------------------------------------------------
    dX = np.zeros_like(X)
    dX[:n] = dx
    dX[n:] = dphi.reshape(n * n)

    return dX

def extract_measurement(X):
    """
    X = [x, y, z, vx, vy, vz]
    generate simulated measurement wrt sensor placed in Earth's center

    Returns:
        Hk_til (3x6 numpy array)
        Gk     (3,) numpy array (ra, dec, range)
    """

    
    posX, posY, posZ = X[0], X[1], X[2]

    r = np.sqrt(posX**2 + posY**2 + posZ**2)

    r2 = r**2
    r3 = r**3
    posX2 = posX**2
    posY2 = posY**2
    posZ2 = posZ**2

    # Measurement vector
    ra = np.arctan2(posY, posX)
    dec = np.arcsin(posZ / r)
    Gk = np.array([ra, dec, r])

    # Jacobian elements
    H11 = -posY / (posX2 + posY2)
    H12 =  posX / (posX2 + posY2)

    denom_xy = np.sqrt(posX2 + posY2)

    H21 = -posZ * posX / (r2 * denom_xy)
    H22 = -posZ * posY / (r2 * denom_xy)
    H23 = (1/r - posZ2/r3) / np.sqrt(1 - posZ2/r2)

    H31 = posX / r
    H32 = posY / r
    H33 = posZ / r

    # Build Jacobian (3x6)
    Hk_til = np.zeros((3, 6))
    Hk_til[0, :3] = [H11, H12, 0.0]
    Hk_til[1, :3] = [H21, H22, H23]
    Hk_til[2, :3] = [H31, H32, H33]

    return Hk_til, Gk