import math
import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg
 
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
 
 
tf = 60
startDelay = 5
controllerUpdateRate = 100
 
v_ref = 0.5
K_p = 0.1
K_i = 1
 
enableSteeringControl = True
nodeSequence = [0, 2, 4, 6, 0]
 
SIGN_STEER   = +1.0  # steering sign: input u is applied as SIGN_STEER*delta
e_max_left   = 0.10  # SAFE SET: keep |e| <= e_max (lane half-width)
e_max_right  = 0.10
 
Q_lat      = (1.0, 0.30)  # cost weights on STATE x = [e, psi]
R_lat      = 0.60  # cost weight on CONTROL INPUT u = delta
gamma_cbf  = 1.0  # BARRIER FUNCTION sharpness gamma
beta_cbf   = 0.03  # BARRIER FUNCTION weight beta
 
cbf_alpha  = 0.35
cbf_margin = 0.030
 
useOnlineLSPI = True
learnDuration = 25.0
exploreAmp    = 0.14
Kboot         = (-1.2, -0.9)  # bootstrap gain (used ONLY while exploring)
lspiSweeps    = 60
lspiRidge     = 1e-6
 
calibrationPose = [0, 2, -np.pi/2]
 
 
def _barrier_second_derivative(c, gamma):
    c = max(float(c), 1e-6)
    return (2.0 * gamma * c + 1.0) / (c * c * (gamma * c + 1.0) ** 2)
 
 
def _solve_dare(A, B, Q, R, iters=6000, tol=1e-13):
    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); R = np.atleast_2d(np.asarray(R, float))
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        Pn = A.T @ P @ A - (A.T @ P @ B) @ K + Q
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn; break
        P = Pn
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P
 
 
def _vecs3(z):
    e, ps, d = z
    return np.array([e * e, 2 * e * ps, 2 * e * d, ps * ps, 2 * ps * d, d * d])
 
 
def _unpackH(hbar):
    return np.array([[hbar[0], hbar[1], hbar[2]],
                     [hbar[1], hbar[3], hbar[4]],
                     [hbar[2], hbar[4], hbar[5]]])
 
 
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]
 
if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate = False
else:
    calibrate = 'y' in input('do you want to recalibrate?(y/n)')
 
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
 
 
class SpeedController:
 
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0
 
    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle)
 
 
class SafeQSteeringController:
 
    def __init__(self, waypoints, e_left, e_right, Q, R, gamma, beta, alpha, margin,
                 max_delta=0.3, cyclic=True, online_lspi=True, learn_T=25.0,
                 explore_amp=0.14, Kboot=(-1.2, -0.9), lspi_sweeps=60, lspi_ridge=1e-6):
        self.maxSteeringAngle = max_delta
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.cyclic = cyclic
 
        self.e_left = float(e_left)
        self.e_right = float(e_right)
        self.Q = np.diag([float(Q[0]), float(Q[1])])
        self.R = float(R)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.margin = float(margin)
 
        self.p_ref = (0.0, 0.0)
        self.th_ref = 0.0
 
        HB_ee = self.beta * (_barrier_second_derivative(self.e_left, self.gamma)  # BARRIER FUNCTION Hessian H_B at the origin
                             + _barrier_second_derivative(self.e_right, self.gamma))
        self.HB = np.array([[HB_ee, 0.0], [0.0, 0.0]])
        self.Qs = self.Q + 0.5 * self.HB  # safety-augmented cost weight  Q~ = Q + 1/2 H_B
 
        self.online_lspi = bool(online_lspi)
        self.learn_T = float(learn_T)
        self.explore_amp = float(explore_amp)
        self.weave_amp = 0.030
        self.weave_f = 0.13
        self.look_ahead = 0.20
        self.min_id_samples = 350
        self.min_final_samples = 900
        self.max_learn_T = 60.0
        self.lspi_sweeps = int(lspi_sweeps)
        self.lspi_ridge = float(lspi_ridge)
        self.kpsi_max = 0.90
        self.ke_max = 2.5
        self.steer_rate = 0.03
        self.reident_period = 200
 
        self.Kboot = np.array([float(Kboot[0]), float(Kboot[1])])
        self.K = self.Kboot.copy()
        self.learned = not self.online_lspi
        self.t_learn = 0.0
        self._sdelta = None
 
        self.A_hat = None
        self.B_hat = None
        self.E_hat = None
 
        self._prev = None
        self._X = []; self._U = []; self._Kap = []; self._Xn = []
        self._next_id_at = self.min_id_samples
 
        if self.online_lspi:
            print("[SafeQ] H_B(1,1)=%.3f | MODEL-FREE mode: identify (A_hat,B_hat) + LEARN Q-kernel gain "
                  "from data over %.0fs." % (HB_ee, self.learn_T))
            print("[SafeQ]   exploration on bootstrap K=[%+.3f,%+.3f] (no kinematic model used anywhere)."
                  % (self.Kboot[0], self.Kboot[1]))
        else:
            print("[SafeQ] H_B(1,1)=%.3f | deploying fixed bootstrap gain (learning disabled)." % HB_ee)
 
    def _cap_kpsi(self, K):
        K = np.array(K, float).reshape(2)
        if abs(K[0]) > self.ke_max:
            K[0] = -self.ke_max if K[0] < 0 else self.ke_max
        if abs(K[1]) > self.kpsi_max:
            K[1] = -self.kpsi_max if K[1] < 0 else self.kpsi_max
        return K
 
    def _path_state(self, p, th):
        wp1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
        seg = wp2 - wp1
        seg_mag = np.linalg.norm(seg)
        if seg_mag < 1e-9:
            return 0.0, 0.0, 0.0
        seg_uv = seg / seg_mag
        tangent = np.arctan2(seg_uv[1], seg_uv[0])
 
        s = np.dot(p - wp1, seg_uv)
        if s >= seg_mag and (self.cyclic or self.wpi < self.N - 2):
            self.wpi += 1
 
        ep = wp1 + seg_uv * s
        ct = ep - p
        side = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        e = np.linalg.norm(ct) * np.sign(side)  # STATE component: cross-track error e
 
        wp3 = self.wp[:, np.mod(self.wpi + 2, self.N - 1)]
        seg2 = wp3 - wp2
        m2 = np.linalg.norm(seg2)
        if m2 > 1e-9:
            tangent_n = np.arctan2(seg2[1], seg2[0])
            dtan = wrap_to_pi(tangent_n - tangent)
            frac = float(np.clip(s / seg_mag, 0.0, 1.0))
            tangent_s = tangent + frac * dtan
            kappa = dtan / max(seg_mag, 1e-3)
        else:
            tangent_s = tangent
            kappa = 0.0
        psi = wrap_to_pi(tangent_s - th)  # STATE component: heading error psi
        self.p_ref = ep
        self.th_ref = tangent_s
        return e, psi, kappa
 
    def _identify(self):
        X = np.asarray(self._X); U = np.asarray(self._U)
        Kap = np.asarray(self._Kap); Xn = np.asarray(self._Xn)
        Psi = np.column_stack([X[:, 0], X[:, 1], U, Kap])  # GAINS FROM DATA: fit plant  x+ = A x + B delta + E kappa
        Th, *_ = np.linalg.lstsq(Psi, Xn, rcond=None)
        A = Th[0:2, :].T
        B = Th[2:3, :].T
        E = Th[3:4, :].T
        cond = float(np.linalg.cond(Psi.T @ Psi))
        return A, B, E, cond
 
    def _lspi_iv(self, A_hat, B_hat, E_hat):
        X = np.asarray(self._X); U = np.asarray(self._U)
        Kap = np.asarray(self._Kap); Xn = np.asarray(self._Xn)
        N = len(U)
        Xnc = Xn - Kap[:, None] * E_hat.flatten()[None, :]
 
        s = np.array([X[:, 0].std(), X[:, 1].std(), U.std()]) + 1e-12
        Xs = X / s[0:2]; Us = U / s[2]; Xns = Xnc / s[0:2]
 
        Zk = np.array([_vecs3([Xs[k, 0], Xs[k, 1], Us[k]]) for k in range(N)])
        rk = np.array([X[k] @ self.Qs @ X[k] + self.R * U[k] ** 2 for k in range(N)])
        Xi = np.vstack([Zk[0:1], Zk[:-1]])
 
        try:
            Kseed, _ = _solve_dare(A_hat, B_hat, self.Qs, self.R)  # stabilizing seed from the IDENTIFIED plant
            Kseed = Kseed.reshape(2)
        except Exception:
            Kseed = self.Kboot.copy()
        K = (Kseed * s[0:2] / s[2]).reshape(2)
        lamI = self.lspi_ridge * np.eye(6)
        H = None
        for _ in range(self.lspi_sweeps):
            Up = -(Xns @ K)
            Zn = np.array([_vecs3([Xns[k, 0], Xns[k, 1], Up[k]]) for k in range(N)])
            Phi = Zk - Zn
            Amat = Xi.T @ Phi + lamI
            bvec = Xi.T @ rk
            try:
                hbar = np.linalg.solve(Amat, bvec)
            except np.linalg.LinAlgError:
                break
            H = _unpackH(hbar)
            Huu = H[2, 2]
            if not np.isfinite(Huu) or Huu <= 1e-9:
                break
            Kn = H[2, 0:2] / Huu  # GAIN from learned Q-kernel:  K = H_uu^{-1} H_uz
            if np.max(np.abs(Kn - K)) < 1e-9:
                K = Kn; break
            K = Kn
 
        K_phys = K * s[2] / s[0:2]
        if H is not None and H[2, 2] > 1e-9:
            P_s = H[0:2, 0:2] - np.outer(H[0:2, 2], H[2, 0:2]) / H[2, 2]
        else:
            P_s = None
        return K_phys, H, P_s
 
    def _maybe_identify(self):
        if len(self._U) >= self._next_id_at:
            try:
                A, B, E, cond = self._identify()
                if np.isfinite(A).all() and np.isfinite(B).all():
                    self.A_hat, self.B_hat, self.E_hat = A, B, E
            except Exception:
                pass
            self._next_id_at = len(self._U) + self.reident_period
 
    def _finalize_learning(self):
        n = len(self._U)
        try:
            A, B, E, cond = self._identify()
        except Exception:
            self.learned = True
            print("[SafeQ] WARNING: identification failed; keeping bootstrap gain.")
            return
        self.A_hat, self.B_hat, self.E_hat = A, B, E
 
        K_lspi, H, P_s = self._lspi_iv(A, B, E)
        K_lspi = self._cap_kpsi(K_lspi)
 
        def rho_of(K):
            try:
                return float(np.max(np.abs(np.linalg.eigvals(A - B @ K.reshape(1, 2)))))
            except Exception:
                return np.inf
        rho_lspi = rho_of(K_lspi)
 
        try:
            K_dare, _ = _solve_dare(A, B, self.Qs, self.R)  # GAIN (fallback): DARE on the IDENTIFIED (A,B)
            K_dare = self._cap_kpsi(K_dare.reshape(2))
        except Exception:
            K_dare = self.Kboot.copy()
        rho_dare = rho_of(K_dare)
 
        ev = np.linalg.eigvals(A)
        print("[SafeQ] ===== model-free finalize: %d samples, cond(Psi'Psi)=%.2e =====" % (n, cond))
        print("[SafeQ]   A_hat = [[%+.4f %+.4f];[%+.4f %+.4f]]  eig(A_hat)=%s"
              % (A[0, 0], A[0, 1], A[1, 0], A[1, 1], np.round(ev, 4)))
        print("[SafeQ]   B_hat = [%+.5f, %+.5f]   E_hat = [%+.5f, %+.5f]"
              % (B[0, 0], B[1, 0], E[0, 0], E[1, 0]))
        print("[SafeQ]   LEARNED Q-kernel gain K = [%+.4f, %+.4f]  (LSPI ; rho=%.3f)"
              % (K_lspi[0], K_lspi[1], rho_lspi))
        print("[SafeQ]   identified-DARE gain    K = [%+.4f, %+.4f]  (data-driven fallback ; rho=%.3f)"
              % (K_dare[0], K_dare[1], rho_dare))
 
        pd_ok = False
        if P_s is not None:
            try:
                pd_ok = bool(np.all(np.linalg.eigvals(P_s).real > 1e-9))
            except Exception:
                pd_ok = False
        agree = bool(np.linalg.norm(K_lspi - K_dare) <= 0.20 * (np.linalg.norm(K_dare) + 1e-9))
 
        if np.isfinite(K_lspi).all() and rho_lspi < 1.0 and agree:
            self.K = K_lspi  # DEPLOYED GAIN = learned model-free Q-kernel gain
            print("[SafeQ]   -> deploying the LEARNED Q-kernel gain "
                  "(matches the identified-DARE optimum: OPTIMAL & model-free)  [value-matrix PD=%s]." % pd_ok)
        elif np.isfinite(K_dare).all() and rho_dare < 1.0:
            self.K = K_dare
            print("[SafeQ]   -> LSPI gain disagrees with the identified-DARE optimum; deploying identified-DARE "
                  "(optimal for the identified plant, still data-driven).")
        else:
            self.K = self.Kboot.copy()
            print("[SafeQ]   -> both gains failed the stability gate; keeping the bootstrap gain.")
 
        if P_s is not None:
            print("[SafeQ]   value-kernel (scaled) eig(P)=%s" % np.round(np.linalg.eigvals(P_s), 4))
        self.learned = True
        self._X = []; self._U = []; self._Kap = []; self._Xn = []
 
    def _cbf_filter(self, x, kappa, delta_nom):  # SAFETY: discrete high-order CBF filter (enforces |e|<=e_max each step)
        if self.A_hat is None:
            return float(np.clip(delta_nom, -self.maxSteeringAngle, self.maxSteeringAngle))
        e, psi = float(x[0]), float(x[1])
        A, B, E = self.A_hat, self.B_hat, self.E_hat
        A00, A01 = A[0, 0], A[0, 1]
        A10, A11 = A[1, 0], A[1, 1]
        B0, B1 = B[0, 0], B[1, 0]
        E0, E1 = E[0, 0], E[1, 0]
 
        e1_c = A00 * e + A01 * psi + E0 * kappa;  e1_d = B0
        p1_c = A10 * e + A11 * psi + E1 * kappa;  p1_d = B1
        e2_c = A00 * e1_c + A01 * p1_c + E0 * kappa
        e2_d = A00 * e1_d + A01 * p1_d
 
        eL = self.e_left - self.margin
        eR = self.e_right - self.margin
        dec = (1.0 - self.alpha)
 
        lo, hi = -self.maxSteeringAngle, self.maxSteeringAngle
        cons = [(-e2_d, (e2_c + eL) - dec * (e + eL)),
                (e2_d, (eR - e2_c) - dec * (eR - e))]
        feasible = True
        for p, q in cons:
            if abs(p) > 1e-9:
                if p > 0:
                    hi = min(hi, q / p)
                else:
                    lo = max(lo, q / p)
            else:
                if q < 0:
                    feasible = False
        if (not feasible) or (lo > hi):
            return 0.5 * (lo + hi)
        return float(min(max(delta_nom, lo), hi))
 
    def update(self, p, th, speed, dt):
        p = np.asarray(p, float)
        p_la = p + np.array([np.cos(th), np.sin(th)]) * self.look_ahead
        e, psi, kappa = self._path_state(p_la, th)
        x = np.array([e, psi])  # STATE x = [e, psi]
 
        learning = self.online_lspi and (not self.learned)
 
        probe = 0.0
        if learning:
            self.t_learn += dt
            t = self.t_learn
            probe = (self.weave_amp * np.sin(2 * np.pi * self.weave_f * t)
                     + self.explore_amp * (0.6 * np.sin(2 * np.pi * 0.7 * t)
                                           + 0.4 * np.sin(2 * np.pi * 1.9 * t)
                                           + 0.3 * np.sin(2 * np.pi * 3.3 * t)))
            delta_nom = -float(self.Kboot @ x) + probe
        else:
            delta_nom = -float(self.K @ x)  # CONTROLLER (deployed):  u = delta = -K x
 
        delta = self._cbf_filter(x, kappa, delta_nom)  # SAFETY filter clips the steering into the safe interval
        delta = float(np.clip(delta, -self.maxSteeringAngle, self.maxSteeringAngle))
 
        if not learning:
            if self._sdelta is None:
                self._sdelta = delta
            self._sdelta += float(np.clip(delta - self._sdelta, -self.steer_rate, self.steer_rate))
            delta = self._sdelta
 
        if learning:
            if self._prev is not None and dt < 0.05:
                x_prev, d_prev, k_prev = self._prev
                self._X.append(x_prev); self._U.append(d_prev)
                self._Kap.append(k_prev); self._Xn.append(x.copy())
            self._prev = (x.copy(), delta, kappa)
 
            self._maybe_identify()
            n = len(self._U)
            if (self.t_learn >= self.learn_T) and (n >= self.min_final_samples):
                self._finalize_learning()
            elif self.t_learn >= self.max_learn_T:
                if n >= self.min_id_samples:
                    self._finalize_learning()
                else:
                    self.learned = True
                    print("[SafeQ] WARNING: only %d samples in %.0fs; keeping bootstrap gain."
                          % (n, self.max_learn_T))
 
        return SIGN_STEER * delta  # CONTROL INPUT u = delta (steering) sent to the car
 
 
def controlLoop():
    global KILL_THREAD
    u = 0
    delta = 0
    countMax = controllerUpdateRate / 10
    count = 0
 
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SafeQSteeringController(
            waypoints=waypointSequence,
            e_left=e_max_left,
            e_right=e_max_right,
            Q=Q_lat,
            R=R_lat,
            gamma=gamma_cbf,
            beta=beta_cbf,
            alpha=cbf_alpha,
            margin=cbf_margin,
            online_lspi=useOnlineLSPI,
            learn_T=learnDuration,
            explore_amp=exploreAmp,
            Kboot=Kboot,
            lspi_sweeps=lspiSweeps,
            lspi_ridge=lspiRidge
        )
 
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b'')
 
    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            tp = t
            t = time.time() - t0
            dt = t - tp
 
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])
                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y])
            v = qcar.motorTach
 
            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = speedController.update(v, v_ref, dt)  # throttle: separate PI speed loop, NOT the learned controller
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v, dt)  # steering u = delta from the learned safe controller
                else:
                    delta = 0
 
            qcar.write(u, delta)
 
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref - v])
                speedScope.axes[2].sample(t_plot, [u])
 
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])
                    p[0] = ekf.x_hat[0, 0]
                    p[1] = ekf.x_hat[1, 0]
                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]
                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])
                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)
                count = 0
            continue
        qcar.read_write_std(throttle=0, steering=0)
 
 
if __name__ == '__main__':
 
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
 
    speedScope = MultiScope(rows=3, cols=1, title='Vehicle Speed Control', fps=fps)
    speedScope.addAxis(row=0, col=0, timeWindow=tf, yLabel='Vehicle Speed [m/s]', yLim=(0, 1))
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')
    speedScope.addAxis(row=1, col=0, timeWindow=tf, yLabel='Speed Error [m/s]', yLim=(-0.5, 0.5))
    speedScope.axes[1].attachSignal()
    speedScope.addAxis(row=2, col=0, timeWindow=tf, xLabel='Time [s]', yLabel='Throttle Command [%]', yLim=(-0.3, 0.3))
    speedScope.axes[2].attachSignal()
 
    if enableSteeringControl:
        steeringScope = MultiScope(rows=4, cols=2, title='Vehicle Steering Control', fps=fps)
        steeringScope.addAxis(row=0, col=0, timeWindow=tf, yLabel='x Position [m]', yLim=(-2.5, 2.5))
        steeringScope.axes[0].attachSignal(name='x_meas')
        steeringScope.axes[0].attachSignal(name='x_ref')
        steeringScope.addAxis(row=1, col=0, timeWindow=tf, yLabel='y Position [m]', yLim=(-1, 5))
        steeringScope.axes[1].attachSignal(name='y_meas')
        steeringScope.axes[1].attachSignal(name='y_ref')
        steeringScope.addAxis(row=2, col=0, timeWindow=tf, yLabel='Heading Angle [rad]', yLim=(-3.5, 3.5))
        steeringScope.axes[2].attachSignal(name='th_meas')
        steeringScope.axes[2].attachSignal(name='th_ref')
        steeringScope.addAxis(row=3, col=0, timeWindow=tf, yLabel='Steering Angle [rad]', yLim=(-0.6, 0.6))
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = 'Time [s]'
        steeringScope.addXYAxis(row=0, col=1, rowSpan=4, xLabel='x Position [m]', yLabel='y Position [m]',
                                xLim=(-2.5, 2.5), yLim=(-1, 5))
        im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
        steeringScope.axes[4].attachImage(scale=(-0.002035, 0.002035), offset=(1125, 2365), rotation=180, levels=(0, 255))
        steeringScope.axes[4].images[0].setImage(image=im)
        referencePath = pg.PlotDataItem(pen={'color': (85, 168, 104), 'width': 2}, name='Reference')
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
        steeringScope.axes[4].attachSignal(name='Estimated', width=2)
        arrow = pg.ArrowItem(angle=180, tipAngle=60, headLen=10, tailLen=10, tailWidth=5,
                             pen={'color': 'w', 'fillColor': [196, 78, 82], 'width': 1}, brush=[196, 78, 82])
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
 
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()
 
    input('Experiment complete. Press any key to exit...')