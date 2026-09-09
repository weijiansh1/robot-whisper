"""Constrained Cartesian recovery MPC using identified observed motion dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize

PROTOCOL = "moe_control.identified_recovery_mpc.v1"
ARMS = ("native", "hold", "withdraw", "mpc", "mpc_disturbance")
SETTINGS = dict(horizon=6, position_weight=1., velocity_weight=.25, input_weight=.15,
    input_radius_cm=1., disturbance_filter=.5, disturbance_radius_cm=.5,
    solver="scipy.optimize.SLSQP", solver_ftol=1e-9, solver_maxiter=120,
    projected_gradient_tolerance=2e-4, target="same frozen 8-step lift and 8-step retreat",
    state_units="position cm; displacement cm/environment-step; input Cartesian delta cm")


def matrices(a, b):
    av, bu = np.diag(a), np.diag(b)
    A = np.block([[np.eye(3), av], [np.zeros((3, 3)), av]])
    B = np.vstack((bu, bu))
    D = np.vstack((np.eye(3), np.eye(3)))
    return A, B, D


def certificate(a, b):
    A, B, _ = matrices(a, b)
    Q = np.diag([SETTINGS["position_weight"]]*3+[SETTINGS["velocity_weight"]]*3)
    R = np.eye(3)*SETTINGS["input_weight"]
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
    closed = A-B@K
    residual = closed.T@P@closed-P+Q+K.T@R@K
    controllability = np.hstack([np.linalg.matrix_power(A, k)@B for k in range(6)])
    C = np.hstack((np.eye(3), np.zeros((3, 3))))
    observability = np.vstack([C@np.linalg.matrix_power(A, k) for k in range(6)])
    return dict(A=A.tolist(), B=B.tolist(), Q=Q.tolist(), R=R.tolist(), P=P.tolist(), K=K.tolist(),
        closed_loop_spectral_radius=float(np.abs(np.linalg.eigvals(closed)).max()),
        riccati_identity_max_residual=float(np.abs(residual).max()),
        minimum_P_eigenvalue=float(np.linalg.eigvalsh(P).min()),
        controllability_rank=int(np.linalg.matrix_rank(controllability)),
        observability_rank=int(np.linalg.matrix_rank(observability)),
        controllability_singular_values=np.linalg.svd(controllability, compute_uv=False).tolist(),
        observability_singular_values=np.linalg.svd(observability, compute_uv=False).tolist(),
        scope="identified six-dimensional Cartesian model, constant target, unconstrained LQR, zero disturbance; no task or contact stability claim")


def project_ball(value, radius=1.):
    value = np.asarray(value, dtype=np.float64)
    return value/np.maximum(1., np.linalg.norm(value, axis=-1, keepdims=True)/radius)


class RecoveryMPC:
    def __init__(self, model):
        if isinstance(model, (str, Path)):
            model = json.loads(Path(model).read_text())
        self.a, self.b = np.asarray(model["a"]), np.asarray(model["b"])
        self.A, self.B, self.D = matrices(self.a, self.b)
        self.proof = certificate(self.a, self.b)
        self.P = np.asarray(self.proof["P"])
        self.Q, self.R = np.asarray(self.proof["Q"]), np.asarray(self.proof["R"])
        self.horizon = SETTINGS["horizon"]
        self.S = np.zeros((6*self.horizon, 3*self.horizon))
        self.T = np.vstack([np.linalg.matrix_power(self.A, k+1) for k in range(self.horizon)])
        for k in range(self.horizon):
            for j in range(k+1):
                self.S[6*k:6*k+6, 3*j:3*j+3] = np.linalg.matrix_power(self.A, k-j)@self.B
        self.W = block_diag(*([self.Q]*(self.horizon-1)+[self.P]))
        self.Rbar = np.kron(np.eye(self.horizon), self.R)
        self.H = 2*(self.S.T@self.W@self.S+self.Rbar)
        self.lipschitz = float(np.linalg.eigvalsh(self.H).max())
        self.warm = np.zeros((self.horizon, 3))

    def solve(self, state_cm, disturbance_cm):
        state, d = np.asarray(state_cm), np.asarray(disturbance_cm)
        if state.shape != (6,) or d.shape != (3,) or not np.isfinite(state).all() or not np.isfinite(d).all():
            raise ValueError("MPC requires finite position, velocity and disturbance observations")
        bias, offset = [], np.zeros(6)
        for _ in range(self.horizon):
            offset = self.A@offset+self.D@d
            bias.append(offset.copy())
        base = self.T@state+np.concatenate(bias)
        f = 2*self.S.T@self.W@base
        constant = float(base@self.W@base)

        def value(u):
            return float(.5*u@self.H@u+f@u+constant)

        def gradient(u):
            return self.H@u+f

        def feasible(u):
            return 1.-np.square(u.reshape(-1, 3)).sum(axis=1)

        def constraint_jacobian(u):
            result = np.zeros((self.horizon, 3*self.horizon))
            for i, row in enumerate(u.reshape(-1, 3)):
                result[i, 3*i:3*i+3] = -2*row
            return result

        result = minimize(value, self.warm.ravel(), jac=gradient, method="SLSQP",
            constraints=[dict(type="ineq", fun=feasible, jac=constraint_jacobian)],
            options=dict(ftol=SETTINGS["solver_ftol"], maxiter=SETTINGS["solver_maxiter"]))
        finite = bool(np.isfinite(result.x).all())
        plan = project_ball(result.x.reshape(-1, 3)) if finite else np.zeros_like(self.warm)
        pg = plan-project_ball(plan-gradient(plan.ravel()).reshape(-1, 3)/self.lipschitz)
        residual = float(np.abs(pg).max())
        passed = bool(result.success and finite and residual<=SETTINGS["projected_gradient_tolerance"])
        if not passed:
            plan[:] = 0.
            plan[0] = project_ball(-state[:3])
        self.warm = np.vstack((plan[1:], np.zeros(3)))
        predicted = (base+self.S@plan.ravel()).reshape(self.horizon, 6)
        return dict(plan_cm=plan, predicted_states_cm=predicted, objective=value(plan.ravel()),
            projected_gradient_residual=residual, solver_success=bool(result.success),
            optimizer_accepted=passed, solver_status=int(result.status), iterations=int(result.nit),
            fallback=not passed)


def update_disturbance(previous, observed_velocity_cm, velocity_cm, input_cm, a, b):
    innovation = observed_velocity_cm-np.asarray(a)*velocity_cm-np.asarray(b)*input_cm
    estimate = (1-SETTINGS["disturbance_filter"])*previous+SETTINGS["disturbance_filter"]*innovation
    return project_ball(estimate, SETTINGS["disturbance_radius_cm"])
