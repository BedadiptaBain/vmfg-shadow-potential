"""Independent numerical check of the equilibrium (does not use Sections 3-5).

Route: (i) fix a guess for the aggregate law dY = -k_g (Y - y_g) dt + sigma0 dW0;
(ii) solve ONE seller's discounted LQ problem by an algebraic Riccati equation;
(iii) average the resulting feedback over the population to get the implied
aggregate drift; (iv) solve the fixed point (k_g, y_g) = implied (k, y).
This is the textbook MFG fixed point that the paper's shadow-potential
reduction avoids; agreement with the closed forms is therefore a genuinely
independent confirmation.

Symbols: xi = (X, Y, 1) augmented state; K = (K0, K1, K2) linear feedback
v = K0 X + K1 Y + K2; Q = cost matrix with cost = -(reward without the
control term) = (chi/2)X^2 + b X Y - a X; R = eta/2; P = Riccati solution
(cost-to-go xi' P xi), so that the value is V = -xi' P xi (+ noise constant).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import brentq

from . import model as M


def _cost_matrix(p: M.Params) -> np.ndarray:
    return np.array(
        [[p.chi / 2.0, p.b / 2.0, -p.a / 2.0],
         [p.b / 2.0, 0.0, 0.0],
         [-p.a / 2.0, 0.0, 0.0]]
    )


def _care_hamiltonian(Ad: np.ndarray, Bc: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Stabilising solution of Ad'P + P Ad - P Bc R^-1 Bc' P + Q = 0 via the
    stable invariant subspace of the Hamiltonian matrix (fallback solver)."""
    n = Ad.shape[0]
    H = np.block([[Ad, -Bc @ np.linalg.solve(R, Bc.T)], [-Q, -Ad.T]])
    w, V = np.linalg.eig(H)
    idx = np.where(w.real < 0)[0]
    if len(idx) != n:
        raise np.linalg.LinAlgError("Hamiltonian has no n-dimensional stable subspace")
    U = V[:, idx]
    P = np.real(U[n:, :] @ np.linalg.inv(U[:n, :]))
    return 0.5 * (P + P.T)


def best_response(p: M.Params, k_g: float, y_g: float) -> tuple[float, float, float]:
    """One seller's optimal feedback against dY = -k_g (Y - y_g) dt + sigma0 dW0.

    Returns (q0, A, B) in the paper's parametrisation v = (2 q0 X + A Y + B)/eta.
    Discounting is removed by the shift Ad = M - (rho/2) I; the noise does not
    affect the feedback (certainty equivalence).
    """
    Mdyn = np.array([[0.0, 0.0, 0.0], [0.0, -k_g, k_g * y_g], [0.0, 0.0, 0.0]])
    Ad = Mdyn - 0.5 * p.rho * np.eye(3)
    Bc = np.array([[1.0], [0.0], [0.0]])
    Q = _cost_matrix(p)
    R = np.array([[p.eta / 2.0]])
    try:
        P = solve_continuous_are(Ad, Bc, Q, R)
    except Exception:  # indefinite Q: fall back to the Hamiltonian route
        P = _care_hamiltonian(Ad, Bc, Q, R)
    # v = -R^{-1} Bc' P xi = -(2/eta)(P00 X + P01 Y + P02)
    return -P[0, 0], -2.0 * P[0, 1], -2.0 * P[0, 2]


def implied_aggregate(p: M.Params, k_g: float, y_g: float) -> tuple[float, float]:
    """Aggregate (kappa, Ybar) implied by averaging the best response."""
    q, a_, b_ = best_response(p, k_g, y_g)
    k_new = -(2.0 * q + a_) / p.eta
    y_new = b_ / (p.eta * k_new)
    return k_new, y_new


def equilibrium_fixed_point(p: M.Params) -> dict:
    """Solve the MFG fixed point numerically.

    The implied speed depends on the guess k_g only (not on y_g), so kappa is
    found by a bracketing root-finder on k_g -> implied kappa - k_g over
    (0, large], which needs no starting value.  Given kappa, the implied mean
    is an affine function of the guess y_g, so two probes (y_g = 0, 1) and one
    linear solve give Ybar exactly.
    """
    def gap(k):
        return implied_aggregate(p, k, 0.0)[0] - k

    k_hi = 10.0 * (M.kappa_ind(p) + np.sqrt((p.chi + p.b) / p.eta) + 1.0)
    k_star = brentq(gap, 1e-12, k_hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    f0 = implied_aggregate(p, k_star, 0.0)[1]
    f1 = implied_aggregate(p, k_star, 1.0)[1]
    y_star = f0 / (1.0 - (f1 - f0))          # solves y = f0 + (f1 - f0) y
    q, a_, b_ = best_response(p, k_star, y_star)
    return dict(kappa=k_star, Ybar=y_star, q0=q, A=a_, B=b_)


def planner_aggregate(p: M.Params) -> tuple[float, float]:
    """Aggregate drift from the planner problem
        max E int e^{-rho t}[a Y - ((b+chi)/2) Y^2 - (eta/2) v^2] dt, dY = v dt + sigma0 dW0,
    solved by its own Riccati equation on (Y, 1)."""
    Ad = -0.5 * p.rho * np.eye(2)
    Bc = np.array([[1.0], [0.0]])
    Q = np.array([[(p.b + p.chi) / 2.0, -p.a / 2.0], [-p.a / 2.0, 0.0]])
    R = np.array([[p.eta / 2.0]])
    try:
        P = solve_continuous_are(Ad, Bc, Q, R)
    except Exception:
        P = _care_hamiltonian(Ad, Bc, Q, R)
    slope = -(2.0 / p.eta) * P[0, 0]   # drift = slope * Y + intercept
    intercept = -(2.0 / p.eta) * P[0, 1]
    k = -slope
    return k, intercept / k


def policy_value(p: M.Params, K, x0: float, y0: float,
                 k_agg: float | None = None, y_agg: float | None = None) -> float:
    """Exact discounted payoff J(x0, y0; v) of the linear feedback
    v = K0 X + K1 Y + K2 when the aggregate follows its equilibrium OU law.

    With xi = (X, Y, 1): d xi = Mc xi dt + noise, noise covariance D per unit time,
    reward xi' W xi with W = -Q - (eta/2) K K'.  Then J = tr(W P) where
    P = int_0^inf e^{-rho t} E[xi_t xi_t'] dt solves
        (Mc - rho/2) P + P (Mc - rho/2)' + xi0 xi0' + D/rho = 0.
    """
    k = M.kappa(p) if k_agg is None else k_agg
    yb = M.Ybar(p) if y_agg is None else y_agg
    K = np.asarray(K, dtype=float)
    Mc = np.array([[K[0], K[1], K[2]], [0.0, -k, k * yb], [0.0, 0.0, 0.0]])
    s2, s02 = p.sigma ** 2, p.sigma0 ** 2
    D = np.array([[s2 + s02, s02, 0.0], [s02, s02, 0.0], [0.0, 0.0, 0.0]])
    xi0 = np.array([x0, y0, 1.0])
    Ad = Mc - 0.5 * p.rho * np.eye(3)
    if np.max(np.linalg.eigvals(Ad).real) >= 0:
        return -np.inf  # payoff diverges to -infinity for such feedbacks
    P = solve_continuous_lyapunov(Ad, -(np.outer(xi0, xi0) + D / p.rho))
    W = -_cost_matrix(p) - 0.5 * p.eta * np.outer(K, K)
    return float(np.sum(W * P))
