"""Two-factor (stochastic demand) equilibrium: closed forms and exact moments of weekly means.

Scaled coordinates: ytil = b (Y - Ybar) (supply), alpha = a - abar (demand), with unit supply noise:
    d ytil  = (-kappa ytil + beta alpha) dt + dW^0,
    d alpha = -kappa_a alpha dt + r dW^a,          lam - lbar = alpha - ytil,
where beta = b C / eta, C = 1/(rho + kappa + kappa_a), r = sigma_a / (b sigma0).
With s = b/(chi + b) in (0, 1] (chi >= 0):  b/eta = s kappa (kappa + rho), so
    beta = s kappa (kappa + rho) / (rho + kappa + kappa_a).

The data are weekly MEANS of daily values, so the model moments are those of
    m_k = (1/5) sum_{j=0..4} xi(k D + j delta),   D = 1/52 year, delta = D/5,
computed exactly from the autocovariance Gamma(tau) = expm(M tau) Sigma (tau >= 0).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

D_WEEK = 1.0 / 52.0
N_DAYS = 5
L_LAM = np.array([-1.0, 1.0])   # lam = alpha - ytil in (ytil, alpha) coordinates
E_Y = np.array([1.0, 0.0])
T_LY = np.array([[-1.0, 1.0], [1.0, 0.0]])  # (lam, ytil) = T (ytil, alpha)


def closed_forms(eta, rho, chi, b, kappa_a, abar):
    kind = (-rho + np.sqrt(rho ** 2 + 4 * chi / eta)) / 2
    kap = (-rho + np.sqrt(rho ** 2 + 4 * (chi + b) / eta)) / 2
    C = 1 / (rho + kap + kappa_a)
    return dict(kappa=kap, kappa_ind=kind, C=C, g=C / eta, Ybar=abar / (chi + b), lbar=abar * chi / (chi + b),
                B=kappa_a * abar * C / (rho + kap))


def beta_of(kap, kap_a, s, rho=0.05):
    return s * kap * (kap + rho) / (rho + kap + kap_a)


def system(kap, kap_a, beta, r):
    M = np.array([[-kap, beta], [0.0, -kap_a]])
    Sig = solve_continuous_lyapunov(M, -np.diag([1.0, r * r]))
    return M, Sig


def expm_ut(M, tau):
    """Exact exponential of the upper-triangular drift [[-k, beta], [0, -ka]] times tau."""
    k, beta, ka = -M[0, 0], M[0, 1], -M[1, 1]
    e1, e2 = np.exp(-k * tau), np.exp(-ka * tau)
    off = beta * (e2 - e1) / (k - ka) if abs(k - ka) > 1e-10 else beta * tau * e1
    return np.array([[e1, off], [0.0, e2]])


def gamma(M, Sig, tau):
    return expm_ut(M, tau) @ Sig if tau >= 0 else (expm_ut(M, -tau) @ Sig).T


def weekly_mean_cov(M, Sig, lag):
    """Cov(m_{k+lag}, m_k) for weekly means of N_DAYS daily point values."""
    delta = D_WEEK / N_DAYS
    acc = np.zeros((2, 2))
    for i in range(N_DAYS):
        for j in range(N_DAYS):
            acc += gamma(M, Sig, lag * D_WEEK + (i - j) * delta)
    return acc / N_DAYS ** 2


def point_cov(M, Sig, lag):
    return gamma(M, Sig, lag * D_WEEK)


def moments(kap, kap_a, beta, r, averaged=True, horizons=(1,)):
    M, Sig = system(kap, kap_a, beta, r)
    cov = weekly_mean_cov if averaged else point_cov
    C0, C1 = cov(M, Sig, 0), cov(M, Sig, 1)
    l, e = L_LAM, E_Y
    phi_l = (l @ C1 @ l) / (l @ C0 @ l)
    phi_y = C1[0, 0] / C0[0, 0]
    corr_lev = (l @ C0 @ e) / np.sqrt((l @ C0 @ l) * C0[0, 0])
    out = dict(k_lam=-np.log(phi_l) / D_WEEK, k_Y=-np.log(phi_y) / D_WEEK, corr_lev=corr_lev)
    for h in horizons:
        Ch = cov(M, Sig, h)
        Sh = 2 * C0 - Ch - Ch.T
        out[f"corr_chg_{h}"] = (l @ Sh @ e) / np.sqrt((l @ Sh @ l) * Sh[0, 0])
    Z0 = T_LY @ C0 @ T_LY.T
    Z1 = T_LY @ C1 @ T_LY.T
    Phi = Z1 @ np.linalg.inv(Z0)                 # population VAR(1) of (lam, ytil)
    out.update(Phi=Phi)
    return out
