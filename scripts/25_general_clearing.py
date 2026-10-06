#!/usr/bin/env python3
"""25 - General clearing: numerical verification of the structural theorem.

(a) Automatic curvature. Against a given premium path, the feedback v = (2 q0 X + p)/eta with
    p_t = int_t^inf e^{-rho_eff (s-t)} lam_s ds satisfies the first-order condition
    eta v_t = P_t := int_t^inf e^{-rho (s-t)} (lam_s - chi X_s) ds  (symbolic check, sympy).
(b) Nonlinear clearing Lambda(Y) = a - b Y - c tanh((Y - y0)/l), strictly decreasing and Lipschitz.
    - MFG fixed point: iterate  u -> drift b(Y) = (u + 2 q0 Y)/eta -> solve the linear equation
      (sigma0^2/2) u'' + b u' - rho_eff u + Lambda = 0  (from three different initial guesses);
    - planner: policy iteration on  rho S = S'^2/(2 eta) + (sigma0^2/2) S'' + G - chi Y^2/2,  G' = Lambda;
    - compare u_MFG with u_planner = S' - 2 q0 Y; check u' <= 0 and -b'(Y) >= kappa_ind;
    - grid refinement.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# (a) is checked in 25a_curvature_identity.py

# ------------------------------------------------------------------ (b) nonlinear clearing
par = dict(eta=1.0, rho=0.10, chi=0.04, a=0.05, b=0.05, s0=0.15, c=0.03, l=0.2, y0=0.5)
e, r, ch, a, b, s0, c, l, y0 = (par[k] for k in ("eta", "rho", "chi", "a", "b", "s0", "c", "l", "y0"))
kin = (-r + np.sqrt(r ** 2 + 4 * ch / e)) / 2
kap = (-r + np.sqrt(r ** 2 + 4 * (ch + b) / e)) / 2
q0n = -e * kin / 2
reff = r + kin
A_aff = e * (kin - kap)                                  # asymptotic slope of u (affine tails, slope b)
Lam = lambda Y: a - b * Y - c * np.tanh((Y - y0) / l)
Gf = lambda Y: a * Y - b * Y ** 2 / 2 - c * l * np.log(np.cosh((Y - y0) / l))


def lin_solve(Y, drift, rate, rhs, bc2):
    """Solve (s0^2/2) f'' + drift f' - rate f = -rhs with f'' = bc2 at both ends (central differences, banded)."""
    n, h = len(Y), Y[1] - Y[0]
    D = s0 ** 2 / 2
    lo = D / h ** 2 - drift / (2 * h)
    di = -2 * D / h ** 2 - rate * np.ones(n)
    up = D / h ** 2 + drift / (2 * h)
    sub, main, sup = lo[1:].copy(), di.copy(), up[:-1].copy()
    sub2, sup2 = np.zeros(n - 2), np.zeros(n - 2)
    # boundary rows: one-sided second difference = bc2
    main[0], sup[0], sup2[0] = 1 / h ** 2, -2 / h ** 2, 1 / h ** 2
    main[-1], sub[-1], sub2[-1] = 1 / h ** 2, -2 / h ** 2, 1 / h ** 2
    M = sp.diags([sub2, sub, main, sup, sup2], [-2, -1, 0, 1, 2], format="csc")
    bvec = -rhs.copy()
    bvec[0] = bvec[-1] = bc2
    return spla.spsolve(M, bvec)


def mfg_fixed_point(Y, u_init, iters=400, damp=0.5):
    u = u_init.copy()
    for k in range(iters):
        drift = (u + 2 * q0n * Y) / e
        u_new = lin_solve(Y, drift, reff, Lam(Y), 0.0)
        if np.max(np.abs(u_new - u)) < 1e-13:
            return u_new, k
        u = (1 - damp) * u + damp * u_new
    return u, iters


def planner(Y, iters=200):
    S = (q0n + A_aff / 2) * Y ** 2
    for k in range(iters):
        nu = np.gradient(S, Y) / e
        S_new = lin_solve(Y, nu, r, Gf(Y) - ch * Y ** 2 / 2 - e * nu ** 2 / 2, 2 * q0n + A_aff)
        if np.max(np.abs(S_new - S)) < 1e-12:
            return S_new, k
        S = S_new
    return S, iters


out = {}
for n in (2001, 4001):
    Y = np.linspace(-4.0, 5.0, n)
    inits = [A_aff * Y + a / (r + kap), np.zeros(n), 2 * A_aff * Y + 0.3]
    us = [mfg_fixed_point(Y, u0)[0] for u0 in inits]
    S, _ = planner(Y)
    up = np.gradient(S, Y) - 2 * q0n * Y
    out[n] = (Y, us, up)
    core = (Y > -1.0) & (Y < 2.0)                       # the region carrying the stationary law
    du = np.gradient(us[0], Y)
    rate = -(du + 2 * q0n) / e
    print(f"(b) grid n = {n}: max |u_MFG - u_planner| on core {np.max(np.abs(us[0] - up)[core]):.2e}; "
          f"spread across initial guesses {max(np.max(np.abs(us[0] - x)) for x in us[1:]):.1e}; "
          f"max u' {du[core].max():.4f} (<= 0); min rate {rate[core].min():.4f} vs kappa_ind {kin:.4f}, kappa(b) {kap:.4f}")

# grid refinement of the shadow price at common points
Y1, us1, _ = out[2001]
Y2, us2, _ = out[4001]
common = np.interp(Y1, Y2, us2[0])
core = (Y1 > -1.0) & (Y1 < 2.0)
print(f"    grid refinement (2001 vs 4001), core: {np.max(np.abs(us1[0] - common)[core]):.2e}")
Yc = Y1[core]
lam_core = Lam(Yc)
print(f"    the curvature of the clearing curve is material: -Lambda' ranges over [{(b + c / l / np.cosh((Yc - y0) / l) ** 2).min():.3f}, "
      f"{(b + c / l / np.cosh((Yc - y0) / l) ** 2).max():.3f}] on the core (affine: {b})")
