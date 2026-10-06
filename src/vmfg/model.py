"""Closed-form solution of the linear-quadratic variance-supply mean-field game.

Paper: B. Bain, "Equilibrium premium dynamics in a variance-supply mean-field
game: a shadow-potential reduction" (SIFIN submission draft, Sept 2026).
Equation numbers in square brackets refer to that draft.

Every symbol used in this module
--------------------------------
Primitives (all real scalars)
  a       intercept of the clearing curve  Lambda(Y) = a - b*Y        [Ass. 2.1]  a > 0
  b       slope of the clearing curve: premium lost per unit of aggregate
          deployment (a price-impact coefficient)          [Ass. 2.1]  b > 0
  chi     risk-penalty coefficient in the flow payoff -(chi/2) X^2   [Ass. 2.2]  chi > 0
  eta     adjustment-cost coefficient in -(eta/2) v^2                [Ass. 2.2]  eta > 0
  rho     discount rate                                             [Ass. 2.2]  rho > 0
  sigma   idiosyncratic volatility of deployed capital X            [eq. (2)]   sigma >= 0
  sigma0  common-noise volatility                                   [eq. (2)]   sigma0 > 0

State and control
  t       time
  X_t     one seller's deployed capital
  v_t     that seller's deployment rate (the control)
  Y_t     aggregate state, the conditional mean E[X_t | F^0_t], where F^0 is the
          filtration of the common Brownian motion W^0
  lam_t   premium, lam_t = Lambda(Y_t)

Derived objects
  q0          curvature of the value function in x                 [eq. (7)]
  kappa_ind   individual mean-reversion speed, kappa_ind = -2 q0/eta [eq. (15)]
  kappa       aggregate mean-reversion speed, kappa = -(A + 2 q0)/eta [eq. (15)]
  rho_eff     rho - 2 q0/eta = rho + kappa_ind                      [eq. (8)]
  A, B, C     shadow potential S(Y) = (A/2) Y^2 + B Y + C and shadow price
              u(Y) = S'(Y) = A Y + B                                [eqs. (13)-(14)]
  D, E, F     x-independent part of V: w(Y) = (D/2) Y^2 + E Y + F  [eq. (9)]
  V(x, Y)     value function q0 x^2 + x u(Y) + w(Y)                 [eq. (5)]
  Ybar        long-run mean of Y;  lam_bar = Lambda(Ybar)           [eq. (19)]
  s_star      stationary standard deviation of Y, sigma0/sqrt(2 kappa) [Ass. 8.1]
  Sigma_star  stationary conditional (cross-sectional) variance of X [eq. (20)]
  At, Bt, Ct  planner ("Lucas-Prescott") potential
              St(Y) = (At/2) Y^2 + Bt Y + Ct; it equals S(Y) + q0 Y^2 + const.
  c           curvature of the Section 8 clearing curve
              Lambda_c(Y) = a - b Y - (c/2) Y^2 on the operating region I_R
  R           half-width of the operating region I_R = [Ybar - R, Ybar + R]
  p1, r1, s1  coefficients of the first-order correction
              u1(Y) = p1 Y^2 + r1 Y + s1                            [eq. (22)]
              (the paper writes p, r, s; renamed because s is also the time
              variable in (3) and in (19)-(20), and r reads as an interest rate)
  theta       b (rho + kappa) / ((chi + b)(rho + 3 kappa)), lies in (0, 1);
              it measures the drift channel of premium skewness relative to
              the clearing-curvature channel (see skew_lambda_first_order)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Primitive parameters of the model (see module docstring)."""

    a: float
    b: float
    chi: float
    eta: float
    rho: float
    sigma: float = 0.20
    sigma0: float = 0.15

    def __post_init__(self) -> None:
        for name in ("a", "b", "chi", "eta", "rho", "sigma0"):
            val = getattr(self, name)
            if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                raise ValueError(f"{name} must be a finite positive number, got {val!r}")
        if not (math.isfinite(self.sigma) and self.sigma >= 0):
            raise ValueError(f"sigma must be finite and >= 0, got {self.sigma!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# The seven parameter rows of the paper's Table 1 (sigma, sigma0 are not used there).
TABLE1_ROWS = [
    Params(a=0.05, b=0.05, chi=0.0400, eta=1.0, rho=0.10),
    Params(a=0.05, b=0.05, chi=0.0400, eta=10.0, rho=0.10),
    Params(a=0.05, b=0.05, chi=0.0400, eta=0.5, rho=0.10),
    Params(a=0.05, b=0.05, chi=0.2000, eta=1.0, rho=0.10),
    Params(a=0.05, b=0.05, chi=0.0080, eta=1.0, rho=0.10),
    Params(a=0.10, b=0.02, chi=0.0900, eta=2.0, rho=0.05),
    Params(a=0.08, b=0.10, chi=0.1875, eta=0.5, rho=0.08),
]

# Values printed in the paper's Table 1: (kappa_ind, kappa, Ybar, lam_bar)
TABLE1_PRINTED = [
    (0.1562, 0.2541, 0.5556, 0.022222),
    (0.0306, 0.0572, 0.5556, 0.022222),
    (0.2372, 0.3772, 0.5556, 0.022222),
    (0.4000, 0.4525, 0.2000, 0.040000),
    (0.0525, 0.1960, 0.8621, 0.006897),
    (0.1886, 0.2108, 0.9091, 0.081818),
    (0.5737, 0.7193, 0.2783, 0.052174),
]


# --------------------------------------------------------------------------
# speeds, curvature, shadow price
# --------------------------------------------------------------------------
def positive_root(rho: float, K: float) -> float:
    """Positive root k of k (k + rho) = K, for K > 0.

    Written as k = 2K / (rho + sqrt(rho^2 + 4K)), which equals
    (-rho + sqrt(rho^2 + 4K))/2 but has no subtractive cancellation.
    """
    if not K > 0:
        raise ValueError("K must be positive")
    return 2.0 * K / (rho + math.sqrt(rho * rho + 4.0 * K))


def kappa_ind(p: Params) -> float:
    """eta * k (k + rho) = chi  [eq. (16), first identity]."""
    return positive_root(p.rho, p.chi / p.eta)


def kappa(p: Params) -> float:
    """eta * k (k + rho) = chi + b  [eq. (16), second identity]."""
    return positive_root(p.rho, (p.chi + p.b) / p.eta)


def q0(p: Params) -> float:
    """Negative root of rho q = -chi/2 + 2 q^2/eta  [eq. (7)]."""
    return -0.5 * p.eta * kappa_ind(p)


def q0_literal(p: Params) -> float:
    """Eq. (7) exactly as printed: (eta/4)(rho - sqrt(rho^2 + 4 chi/eta))."""
    return 0.25 * p.eta * (p.rho - math.sqrt(p.rho ** 2 + 4.0 * p.chi / p.eta))


def rho_eff(p: Params) -> float:
    """rho - 2 q0/eta = rho + kappa_ind  [eq. (8), Prop. 5.1]."""
    return p.rho + kappa_ind(p)


def A(p: Params) -> float:
    """Selected (negative) root A_- = eta (kappa_ind - kappa)  [Prop. 5.1]."""
    return p.eta * (kappa_ind(p) - kappa(p))


def A_roots(p: Params) -> tuple[float, float]:
    """Both roots (A_minus, A_plus) of A^2 + (4 q0 - rho eta) A - b eta = 0 [eq. (14)].
    The product of the roots is -b eta, which gives A_plus without cancellation."""
    am = A(p)
    return am, -p.b * p.eta / am


def B(p: Params) -> float:
    """a / (rho + kappa)  [eq. (19)]."""
    return p.a / (p.rho + kappa(p))


def C(p: Params) -> float:
    """(B^2/(2 eta) + sigma0^2 A / 2) / rho  [eq. (14)]."""
    return (B(p) ** 2 / (2.0 * p.eta) + 0.5 * p.sigma0 ** 2 * A(p)) / p.rho


def Ybar(p: Params) -> float:
    """a / (b + chi)  [eq. (19)]."""
    return p.a / (p.b + p.chi)


def lam_bar(p: Params) -> float:
    """a chi / (chi + b)  [eq. (19)] -- the static competitive-equilibrium price:
    price-taking sellers with private cost (chi/2) x^2 supply x = lam/chi, and
    lam = a - b x then gives lam = a chi/(chi + b)."""
    return p.a * p.chi / (p.chi + p.b)


def Lambda(p: Params, y):
    """Affine clearing curve a - b y  [Ass. 2.1]."""
    return p.a - p.b * np.asarray(y, dtype=float)


def var_Y(p: Params) -> float:
    return p.sigma0 ** 2 / (2.0 * kappa(p))


def s_star(p: Params) -> float:
    return math.sqrt(var_Y(p))


def sd_lambda(p: Params) -> float:
    """Stationary standard deviation of the premium, b * s_star."""
    return p.b * s_star(p)


def prob_premium_nonpositive(p: Params) -> float:
    """P(lam <= 0) under the Gaussian stationary law of Theorem 5.3."""
    return 0.5 * math.erfc((lam_bar(p) / sd_lambda(p)) / math.sqrt(2.0))


def Sigma_star(p: Params) -> float:
    """sigma^2 / (2 kappa_ind)  [eq. (20)]."""
    return p.sigma ** 2 / (2.0 * kappa_ind(p))


def w_coeffs(p: Params) -> tuple[float, float, float]:
    """(D, E, F) with w(Y) = (D/2) Y^2 + E Y + F solving eq. (9) along (18)."""
    k, yb, a_, b_, eta = kappa(p), Ybar(p), A(p), B(p), p.eta
    D = a_ ** 2 / (eta * (p.rho + 2.0 * k))
    E = (a_ * b_ / eta + k * yb * D) / (p.rho + k)
    F = (
        b_ ** 2 / (2.0 * eta)
        + k * yb * E
        + 0.5 * p.sigma0 ** 2 * D
        + (p.sigma ** 2 + p.sigma0 ** 2) * q0(p)
        + p.sigma0 ** 2 * a_
    ) / p.rho
    return D, E, F


def value(p: Params, x, y):
    """V(x, Y) = q0 x^2 + x (A Y + B) + (D/2) Y^2 + E Y + F."""
    D, E, F = w_coeffs(p)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return q0(p) * x * x + x * (A(p) * y + B(p)) + 0.5 * D * y * y + E * y + F


def feedback(p: Params) -> tuple[float, float, float]:
    """Equilibrium feedback v* = K0 X + K1 Y + K2 = (2 q0 X + A Y + B)/eta."""
    return (2.0 * q0(p) / p.eta, A(p) / p.eta, B(p) / p.eta)


def aggregate_drift(p: Params, y):
    """b_hat(Y) = -kappa (Y - Ybar)  [eq. (18)]."""
    return -kappa(p) * (np.asarray(y, dtype=float) - Ybar(p))


def planner_coeffs(p: Params) -> tuple[float, float, float]:
    """Lucas-Prescott planner potential St(Y) = (At/2)Y^2 + Bt Y + Ct solving
        rho St = (St')^2/(2 eta) + (sigma0^2/2) St'' + a Y - ((b + chi)/2) Y^2,
    whose optimal aggregate drift St'/eta coincides with eq. (18)."""
    k = kappa(p)
    At = -p.eta * k
    Bt = p.a / (p.rho + k)
    Ct = (Bt ** 2 / (2.0 * p.eta) + 0.5 * p.sigma0 ** 2 * At) / p.rho
    return At, Bt, Ct


def summary(p: Params) -> dict:
    """All scalar objects in one dictionary."""
    D, E, F = w_coeffs(p)
    am, ap = A_roots(p)
    return dict(
        q0=q0(p), rho_eff=rho_eff(p), kappa_ind=kappa_ind(p), kappa=kappa(p),
        A_minus=am, A_plus=ap, B=B(p), C=C(p), D=D, E=E, F=F,
        Ybar=Ybar(p), lam_bar=lam_bar(p), s_star=s_star(p), sd_lambda=sd_lambda(p),
        P_lam_nonpos=prob_premium_nonpositive(p), Sigma_star=Sigma_star(p),
    )


# --------------------------------------------------------------------------
# Section 8: curved clearing and stationary skewness
# --------------------------------------------------------------------------
def clearing_curved(p: Params, c: float, R: float, ybar: float | None = None):
    """Lambda_c of Assumption 8.1: a - bY - (c/2)Y^2 on I_R = [Ybar - R, Ybar + R],
    continued affinely with slope -b outside I_R (continuous, not C^1)."""
    yb = Ybar(p) if ybar is None else ybar
    lo, hi = yb - R, yb + R
    f_lo = p.a - p.b * lo - 0.5 * c * lo * lo
    f_hi = p.a - p.b * hi - 0.5 * c * hi * hi

    def lam_c(y):
        y = np.asarray(y, dtype=float)
        quad = p.a - p.b * y - 0.5 * c * y * y
        return np.where(y < lo, f_lo - p.b * (y - lo), np.where(y > hi, f_hi - p.b * (y - hi), quad))

    return lam_c


def first_order_coeffs(rho: float, sigma0: float, kap: float, ybar: float) -> tuple[float, float, float]:
    """(p1, r1, s1) of eq. (22) for the correction u1 = p1 Y^2 + r1 Y + s1."""
    p1 = -1.0 / (2.0 * (rho + 3.0 * kap))
    r1 = 2.0 * kap * ybar * p1 / (rho + 2.0 * kap)
    s1 = (sigma0 ** 2 * p1 + kap * ybar * r1) / (rho + kap)
    return p1, r1, s1


def first_order_drift(rho: float, sigma0: float, eta: float, kap: float, ybar: float, c: float):
    """b_hat(Y) = -kappa (Y - Ybar) + (c/eta)(p1 Y^2 + r1 Y + s1) on I_R (Thm 8.3)."""
    p1, r1, s1 = first_order_coeffs(rho, sigma0, kap, ybar)

    def drift(y):
        y = np.asarray(y, dtype=float)
        return -kap * (y - ybar) + (c / eta) * (p1 * y * y + r1 * y + s1)

    return drift


def skew_Y_eq23(rho: float, sigma0: float, eta: float, kap: float, c: float) -> float:
    """Eq. (23): Skew(Y) = -2 c sigma0 / (eta (rho + 3 kappa) (2 kappa)^{3/2}) + O(c^2)."""
    return -2.0 * c * sigma0 / (eta * (rho + 3.0 * kap) * (2.0 * kap) ** 1.5)


def skew_Y_first_order(p: Params, c: float) -> float:
    return skew_Y_eq23(p.rho, p.sigma0, p.eta, kappa(p), c)


def skew_lambda_drift_channel_only(p: Params, c: float) -> float:
    """Premium skewness from the drift channel alone, -Skew(Y): what Skew(lam) would be if the clearing
    map were linear. Proposition 14.2 shows that the clearing channel dominates and reverses its sign."""
    return -skew_Y_first_order(p, c)


def skew_lambda_theta(p: Params) -> float:
    k = kappa(p)
    return p.b * (p.rho + k) / ((p.chi + p.b) * (p.rho + 3.0 * k))


def skew_lambda_first_order(p: Params, c: float) -> float:
    """First-order skewness of the premium (Proposition 14.2) lam = Lambda_c(Y):

        Skew(lam) = -Skew(Y) - 3 c s_star / b + O(c^2)
                  = -(c s_star / b) * (3 - theta) + O(c^2),   0 < theta < 1,

    so Skew(lam) < 0: concave clearing gives a LEFT-skewed premium.  The term
    -3 c s_star/b comes from the curvature of the map Y -> Lambda_c(Y) itself,
    which Theorem 8.3 omits; it is of the same order c as Skew(Y) and always
    dominates it (by a factor 3/theta > 3).
    """
    return -(c * s_star(p) / p.b) * (3.0 - skew_lambda_theta(p))
