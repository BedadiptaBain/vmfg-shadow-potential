#!/usr/bin/env python3
"""01 - Symbolic (SymPy) verification of the algebra in Sections 3-8 and 11.

Each check prints PASS/FAIL and is written to results/01_symbolic_checks.md.
Checks marked "(paper error)" are constructed so that PASS confirms an error
in the draft (e.g. the omitted term in the premium skewness).

Symbols: x individual state, Y aggregate state, z = Y - Ybar, c clearing
curvature, eps a bookkeeping parameter for first-order expansions, q0 value
curvature, ki = kappa_ind, k = kappa, u, w, S, St unknown functions of Y
(shadow price, x-free part of V, shadow potential, planner potential),
Lam a general clearing curve and G an antiderivative of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RESULTS: list[tuple[str, bool]] = []


def is_zero(e) -> bool:
    e = sp.expand(e)
    if e == 0:
        return True
    if sp.cancel(sp.together(e)) == 0:
        return True
    return sp.simplify(e) == 0


def check(name: str, ok: bool) -> bool:
    RESULTS.append((name, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


x, Y, z, c, eps = sp.symbols("x Y z c epsilon", real=True)
a, b, chi, eta, rho, sig, sig0 = sp.symbols("a b chi eta rho sigma sigma_0", positive=True)
q = sp.Symbol("q_0", real=True)
ki, k, Yb = sp.symbols("kappa_ind kappa Ybar", positive=True)
u, w = sp.Function("u")(Y), sp.Function("w")(Y)
S, St = sp.Function("S")(Y), sp.Function("Stilde")(Y)
Lam, G = sp.Function("Lambda")(Y), sp.Function("G")(Y)


def hjb(V, bhat, Lam_):
    """rho V = V_x^2/(2 eta) + bhat V_Y + (sig^2+sig0^2)/2 V_xx + sig0^2/2 V_YY
               + sig0^2 V_xY + Lam x - chi/2 x^2   (written as RHS - LHS)."""
    return (-rho * V + V.diff(x) ** 2 / (2 * eta) + bhat * V.diff(Y)
            + (sig ** 2 + sig0 ** 2) / 2 * V.diff(x, 2) + sig0 ** 2 / 2 * V.diff(Y, 2)
            + sig0 ** 2 * V.diff(x).diff(Y) + Lam_ * x - chi / 2 * x ** 2)


# ---------------------------------------------------------------- Section 3
V = q * x ** 2 + x * u + w
He = sp.expand(hjb(V, (u + 2 * q * Y) / eta, Lam))
eq7 = -rho * q - chi / 2 + 2 * q ** 2 / eta
rho_eff = rho - 2 * q / eta
eq1 = sig0 ** 2 / 2 * u.diff(Y, 2) + (u + 2 * q * Y) * u.diff(Y) / eta - rho_eff * u + Lam
eq9 = (-rho * w + u ** 2 / (2 * eta) + (u + 2 * q * Y) / eta * w.diff(Y) + sig0 ** 2 / 2 * w.diff(Y, 2)
       + (sig ** 2 + sig0 ** 2) * q + sig0 ** 2 * u.diff(Y))
check("Prop 3.1: HJB maps quadratics in x to quadratics (no x^3, x^4 terms)",
      He.coeff(x, 3) == 0 and He.coeff(x, 4) == 0)
check("Lemma 3.3: x^2 coefficient is eq. (7)", is_zero(He.coeff(x, 2) - eq7))
check("Lemma 3.3: x^1 coefficient is eq. (1) with rho_eff = rho - 2 q0/eta", is_zero(He.coeff(x, 1) - eq1))
check("Lemma 3.3: x^0 coefficient is eq. (9)", is_zero(He.coeff(x, 0) - eq9))
qneg = eta / 4 * (rho - sp.sqrt(rho ** 2 + 4 * chi / eta))
check("(7): the printed negative root solves (7)", is_zero(eq7.subs(q, qneg)))
check("(8): rho_eff = rho/2 + sqrt(rho^2 + 4 chi/eta)/2",
      is_zero(rho - 2 * qneg / eta - (rho / 2 + sp.sqrt(rho ** 2 + 4 * chi / eta) / 2)))
check("Remark 3.4: rho_eff/rho = 2.5616 at chi=0.04, rho=0.1, eta=1",
      abs(float((rho - 2 * qneg / eta).subs({chi: 0.04, rho: 0.1, eta: 1}) / 0.1) - 2.5616) < 5e-5)

# Remark 3.2: non-constant curvature q(Y)
qf = sp.Function("q")(Y)
H2 = sp.expand(hjb(qf * x ** 2 + x * u + w, (u + 2 * qf * Y) / eta, Lam))
extra = sp.simplify(H2.coeff(x, 1) - eq1.subs(q, qf))
check("Remark 3.2 (paper error): with q = q(Y) the x^1 equation gains the SOURCE term "
      "2 sigma0^2 q'(Y), not a change in the coefficient of u'",
      is_zero(extra - 2 * sig0 ** 2 * qf.diff(Y)) and not extra.has(u.diff(Y)))
check("Remark 3.2: general x^2 equation rho q = -chi/2 + 2q^2/eta + bhat q' + sig0^2/2 q''",
      is_zero(H2.coeff(x, 2) - (-rho * qf - chi / 2 + 2 * qf ** 2 / eta
                                + (u + 2 * qf * Y) / eta * qf.diff(Y) + sig0 ** 2 / 2 * qf.diff(Y, 2))))

# ---------------------------------------------------------------- Section 4
Lop = lambda F_: (sig0 ** 2 / 2 * F_.diff(Y, 2) + F_.diff(Y) ** 2 / (2 * eta)
                  + 2 * q / eta * Y * F_.diff(Y) - rho * F_ + G)
dL = Lop(S).diff(Y).subs(G.diff(Y), Lam)
check("Theorem 4.1: d/dY L[S] equals the LHS of (1) with u = S'",
      is_zero(dL - eq1.subs(u, S.diff(Y)).doit()))
check("Theorem 4.1: L[S + const] = L[S] - rho*const (constant absorbed into S)",
      is_zero(Lop(S + sp.Symbol("K0")) - Lop(S) + rho * sp.Symbol("K0")))
check("Remark 4.2: grouped form (2q0/eta)(Y S' - S) - rho_eff S equals (2q0/eta) Y S' - rho S",
      is_zero(2 * q / eta * (Y * S.diff(Y) - S) - rho_eff * S - (2 * q / eta * Y * S.diff(Y) - rho * S)))
Pop = sig0 ** 2 / 2 * St.diff(Y, 2) + St.diff(Y) ** 2 / (2 * eta) - rho * St + G - chi / 2 * Y ** 2
Lsub = Lop(St - q * Y ** 2).doit()
check("NEW (prior art): L[St - q0 Y^2] = P[St] - sigma0^2 q0 - Y^2 * (eq.7 residual), i.e. the shadow "
      "potential is the Lucas-Prescott planner potential St shifted by -q0 Y^2, for ANY clearing curve",
      is_zero(Lsub - Pop + sig0 ** 2 * q + Y ** 2 * eq7))

A_, B_, C_ = sp.symbols("A B C", real=True)
Sq = A_ / 2 * Y ** 2 + B_ * Y + C_
Lq = sp.expand(Lop(Sq).subs(G, a * Y - b / 2 * Y ** 2))
check("Cor 4.3: 2 eta * [Y^2] = A^2 + (4 q0 - rho eta) A - b eta",
      is_zero(2 * eta * Lq.coeff(Y, 2) - (A_ ** 2 + (4 * q - rho * eta) * A_ - b * eta)))
check("Cor 4.3: [Y^1] = 0 gives B = a/(rho - (2 q0 + A)/eta)",
      is_zero(Lq.coeff(Y, 1).subs(B_, a / (rho - (2 * q + A_) / eta))))
check("Cor 4.3: [Y^0] = 0 gives C = (B^2/(2 eta) + sigma0^2 A/2)/rho",
      is_zero(Lq.coeff(Y, 0).subs(C_, (B_ ** 2 / (2 * eta) + sig0 ** 2 * A_ / 2) / rho)))

# ------------------------------------------------------------ Section 5 (kappa parametrisation)
chi_k = eta * ki * (ki + rho)
b_k = eta * k * (k + rho) - chi_k
qk, Ak = -eta * ki / 2, eta * (ki - k)
check("Prop 5.1: q0 = -eta kappa_ind/2 solves (7) iff eta ki(ki + rho) = chi",
      is_zero(eq7.subs(q, qk).subs(chi, chi_k)))
check("Prop 5.1: A^2 + (4q0 - rho eta)A - b eta = eta[eta k(k+rho) - chi - b] at A = eta(ki - k)",
      is_zero(((Ak ** 2 + (4 * qk - rho * eta) * Ak - b * eta)
               - eta * (eta * k * (k + rho) - chi - b)).subs(chi, chi_k)))
check("Prop 5.1: rho_eff = rho + kappa_ind", is_zero(rho_eff.subs(q, qk) - (rho + ki)))
Bk = a / (rho - (2 * qk + Ak) / eta)
check("Thm 5.4: B = a/(rho + kappa)", is_zero(Bk - a / (rho + k)))
Ybar_k = Bk / (eta * k)
check("Thm 5.4: Ybar = B/(eta kappa) = a/(b + chi)", is_zero((Ybar_k - a / (b + chi)).subs(b, b_k).subs(chi, chi_k)))
check("Thm 5.4: lam_bar = a chi/(chi + b)", is_zero(a - b * a / (b + chi) - a * chi / (chi + b)))
check("Thm 5.3: aggregate drift (u + 2 q0 Y)/eta = -kappa (Y - Ybar)",
      is_zero(((Ak * Y + Bk + 2 * qk * Y) / eta + k * (Y - Ybar_k))))
check("Lucas-Prescott planner: At = -eta kappa solves At^2 - rho eta At - (b+chi) eta = 0",
      is_zero(((-eta * k) ** 2 - rho * eta * (-eta * k) - (b + chi) * eta).subs(b, b_k).subs(chi, chi_k)))

# full HJB residual for V = q0 x^2 + x(AY+B) + w with the closed-form w
Dk = Ak ** 2 / (eta * (rho + 2 * k))
Ek = (Ak * Bk / eta + k * Ybar_k * Dk) / (rho + k)
Fk = (Bk ** 2 / (2 * eta) + k * Ybar_k * Ek + sig0 ** 2 * Dk / 2 + (sig ** 2 + sig0 ** 2) * qk + sig0 ** 2 * Ak) / rho
Vfull = qk * x ** 2 + x * (Ak * Y + Bk) + Dk / 2 * Y ** 2 + Ek * Y + Fk
res = sp.expand(hjb(Vfull, (Ak * Y + Bk + 2 * qk * Y) / eta, a - b * Y).subs(b, b_k).subs(chi, chi_k))
check("Prop 6.1: V = q0 x^2 + x u + w with w = (D/2)Y^2 + EY + F solves the HJB identically",
      all(is_zero(res.coeff(x, i).coeff(Y, j)) for i in range(3) for j in range(3)))

# ---------------------------------------------------------------- Section 7
check("Prop 7.1 (strengthened): d(X - Y) has drift -kappa_ind (X - Y) for ANY shadow price u(Y), "
      "so X - Y is independent of F^0 and ALL centred conditional moments are deterministic",
      is_zero((2 * qk * x + u) / eta - (2 * qk * Y + u) / eta + ki * (x - Y)))
check("(20): sigma^2/(2 kappa_ind) = eta sigma^2/(4|q0|)", is_zero(sig ** 2 / (2 * ki) - eta * sig ** 2 / (4 * (-qk))))

# ---------------------------------------------------------------- Section 8
a_Y = eta * k * (k + rho) * Yb                     # a = (b + chi) Ybar
B0 = a_Y / (rho + k)
u1 = sp.Function("u1")(Y)
uu = Ak * Y + B0 + c * u1
E1 = sp.expand(sig0 ** 2 / 2 * uu.diff(Y, 2) + (uu + 2 * qk * Y) * uu.diff(Y) / eta
               - (rho - 2 * qk / eta) * uu + a_Y - b_k * Y - c / 2 * Y ** 2)
check("Thm 8.3: order c^0 of (1) vanishes for u0 = AY + B", is_zero(E1.coeff(c, 0)))
eq24 = sig0 ** 2 / 2 * u1.diff(Y, 2) - k * (Y - Yb) * u1.diff(Y) - (rho + k) * u1 - Y ** 2 / 2
check("Thm 8.3: order c^1 of (1) is eq. (24)", is_zero(E1.coeff(c, 1) - eq24))
p1 = -1 / (2 * (rho + 3 * k))
r1 = 2 * k * Yb * p1 / (rho + 2 * k)
s1 = (sig0 ** 2 * p1 + k * Yb * r1) / (rho + k)
check("Thm 8.3: (22) solves (24)", is_zero(eq24.subs(u1, p1 * Y ** 2 + r1 * Y + s1).doit()))

s_, d1, c1 = sp.symbols("s delta_1 c_1", positive=True)
l1 = sp.Symbol("ell_1", real=True)


def gauss_E(expr):
    """E[expr] for z ~ N(0, s^2), expr polynomial in z."""
    P = sp.Poly(sp.expand(expr), z)
    return sum(co * s_ ** d * sp.factorial2(d - 1) for (d,), co in P.terms() if d % 2 == 0)


def E1st(expr):
    """First-order expectation under density ~ phi_s(z) (1 + eps (d1 z^3 + l1 z))."""
    return sp.series(gauss_E(expr * (1 + eps * (d1 * z ** 3 + l1 * z))), eps, 0, 2).removeO()


m1 = E1st(z)
m3 = sp.series(E1st(sp.expand((z - m1) ** 3)), eps, 0, 2).removeO()
check("Thm 8.3 proof (imprecise): E[z] = 3 delta s^4 + ell s^2 -- the linear term DOES shift the mean "
      "(the draft writes 3 delta s^4)", is_zero(m1 - eps * (3 * d1 * s_ ** 4 + l1 * s_ ** 2)))
check("Thm 8.3 proof: m3 = 6 delta s^6 (the ell-terms cancel), so Skew(Y) = 6 delta s^3",
      is_zero(m3 - 6 * eps * d1 * s_ ** 6))
lam_z = -b * (Yb + z) - eps * c1 / 2 * (Yb + z) ** 2          # premium up to the constant a
mL = E1st(lam_z)
m3L = sp.series(E1st(sp.expand((lam_z - mL) ** 3)), eps, 0, 2).removeO()
skewL = sp.expand(m3L / (b ** 3 * s_ ** 3))                 # m3L = O(eps), var = b^2 s^2 + O(eps)
check("Thm 8.3 (paper error): Skew(lam) = -Skew(Y) - 3 c s/b at first order "
      "(the map Y -> Lambda_c(Y) has curvature of the same order c)",
      is_zero(skewL - (-6 * eps * d1 * s_ ** 3 - 3 * eps * c1 * s_ / b)))
check("Negative control: the draft's relation Skew(lam) = -Skew(Y) is NOT an identity",
      not is_zero(skewL - (-6 * eps * d1 * s_ ** 3)))
check("Negative control: Remark 3.2's reading (extra term 2 sigma0^2 q' multiplying u') is NOT what the HJB gives",
      not is_zero(extra - 2 * sig0 ** 2 * qf.diff(Y) * u.diff(Y)))
skew_lam_closed = skewL.subs({d1: 2 * c1 * (p1 / eta) / (3 * sig0 ** 2)}).subs(s_, sig0 / sp.sqrt(2 * k))
theta = b * (rho + k) / ((chi + b) * (rho + 3 * k))
target = -(eps * c1 * (sig0 / sp.sqrt(2 * k)) / b) * (3 - theta)
check("Corrected closed form: Skew(lam) = -(c s*/b)(3 - theta), theta = b(rho+k)/((chi+b)(rho+3k)) in (0,1)",
      is_zero((skew_lam_closed - target).subs(chi, eta * k * (k + rho) - b)))
skewY_closed = (6 * eps * d1 * s_ ** 3).subs({d1: 2 * c1 * (p1 / eta) / (3 * sig0 ** 2)}).subs(s_, sig0 / sp.sqrt(2 * k))
check("(23): Skew(Y) = -2 c sigma0 / (eta (rho + 3k)(2k)^{3/2})",
      is_zero(skewY_closed + 2 * eps * c1 * sig0 / (eta * (rho + 3 * k) * (2 * k) ** sp.Rational(3, 2))))
fk = (k + rho) / ((rho + 3 * k) * sp.sqrt(k))
check("(25): |Skew(Y)| = c sigma0 f(k)/(sqrt2 (b+chi)) after eta = (b+chi)/(k(k+rho))",
      is_zero((-skewY_closed / eps).subs(eta, (b + chi) / (k * (k + rho))) - c1 * sig0 * fk / (sp.sqrt(2) * (b + chi))))
check("(25): f'/f = -2 rho/((k+rho)(rho+3k)) - 1/(2k)",
      is_zero(sp.diff(sp.log(fk), k) - (-2 * rho / ((k + rho) * (rho + 3 * k)) - 1 / (2 * k))))

# ---------------------------------------------------------------- Section 11(2a)
Kc, Sig2 = sp.symbols("K Sigma2", positive=True)
Vlog = Kc * sp.log(x)
bal = sp.expand(Vlog.diff(x) ** 2 / (2 * eta) + Sig2 / 2 * Vlog.diff(x, 2))
check("Sec 11(2a) (paper error): dominant balance V_x^2/(2eta) + (Sigma^2/2) V_xx = 0 with V = K ln x "
      "forces K = eta*Sigma^2, Sigma^2 = sigma^2 + sigma0^2 (draft: eta sigma^2/2); optimal drift "
      "Sigma^2/x is a Bessel-3 drift",
      sp.solve(sp.Eq(bal * x ** 2, 0), Kc) == [eta * Sig2])

# ---------------------------------------------------------------- Remark 2.6 limit
check("Remark 2.6: chi -> 0 gives q0 -> 0 and lam_bar -> 0",
      sp.limit(qneg, chi, 0) == 0 and sp.limit(a * chi / (chi + b), chi, 0) == 0)

# ---------------------------------------------------------------- report
n_ok = sum(ok for _, ok in RESULTS)
lines = ["# 01 - Symbolic verification", "",
         f"{n_ok}/{len(RESULTS)} checks pass. Checks labelled *(paper error)* pass because "
         "they confirm an error in the draft.", "", "| # | check | result |", "|---|---|---|"]
lines += [f"| {i} | {name} | {'PASS' if ok else 'FAIL'} |" for i, (name, ok) in enumerate(RESULTS, 1)]
(OUT / "01_symbolic_checks.md").write_text("\n".join(lines) + "\n")
print(f"\n{n_ok}/{len(RESULTS)} passed -> results/01_symbolic_checks.md")
sys.exit(0 if n_ok == len(RESULTS) else 1)
