"""Closed forms of Sections 3-5 and the printed Table 1."""
import math

import pytest

from vmfg import model as M


@pytest.mark.parametrize("p,printed", list(zip(M.TABLE1_ROWS, M.TABLE1_PRINTED)))
def test_table1_values(p, printed):
    kin, k, yb, lb = printed
    assert round(M.kappa_ind(p), 4) == kin
    assert round(M.kappa(p), 4) == k
    assert round(M.Ybar(p), 4) == yb
    assert round(M.lam_bar(p), 6) == lb


@pytest.mark.parametrize("p", M.TABLE1_ROWS)
def test_identities(p):
    q, kin, k, A = M.q0(p), M.kappa_ind(p), M.kappa(p), M.A(p)
    assert math.isclose(p.rho * q, -p.chi / 2 + 2 * q * q / p.eta, rel_tol=0, abs_tol=1e-15)       # (7)
    assert math.isclose(q, M.q0_literal(p), rel_tol=1e-12)                                          # printed (7)
    assert abs(A * A + (4 * q - p.rho * p.eta) * A - p.b * p.eta) < 1e-14                          # (14)
    assert math.isclose(p.eta * k * (k + p.rho), p.chi + p.b, rel_tol=1e-13)                        # (16)
    assert k > kin > 0 and A < 0                                                                    # Prop 5.1
    am, ap = M.A_roots(p)
    assert am + 2 * q < 0 < ap + 2 * q                                                              # Cor 4.3
    assert math.isclose(M.B(p) / (p.eta * k), M.Ybar(p), rel_tol=1e-13)                             # Thm 5.4
    assert math.isclose(p.a - p.b * M.Ybar(p), M.lam_bar(p), rel_tol=1e-13)
    assert math.isclose(M.lam_bar(p), p.chi * M.Ybar(p), rel_tol=1e-13)   # static competitive price


def test_remark_3_4_factor():
    p = M.Params(a=0.05, b=0.05, chi=0.04, eta=1.0, rho=0.1)
    assert round(M.rho_eff(p) / p.rho, 2) == 2.56


def test_eta_span_is_twentyfold_not_fortyfold():
    etas = [p.eta for p in M.TABLE1_ROWS]
    assert max(etas) / min(etas) == 20


def test_planner_potential_matches_aggregate_drift():
    p = M.TABLE1_ROWS[0]
    At, Bt, _ = M.planner_coeffs(p)
    # planner drift St'/eta = (At Y + Bt)/eta must equal -kappa (Y - Ybar)
    assert math.isclose(-At / p.eta, M.kappa(p), rel_tol=1e-13)
    assert math.isclose(Bt / (-At), M.Ybar(p), rel_tol=1e-13)
    # and the shadow potential is the planner potential shifted by -q0 Y^2
    assert math.isclose(M.A(p), At - 2 * M.q0(p), rel_tol=1e-13)
    assert math.isclose(M.B(p), Bt, rel_tol=1e-13)


def test_params_validation():
    with pytest.raises(ValueError):
        M.Params(a=0.05, b=-1.0, chi=0.04, eta=1.0, rho=0.1)
