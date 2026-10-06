"""Proposition 14.2 and Appendix B: the sign and size of the premium skewness under curved clearing."""
import pytest

from vmfg import model as M, numerics as N

P1 = M.Params(a=0.05, b=0.05, chi=0.04, eta=1.0, rho=0.10, sigma=0.2, sigma0=0.15)


@pytest.mark.parametrize("kap,c,printed", [(0.25, 0.2, 0.5752), (0.40, 0.01, 0.9797), (0.60, 0.0005, 0.9995)])
def test_table2_entries(kap, c, printed):
    assert round(N.table2_ratio(c, kap), 4) == printed


def test_premium_skewness_is_negative_and_matches_proposition_14_2():
    c = 0.0005
    q = N.skewness_first_order_drift(P1, c)
    assert q["skew_lam"] < 0 < M.skew_lambda_drift_channel_only(P1, c)   # the clearing channel reverses the sign
    assert abs(q["skew_lam"] / M.skew_lambda_first_order(P1, c) - 1) < 0.01
    assert abs(q["skew_Y"] / M.skew_Y_first_order(P1, c) - 1) < 0.01


def test_nonperturbative_solver_agrees():
    c = 0.002
    nl = N.skewness_nonlinear(P1, c, n=4001)
    assert nl["skew_lam"] < 0
    assert abs(nl["skew_lam"] / M.skew_lambda_first_order(P1, c) - 1) < 0.05


def test_nonlinear_solver_recovers_affine_shadow_price():
    import numpy as np
    s = M.s_star(P1)
    sol = N.solve_shadow_potential(P1, lambda y: P1.a - P1.b * y, M.Ybar(P1) - 10 * s, M.Ybar(P1) + 10 * s, n=4001)
    core = np.abs(sol["y"] - M.Ybar(P1)) < 5 * s
    assert np.max(np.abs(sol["u"][core] - (M.A(P1) * sol["y"][core] + M.B(P1)))) < 1e-3
    assert sol["iterations"] < 20


def test_theta_in_unit_interval():
    for p in M.TABLE1_ROWS:
        assert 0 < M.skew_lambda_theta(p) < 1
