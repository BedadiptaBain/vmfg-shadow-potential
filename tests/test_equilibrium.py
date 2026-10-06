"""Independent Riccati fixed point, exact payoffs, optimality."""
import numpy as np
import pytest

from vmfg import lq_check as L, model as M


@pytest.mark.parametrize("p", M.TABLE1_ROWS)
def test_fixed_point_equals_closed_form(p):
    fp = L.equilibrium_fixed_point(p)
    assert abs(fp["kappa"] - M.kappa(p)) < 1e-12
    assert abs(fp["Ybar"] - M.Ybar(p)) < 1e-11
    assert abs(fp["A"] - M.A(p)) < 1e-12
    assert abs(fp["B"] - M.B(p)) < 1e-12
    assert abs(fp["q0"] - M.q0(p)) < 1e-12


@pytest.mark.parametrize("p", M.TABLE1_ROWS)
def test_planner_aggregate(p):
    k, yb = L.planner_aggregate(p)
    assert abs(k - M.kappa(p)) < 1e-12 and abs(yb - M.Ybar(p)) < 1e-11


def test_exact_value_and_local_optimality():
    p = M.Params(a=0.05, b=0.05, chi=0.04, eta=1.0, rho=0.1, sigma=0.2, sigma0=0.15)
    K = np.array(M.feedback(p))
    rng = np.random.default_rng(0)
    for x0, y0 in [(0.0, 0.0), (1.0, 0.3), (-0.5, 1.2)]:
        J = L.policy_value(p, K, x0, y0)
        assert abs(J - float(M.value(p, x0, y0))) < 1e-12
        for _ in range(50):
            dK = rng.normal(size=3) * 1e-2
            assert L.policy_value(p, K + dK, x0, y0) <= J + 1e-13
