import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from paths import simulate_gbm_paths, simulate_gbm_paths_antithetic
from mc_engine import monte_carlo_price, monte_carlo_price_antithetic
from barrier import payoff_down_and_out_call, payoff_down_and_in_call
from autocallable import price_autocall

PARAMS = dict(S0=100, T=1, r=0.05, sigma=0.20, n_steps=252,
              n_simulations=100_000, seed=42)
BS_CALL = 10.4506


def test_paths_shape():
    p = simulate_gbm_paths(**PARAMS)
    assert p.shape == (100_000, 253)


def test_paths_start_at_spot():
    p = simulate_gbm_paths(**PARAMS)
    assert np.allclose(p[:, 0], 100)


def test_risk_neutral_martingale():
    """E[exp(-rT) * S_T] doit valoir S0 : valide le drift et le -sigma^2/2."""
    p = simulate_gbm_paths(**PARAMS)
    esp = np.exp(-0.05) * p[:, -1].mean()
    assert abs(esp - 100) < 0.5


def test_mc_converges_to_black_scholes():
    p = simulate_gbm_paths(**PARAMS)
    price, se, ci = monte_carlo_price(np.maximum(p[:, -1] - 100, 0), r=0.05, T=1)
    assert ci[0] < BS_CALL < ci[1]


def test_antithetic_reduces_variance():
    p_std = simulate_gbm_paths(**PARAMS)
    p_anti = simulate_gbm_paths_antithetic(**PARAMS)
    _, se_std, _ = monte_carlo_price(np.maximum(p_std[:, -1] - 100, 0), r=0.05, T=1)
    _, se_anti, _ = monte_carlo_price_antithetic(np.maximum(p_anti[:, -1] - 100, 0), r=0.05, T=1)
    assert se_anti < se_std


def test_barrier_parity():
    """in + out = vanille, sur les memes trajectoires."""
    p = simulate_gbm_paths(**PARAMS)
    do = monte_carlo_price(payoff_down_and_out_call(p, 100, 85), r=0.05, T=1)[0]
    di = monte_carlo_price(payoff_down_and_in_call(p, 100, 85), r=0.05, T=1)[0]
    van = monte_carlo_price(np.maximum(p[:, -1] - 100, 0), r=0.05, T=1)[0]
    assert abs((do + di) - van) < 1e-8


def test_barrier_out_cheaper_than_vanilla():
    p = simulate_gbm_paths(**PARAMS)
    do = monte_carlo_price(payoff_down_and_out_call(p, 100, 85), r=0.05, T=1)[0]
    van = monte_carlo_price(np.maximum(p[:, -1] - 100, 0), r=0.05, T=1)[0]
    assert do < van


def test_autocall_price_decreases_with_vol():
    """L'investisseur est short le put : vega negatif."""
    base = dict(S0=100, T=4, r=0.05, n_obs=4, coupon_rate=0.08,
                call_level=1.00, barrier_level=0.65,
                n_simulations=50_000, seed=42)
    low = price_autocall(sigma=0.15, **base)["prix"]
    high = price_autocall(sigma=0.40, **base)["prix"]
    assert high < low


def test_autocall_higher_coupon_raises_price():
    base = dict(S0=100, T=4, r=0.05, sigma=0.20, n_obs=4,
                call_level=1.00, barrier_level=0.65,
                n_simulations=50_000, seed=42)
    p8 = price_autocall(coupon_rate=0.08, **base)["prix"]
    p15 = price_autocall(coupon_rate=0.15, **base)["prix"]
    assert p15 > p8