import numpy as np

from paths import simulate_gbm_paths
from mc_engine import monte_carlo_price


def payoff_down_and_out_call(paths, K, barrier):
    """Call desactive si le minimum du chemin touche la barriere."""
    touched = paths.min(axis=1) <= barrier
    vanilla = np.maximum(paths[:, -1] - K, 0)
    return vanilla * (~touched)


def payoff_down_and_in_call(paths, K, barrier):
    """Call active seulement si la barriere est touchee."""
    touched = paths.min(axis=1) <= barrier
    vanilla = np.maximum(paths[:, -1] - K, 0)
    return vanilla * touched


def payoff_digital_barrier(paths, coupon, barrier):
    """Coupon fixe verse si le chemin reste au-dessus de la barriere (produit D)."""
    touched = paths.min(axis=1) <= barrier
    return coupon * (~touched)

if __name__ == "__main__":
    params = dict(S0=100, T=1, r=0.05, sigma=0.20, n_simulations=200_000, seed=42)
    K, B = 100, 85

    # 1. Relation in + out = vanille
    p = simulate_gbm_paths(n_steps=252, **params)
    do = monte_carlo_price(payoff_down_and_out_call(p, K, B), r=0.05, T=1)[0]
    di = monte_carlo_price(payoff_down_and_in_call(p, K, B), r=0.05, T=1)[0]
    van = monte_carlo_price(np.maximum(p[:, -1] - K, 0), r=0.05, T=1)[0]

    print(f"Down-and-out  = {do:.4f}")
    print(f"Down-and-in   = {di:.4f}")
    print(f"Somme         = {do + di:.4f}")
    print(f"Vanille       = {van:.4f}   (Black-Scholes = 10.4506)")

    # 2. Biais de discretisation
    print("\nBiais de discretisation (down-and-out) :")
    for n in [12, 52, 252, 1_000, 5_000]:
        p_n = simulate_gbm_paths(n_steps=n, **params)
        price, se, _ = monte_carlo_price(payoff_down_and_out_call(p_n, K, B), r=0.05, T=1)
        print(f"  n_steps = {n:5d}   prix = {price:.4f}   SE = {se:.4f}")

    # 3. Probabilite de toucher la barriere
    print("\nProbabilite de franchissement :")
    for n in [12, 252, 5_000]:
        p_n = simulate_gbm_paths(n_steps=n, **params)
        prob = (p_n.min(axis=1) <= B).mean()
        print(f"  n_steps = {n:5d}   P(touche) = {prob:.2%}")

    