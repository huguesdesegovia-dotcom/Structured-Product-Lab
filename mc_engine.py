import numpy as np
from scipy.stats import norm


def monte_carlo_price(payoffs, r, T, confidence=0.95):
    """
    Estime le prix d'un produit a partir de ses payoffs simules.

    payoffs : tableau 1D des payoffs a l'echeance (un par trajectoire)
    Retourne (prix, erreur_standard, (borne_basse, borne_haute))
    """
    discounted = payoffs * np.exp(-r * T)
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(len(discounted))
    z = norm.ppf(0.5 + confidence / 2)
    ci = (price - z * se, price + z * se)

    return price, se, ci


if __name__ == "__main__":
    from paths import simulate_gbm_paths

    p = simulate_gbm_paths(S0=100, T=1, r=0.05, sigma=0.20,
                           n_steps=252, n_simulations=100_000, seed=42)
    S_T = p[:, -1]
    payoffs = np.maximum(S_T - 100, 0)

    price, se, ci = monte_carlo_price(payoffs, r=0.05, T=1)
    print(f"Prix MC  = {price:.4f}")
    print(f"SE       = {se:.4f}")
    print(f"IC 95%   = [{ci[0]:.4f} ; {ci[1]:.4f}]")
    print(f"Attendu (Black-Scholes) = 10.4506")