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

def monte_carlo_price_antithetic(payoffs, r, T, confidence=0.95):
    """
    Prix Monte Carlo pour des payoffs issus de trajectoires antithetiques.

    L'unite d'observation independante est la PAIRE, pas le payoff individuel.
    """
    discounted = payoffs * np.exp(-r * T)
    half = len(discounted) // 2

    paired = 0.5 * (discounted[:half] + discounted[half:])

    price = paired.mean()
    se = paired.std(ddof=1) / np.sqrt(half)
    z = norm.ppf(0.5 + confidence / 2)
    ci = (price - z * se, price + z * se)

    return price, se, ci

if __name__ == "__main__":
    from paths import simulate_gbm_paths, simulate_gbm_paths_antithetic

    params = dict(S0=100, T=1, r=0.05, sigma=0.20,
                  n_steps=252, n_simulations=100_000, seed=42)

    # Standard
    p_std = simulate_gbm_paths(**params)
    payoffs_std = np.maximum(p_std[:, -1] - 100, 0)
    price_std, se_std, ci_std = monte_carlo_price(payoffs_std, r=0.05, T=1)

    # Antithetique
    p_anti = simulate_gbm_paths_antithetic(**params)
    payoffs_anti = np.maximum(p_anti[:, -1] - 100, 0)
    price_anti, se_anti, ci_anti = monte_carlo_price_antithetic(payoffs_anti, r=0.05, T=1)

    print(f"Black-Scholes           = 10.4506")
    print(f"MC standard             = {price_std:.4f}  SE = {se_std:.4f}")
    print(f"MC antithetique         = {price_anti:.4f}  SE = {se_anti:.4f}")
    print(f"Reduction du SE         = {se_std / se_anti:.2f}x")
    print(f"Gain en simulations     = {(se_std / se_anti)**2:.1f}x")