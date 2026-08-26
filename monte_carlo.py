import numpy as np


def monte_carlo_call_price(S, K, T, r, sigma, n_simulations=100000):
    Z = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)

    return price


if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    mc_price = monte_carlo_call_price(S, K, T, r, sigma)
    print("Monte Carlo call price =", mc_price)