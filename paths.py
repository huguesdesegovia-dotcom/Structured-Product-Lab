import numpy as np


def simulate_gbm_paths(S0, T, r, sigma, n_steps, n_simulations, seed=None):
    """
    Simule des trajectoires de prix sous un mouvement brownien geometrique.

    Retourne un tableau de forme (n_simulations, n_steps + 1),
    ou la colonne 0 vaut S0 pour toutes les trajectoires.
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_simulations, n_steps))

    paths = np.empty((n_simulations, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    return paths

def simulate_gbm_paths_antithetic(S0, T, r, sigma, n_steps, n_simulations, seed=None):
    """
    Simule des trajectoires GBM avec variables antithetiques.

    Les n_simulations/2 premieres lignes utilisent Z, les suivantes -Z.
    La ligne i et la ligne i + n_simulations//2 forment une paire.
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    half = n_simulations // 2
    Z_half = rng.standard_normal((half, n_steps))
    Z = np.concatenate([Z_half, -Z_half], axis=0)

    paths = np.empty((2 * half, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    return paths

if __name__ == "__main__":
    # Test visuel : 5 trajectoires, 5 pas
    p = simulate_gbm_paths(S0=100, T=1, r=0.05, sigma=0.20,
                           n_steps=5, n_simulations=5, seed=42)
    print(p.shape)
    print(p)

    # Test de coherence risque-neutre : E[S_T] actualise doit valoir S0
    p_big = simulate_gbm_paths(S0=100, T=1, r=0.05, sigma=0.20,
                               n_steps=252, n_simulations=100_000, seed=42)
    S_T = p_big[:, -1]
    print("\nE[S_T] actualise =", np.exp(-0.05) * S_T.mean())
    print("Attendu           = 100")

    dt = 1 / 252
    for t in [50, 100, 150, 200, 252]:
        esp = np.exp(-0.05 * t * dt) * p_big[:, t].mean()
        print(f"t={t:3d}  E[S_t] actualise = {esp:.3f}")

