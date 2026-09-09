import numpy as np

from paths import simulate_gbm_paths


def price_autocall(S0, T, r, sigma, n_obs, coupon_rate, call_level,
                   barrier_level, notional=1000, n_simulations=200_000, seed=None):
    """
    Price un autocallable a observation annuelle.

    n_obs         : nombre de dates d'observation (une par an)
    coupon_rate   : coupon annuel, ex 0.08
    call_level    : niveau de rappel en % de S0, ex 1.00
    barrier_level : barriere de protection en % de S0, ex 0.65
    """
    paths = simulate_gbm_paths(S0, T, r, sigma, n_steps=n_obs,
                               n_simulations=n_simulations, seed=seed)

    call_barrier = S0 * call_level
    protection = S0 * barrier_level

    payoffs = np.zeros(n_simulations)
    exit_times = np.full(n_simulations, T, dtype=float)
    alive = np.ones(n_simulations, dtype=bool)

    dt = T / n_obs

    # Dates d'observation intermediaires : test de rappel
    for i in range(1, n_obs + 1):
        t = i * dt
        S_obs = paths[:, i]

        called = alive & (S_obs >= call_barrier)

        payoffs[called] = notional * (1 + coupon_rate * i)
        exit_times[called] = t
        alive[called] = False

    # Les survivants : traitement a l'echeance
    S_final = paths[:, -1]
    protected = alive & (S_final >= protection)
    breached = alive & (S_final < protection)

    payoffs[protected] = notional
    payoffs[breached] = notional * S_final[breached] / S0

    discounted = payoffs * np.exp(-r * exit_times)
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(n_simulations)

    stats = {
        "prix": price,
        "se": se,
        "prob_rappel_an1": (exit_times == dt).mean(),
        "prob_survie_echeance": alive.mean(),
        "prob_perte_capital": breached.mean(),
        "maturite_moyenne": exit_times.mean(),
    }
    return stats

if __name__ == "__main__":
    base = dict(S0=100, T=4, r=0.05, n_obs=4, coupon_rate=0.08,
                call_level=1.00, barrier_level=0.65,
                n_simulations=200_000, seed=42)

    s = price_autocall(sigma=0.20, **base)
    print("Autocall 4 ans, coupon 8%, rappel 100%, barriere 65%, sigma 20%\n")
    print(f"  Prix                    = {s['prix']:.2f} EUR  (nominal 1000)")
    print(f"  SE                      = {s['se']:.2f}")
    print(f"  P(rappel en annee 1)    = {s['prob_rappel_an1']:.1%}")
    print(f"  P(survie a l'echeance)  = {s['prob_survie_echeance']:.1%}")
    print(f"  P(perte en capital)     = {s['prob_perte_capital']:.1%}")
    print(f"  Maturite moyenne        = {s['maturite_moyenne']:.2f} ans")

    print("\nSensibilite a la volatilite :")
    for vol in [0.15, 0.20, 0.30, 0.40]:
        r = price_autocall(sigma=vol, **base)
        print(f"  sigma = {vol:.0%}   prix = {r['prix']:7.2f}   "
              f"P(perte) = {r['prob_perte_capital']:5.1%}   "
              f"maturite = {r['maturite_moyenne']:.2f} ans")

    ref = s["prix"]
    print(f"\nRisque de modele (reference = {ref:.2f}) :")
    for label, kw in [
        ("sigma 18%",              dict(sigma=0.18)),
        ("sigma 22%",              dict(sigma=0.22)),
        ("sigma 28% (skew du put)", dict(sigma=0.28)),
        ("taux 3%",                dict(sigma=0.20, r=0.03)),
        ("taux 7%",                dict(sigma=0.20, r=0.07)),
    ]:
        p = price_autocall(**{**base, **kw})["prix"]
        print(f"  {label:24s} prix = {p:7.2f}   ecart = {p - ref:+7.2f}")

    print(f"\n  Erreur Monte Carlo (SE)  = {s['se']:.2f} EUR")
    print(f"  Risque de modele (sigma) = ~50 EUR")