# Structured Product Lab

🚀 **[Live Demo](https://structured-appuct-lab-qqdtpjfr6gawftzzobih9x.streamlit.app)**

Monte Carlo pricing of barrier options and autocallables, with variance reduction and model risk analysis.

## Overview

This project prices path-dependent structured products that have no closed-form solution — barrier options and autocallables — using Monte Carlo simulation under a geometric Brownian motion. It implements two variance reduction techniques, measures the discretization bias on barrier monitoring, and quantifies model risk against the statistical error of the estimator.

The architecture separates three concerns that change independently: the path generator (the model), the payoff functions (the product), and the pricing engine (the numerical method). Only the payoff changes from one product to the next.

## Features

- **GBM path simulator** — vectorized, seeded, returns full trajectories
- **Monte Carlo engine** — price, standard error, confidence interval
- **Variance reduction** — antithetic variates (pair-based SE) and control variate on the discounted underlying
- **Barrier options** — down-and-out, down-and-in, digital barrier
- **Autocallable** — annual observation, early redemption, capital protection barrier, stochastic exit time
- **Model risk analysis** — sensitivity to volatility and rates, benchmarked against Monte Carlo error
- **Automated test suite** (pytest) — martingale property, Black-Scholes convergence, barrier parity, vega sign
- **Interactive dashboard** (Streamlit + Plotly) — three tabs: barriers, autocall, model risk

## Project structure

    Structured-Product-Lab/
    ├── paths.py              # GBM path simulation (standard + antithetic)
    ├── mc_engine.py          # Pricing engine: mean, SE, CI, control variate
    ├── barrier.py            # Barrier option payoffs
    ├── autocallable.py       # Autocall pricer with early redemption
    ├── dashboard.py          # Interactive Streamlit dashboard
    ├── tests/
    │   └── test_structured.py
    ├── requirements.txt
    └── README.md

## The math

### Monte Carlo pricing

Price = e^(-rT) x (1/N) x sum of Payoff(path_i)

The law of large numbers guarantees convergence; the central limit theorem gives the rate: the standard error decreases as sigma_payoff / sqrt(N). Dividing the error by 10 requires 100 times more simulations.

### Path discretization

S(t+dt) = S(t) x exp[(r - sigma^2/2) x dt + sigma x sqrt(dt) x Z]

Applied recursively with an independent Z at each step. The drift scales with dt, the volatility with sqrt(dt) — because volatility is a standard deviation, and only variances add over time. Summing the exponents recovers the single-step law of S_T, so the multi-step simulation is consistent with Black-Scholes.

### Variance reduction

**Antithetic variates**: pair each Z with -Z. Both are valid draws, and their payoffs are negatively correlated, so the pair average is more stable than two independent draws. Critically, the standard error must be computed on the pairs — the naive formula on raw payoffs sees no reduction and hides the entire benefit.

**Control variate**: the discounted underlying has a known exact expectation, E[e^(-rT) x S_T] = S0. The simulated deviation from that reference reveals the sampling error of the run, which is then subtracted from the payoff, scaled by a regression coefficient. That coefficient converges to the option's delta — the same sensitivity, estimated statistically rather than analytically.

### Barrier options

The path decides whether the contract pays; the strike decides how much.

Payoff = max(S_T - K, 0) x indicator{min(S_t) > B}

An *out* option starts alive and dies if the barrier is touched. An *in* option starts dormant and only activates if it is touched. The switch is irreversible, and in + out = vanilla.

### Autocallable

At each annual observation, if the underlying is above the call level, the product redeems early with capital plus accrued coupons. If it survives to maturity, the capital is returned in full above the protection barrier, or pro rata below it.

Economically, the investor is long a deposit and **short** two options: a down-and-in put at the protection barrier, and the early redemption feature. The coupon is the premium of what is sold, not a yield.

## Key results

| Result | Value |
|---|---|
| MC vs Black-Scholes (vanilla call) | 10.4829 vs 10.4506 — BS inside the 95% CI |
| Antithetic variates | SE / 1.42, equivalent to 2x more simulations |
| Control variate | SE / 2.63, equivalent to 6.9x — implied correlation 0.93 |
| Optimal control coefficient c* | 0.676, against a Black-Scholes delta of 0.637 |
| Barrier parity (in + out = vanilla) | matched to 1e-8 |
| Discretization bias (down-and-out) | +2.45% at 12 steps, +0.78% at 252 steps |
| Autocall (4y, 8% coupon, 65% barrier) | 969.28 EUR per 1000 — 3.1% issuer margin |
| Average life vs stated maturity | 1.99 years against a 4-year product |
| Autocall vega | approximately -6 EUR per volatility point |
| **Monte Carlo SE vs model risk** | **0.36 EUR vs 49 EUR — a factor of 137** |

The last line is the central finding: the confidence interval measures the precision of the estimator, not the accuracy of the model. Adding simulations is worthless until the volatility is properly calibrated.

## Running the project

Install dependencies: pip3 install -r requirements.txt

Run the tests: pytest

Launch the dashboard: streamlit run dashboard.py

## Tech stack

Python, NumPy, SciPy, Plotly, Streamlit, pytest

## What I learned

- Why Monte Carlo is the only option for path-dependent payoffs, and why it is useless for a vanilla call — where it serves only as a cross-validation of the engine
- That variance reduction requires rethinking the estimator, not just the simulation: antithetic variates change what counts as an independent observation
- That the optimal control variate coefficient is the option's delta, connecting the Greeks of my first project to the numerical methods of this one
- The difference between a random error and a systematic bias: more simulations converge precisely toward a wrong value if the discretization is too coarse
- That an autocallable is a short volatility position dressed as a fixed-income product — the coupon is the premium of a down-and-in put the investor sells without always realizing it
- That model risk dominates statistical error by two orders of magnitude, which reorders every priority in pricing work

## Next steps

- Add a continuous dividend yield to the drift (r - q) — autocallables are typically written on high-dividend underlyings, and ignoring q overstates redemption probabilities
- Calibrate a volatility smile rather than a flat sigma: the protection barrier sits deep out of the money, where implied volatility is materially higher
- Implement a jump-diffusion model (Merton, Bates) — jump risk is structurally unhedgeable and the flat GBM understates tail events
- Brownian bridge correction for the barrier discretization bias
- Solve for the fair coupon by bisection, as in my fixed income project