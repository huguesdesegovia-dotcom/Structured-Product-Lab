import streamlit as st
import numpy as np
import plotly.graph_objects as go

from paths import simulate_gbm_paths
from mc_engine import monte_carlo_price
from barrier import payoff_down_and_out_call, payoff_down_and_in_call
from autocallable import price_autocall

st.set_page_config(page_title="Structured Product Lab", page_icon="🧱", layout="wide")

st.title("🧱 Structured Product Lab")
st.markdown("Monte Carlo pricing of barrier options and autocallables, with model risk analysis.")

st.sidebar.header("⚙️ Market parameters")
S0 = st.sidebar.slider("Spot (S0)", 50, 150, 100)
r = st.sidebar.slider("Risk-free rate", 0.0, 0.10, 0.05)
sigma = st.sidebar.slider("Volatility", 0.05, 0.60, 0.20)
n_sim = st.sidebar.select_slider("Simulations", [10_000, 50_000, 100_000], value=50_000)

tab1, tab2, tab3 = st.tabs(["📉 Barrier options", "🧱 Autocallable", "⚠️ Model risk"])

with tab1:
    col_p, col_c = st.columns([1, 2])

    with col_p:
        K = st.slider("Strike (K)", 50, 150, 100)
        B = st.slider("Barrier", 50, 100, 85)
        n_steps = st.select_slider("Observations per year", [12, 52, 252, 1000], value=252)

    p = simulate_gbm_paths(S0, 1, r, sigma, n_steps, n_sim, seed=42)
    do, se_do, _ = monte_carlo_price(payoff_down_and_out_call(p, K, B), r, 1)
    di, _, _ = monte_carlo_price(payoff_down_and_in_call(p, K, B), r, 1)
    van, _, _ = monte_carlo_price(np.maximum(p[:, -1] - K, 0), r, 1)

    with col_p:
        st.metric("Down-and-out", f"{do:.3f} €")
        st.metric("Down-and-in", f"{di:.3f} €")
        st.metric("Vanilla (= sum)", f"{van:.3f} €")
        st.caption(f"P(barrier touched) = {(p.min(axis=1) <= B).mean():.1%}")

    with col_c:
        sample = p[:200]
        fig = go.Figure()
        for i in range(200):
            fig.add_trace(go.Scatter(y=sample[i], mode="lines", showlegend=False,
                                     line=dict(width=0.5, color="#4DA3FF"), opacity=0.3))
        fig.add_hline(y=B, line_dash="dash", line_color="#FF6B6B", annotation_text="Barrier")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B1F3A",
                          plot_bgcolor="#0B1F3A", height=450,
                          xaxis_title="Time step", yaxis_title="Spot",
                          title="200 simulated paths")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_p, col_c = st.columns([1, 2])

    with col_p:
        coupon = st.slider("Annual coupon", 0.0, 0.25, 0.08)
        call_lvl = st.slider("Call level (% of S0)", 0.80, 1.20, 1.00)
        prot_lvl = st.slider("Protection barrier (% of S0)", 0.40, 0.90, 0.65)
        n_obs = st.slider("Years", 1, 6, 4)

    s = price_autocall(S0=S0, T=n_obs, r=r, sigma=sigma, n_obs=n_obs,
                       coupon_rate=coupon, call_level=call_lvl,
                       barrier_level=prot_lvl, n_simulations=n_sim, seed=42)

    with col_p:
        st.metric("Price", f"{s['prix']:.2f} € / 1000")
        st.metric("Issuer margin", f"{1000 - s['prix']:.2f} €")
        st.metric("P(capital loss)", f"{s['prob_perte_capital']:.1%}")
        st.metric("Average maturity", f"{s['maturite_moyenne']:.2f} yrs")

    with col_c:
        vols = np.arange(0.10, 0.55, 0.05)
        prices = [price_autocall(S0=S0, T=n_obs, r=r, sigma=v, n_obs=n_obs,
                                 coupon_rate=coupon, call_level=call_lvl,
                                 barrier_level=prot_lvl, n_simulations=20_000,
                                 seed=42)["prix"] for v in vols]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=vols * 100, y=prices, mode="lines+markers",
                                  line=dict(color="#4DA3FF", width=3)))
        fig2.add_hline(y=1000, line_dash="dash", line_color="#FFB84D",
                       annotation_text="Issue price")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0B1F3A",
                           plot_bgcolor="#0B1F3A", height=450,
                           xaxis_title="Volatility (%)", yaxis_title="Price (€)",
                           title="Autocall value vs volatility — investor is short vega")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Statistical error vs model risk")
    st.markdown(
        "The confidence interval measures the **precision of the estimator**, "
        "not the **accuracy of the model**. Below, both are put side by side."
    )

    base = dict(S0=S0, T=4, r=r, n_obs=4, coupon_rate=0.08,
                call_level=1.00, barrier_level=0.65,
                n_simulations=n_sim, seed=42)
    ref = price_autocall(sigma=sigma, **base)

    rows = []
    for label, kw in [
        ("σ − 2 pts", dict(sigma=sigma - 0.02)),
        ("σ + 2 pts", dict(sigma=sigma + 0.02)),
        ("σ + 8 pts (put skew)", dict(sigma=sigma + 0.08)),
        ("r − 2 pts", dict(sigma=sigma, r=max(r - 0.02, 0.0))),
        ("r + 2 pts", dict(sigma=sigma, r=r + 0.02)),
    ]:
        pr = price_autocall(**{**base, **kw})["prix"]
        rows.append({"Scenario": label, "Price": f"{pr:.2f}", "Gap": f"{pr - ref['prix']:+.2f}"})

    st.table(rows)