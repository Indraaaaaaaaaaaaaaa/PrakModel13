import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

st.set_page_config(page_title="TA-13 Lotka–Volterra (COVID-19)", layout="wide")

st.title("TA-13 — Lotka–Volterra (Predator–Prey) dengan Dataset COVID-19")

st.markdown("""
**Pemetaan variabel (agar cocok untuk Lotka–Volterra):**
- **x(t)** = kasus baru harian (new confirmed) → proxy *prey*
- **y(t)** = removal harian = sembuh + meninggal (new recovered + new deaths) → proxy *predator*

Catatan: Ini pendekatan interaksi 2 variabel untuk kebutuhan tugas.
""")

DEFAULT_PATH = "covid_19_clean_complete.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def simulate_lv(params, t_eval, x0, y0):
    alpha, beta, delta, gamma = params
    alpha = max(alpha, 1e-9)
    beta = max(beta, 1e-9)
    delta = max(delta, 1e-9)
    gamma = max(gamma, 1e-9)

    sol = solve_ivp(
        fun=lambda tt, zz: lotka_volterra(tt, zz, alpha, beta, delta, gamma),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=[x0, y0],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )
    xs = np.clip(sol.y[0], 0, None)
    ys = np.clip(sol.y[1], 0, None)
    return xs, ys


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_daily_series(df: pd.DataFrame, country: str) -> pd.DataFrame:
    ts = (
        df[df["Country/Region"] == country]
        .groupby("Date", as_index=False)[["Confirmed", "Deaths", "Recovered", "Active"]]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    for c in ["Confirmed", "Deaths", "Recovered", "Active"]:
        ts[c] = ts[c].fillna(0)

    ts["new_confirmed"] = ts["Confirmed"].diff().fillna(0)
    ts["new_deaths"] = ts["Deaths"].diff().fillna(0)
    ts["new_recovered"] = ts["Recovered"].diff().fillna(0)

    for c in ["new_confirmed", "new_deaths", "new_recovered"]:
        ts[c] = ts[c].clip(lower=0)

    ts["x_raw"] = ts["new_confirmed"]
    ts["y_raw"] = ts["new_recovered"] + ts["new_deaths"]

    ts = ts[(ts["x_raw"] > 0) | (ts["y_raw"] > 0)].reset_index(drop=True)
    return ts


# Sidebar
with st.sidebar:
    st.header("Input Data")
    uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"])
    st.caption("Jika tidak upload, app akan pakai file lokal: covid_19_clean_complete.csv")

    st.header("Pengaturan")
    scale = st.number_input("Skala (bagi x & y)", min_value=1.0, value=1000.0, step=100.0)

    st.header("Parameter LV (manual)")
    alpha = st.slider("alpha", 0.0001, 5.0, 1.0)
    beta = st.slider("beta", 0.0001, 5.0, 0.10)
    delta = st.slider("delta", 0.0001, 5.0, 0.075)
    gamma = st.slider("gamma", 0.0001, 5.0, 1.5)

    st.header("Auto Fit")
    maxiter = st.number_input("maxiter", min_value=50, value=250, step=50)
    do_fit = st.button("Jalankan Auto Fit (Optimisasi RMSE)")


# Load data
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    if not os.path.exists(DEFAULT_PATH):
        st.error("File covid_19_clean_complete.csv tidak ditemukan. Taruh di folder yang sama atau upload lewat sidebar.")
        st.stop()
    df = load_data(DEFAULT_PATH)

# Country picker
countries = sorted(df["Country/Region"].dropna().unique().tolist())
default_idx = countries.index("Indonesia") if "Indonesia" in countries else 0
country = st.selectbox("Pilih negara", countries, index=default_idx)

ts = make_daily_series(df, country)
if len(ts) < 10:
    st.warning("Data terlalu sedikit setelah preprocessing. Coba negara lain.")
    st.stop()

min_d = ts["Date"].min().date()
max_d = ts["Date"].max().date()
d1, d2 = st.date_input("Rentang tanggal", value=(min_d, max_d), min_value=min_d, max_value=max_d)

tsw = ts[(ts["Date"].dt.date >= d1) & (ts["Date"].dt.date <= d2)].reset_index(drop=True)
if len(tsw) < 10:
    st.warning("Rentang tanggal terlalu sempit. Perlebar rentang.")
    st.stop()

t = np.arange(len(tsw), dtype=float)
x_obs = tsw["x_raw"].to_numpy(float) / float(scale)
y_obs = tsw["y_raw"].to_numpy(float) / float(scale)

x0 = float(max(x_obs[0], 1e-6))
y0 = float(max(y_obs[0], 1e-6))

p_best = np.array([alpha, beta, delta, gamma], dtype=float)
status = "Manual"

# Auto fit
if do_fit:
    dx = np.diff(x_obs)
    dy = np.diff(y_obs)

    x_mid = x_obs[:-1]
    y_mid = y_obs[:-1]
    xy = x_mid * y_mid

    A1 = np.column_stack([x_mid, -xy])
    coef1, *_ = np.linalg.lstsq(A1, dx, rcond=None)
    a0, b0 = float(max(coef1[0], 1e-6)), float(max(coef1[1], 1e-6))

    A2 = np.column_stack([xy, -y_mid])
    coef2, *_ = np.linalg.lstsq(A2, dy, rcond=None)
    d0, g0 = float(max(coef2[0], 1e-6)), float(max(coef2[1], 1e-6))

    p0 = np.array([a0, b0, d0, g0], dtype=float)

    def objective(log_params):
        params = np.exp(log_params)
        try:
            xs, ys = simulate_lv(params, t, x0, y0)
            return rmse(xs, x_obs) + rmse(ys, y_obs)
        except Exception:
            return 1e9

    res = minimize(
        objective,
        np.log(p0),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )

    if res.success:
        p_best = np.exp(res.x)
        status = "Auto Fit (success)"
    else:
        status = f"Auto Fit gagal: {res.message} (pakai manual)"

# Simulate
x_sim, y_sim = simulate_lv(p_best, t, x0, y0)

# Layout
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader("Overlay Plot (Data vs Simulasi)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tsw["Date"], x_obs, "--", label="Data x(t)")
    ax.plot(tsw["Date"], x_sim, "-", label="Sim x(t)")
    ax.plot(tsw["Date"], y_obs, "--", label="Data y(t)")
    ax.plot(tsw["Date"], y_sim, "-", label="Sim y(t)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Populasi (scaled)")
    ax.grid(True)
    ax.legend(ncol=2)
    st.pyplot(fig)

with col2:
    st.subheader("Parameter & Metrik")
    st.write("Status:", status)
    st.write({
        "alpha": float(p_best[0]),
        "beta": float(p_best[1]),
        "delta": float(p_best[2]),
        "gamma": float(p_best[3]),
    })
    st.metric("RMSE x", rmse(x_sim, x_obs))
    st.metric("RMSE y", rmse(y_sim, y_obs))

    st.subheader("Phase Portrait")
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    ax2.plot(x_obs, y_obs, "--", label="Data")
    ax2.plot(x_sim, y_sim, "-", label="Sim LV")
    ax2.set_xlabel("x(t) (scaled)")
    ax2.set_ylabel("y(t) (scaled)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

st.caption("Tips: kalau overlay kurang bagus, coba ubah rentang tanggal, skala, atau jalankan Auto Fit.")
