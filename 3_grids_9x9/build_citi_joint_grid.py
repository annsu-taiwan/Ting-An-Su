# Offline builder for the Citigroup 9x9 conditional-AAD grids (output: ../3_grids_9x9/*.npz).
# Usage (run from this folder):  python build_citi_joint_grid.py <scenario>
#   scenario in {independent, wwr, wwr_strong, mixed_a, mixed_b, diff_04, diff_08, jump_04, jump_08}
# Roughly 3-4 hours per scenario on CPU; resumes from the .npz checkpoint if interrupted.
import matplotlib
matplotlib.use("Agg")
import sys
import time
import numpy as np
import pandas as pd
import QuantLib as ql
import torch
import nbformat
from scipy.stats import ncx2
from scipy.optimize import minimize

NB_PATH = "../2_engine_notebook/dual_wwr_cva_engine.ipynb"

SCENARIO = sys.argv[1]  # e.g. "independent", "wwr", "wwr_strong", "mixed_a", "mixed_b"
N_PILOT_PATHS = 10000
N_AAD_PATHS = 1000
REBAL_STEP = 4  # monthly (weekly grid at steps_per_year=52 -> every 4th week)
PCTS = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

SCEN_MAP = {
    "independent": (0.0, 0.0),
    "wwr": (0.4, 0.4),
    "wwr_strong": (0.8, 0.8),
    "mixed_a": (0.4, 0.8),
    "mixed_b": (0.8, 0.4),
    "diff_04": (0.4, 0.0),
    "diff_08": (0.8, 0.0),
    "jump_04": (0.0, 0.4),
    "jump_08": (0.0, 0.8),
}
RHO_R_LAMBDA, RHO_COPULA = SCEN_MAP[SCENARIO]
OUT_NAMES = {"mixed_a": "mixed_(04,08)", "mixed_b": "mixed_(08,04)"}  # file label = (rho_r_lambda, rho_copula)
OUT_PATH = f"dual_wwr_joint_grid_citi_{OUT_NAMES.get(SCENARIO, SCENARIO)}_monthly.npz"

print(f"=== Citigroup joint grid builder: scenario={SCENARIO} (rho_r_lambda={RHO_R_LAMBDA}, rho_copula={RHO_COPULA}) ===", flush=True)

# ---------------------------------------------------------------------------
# STEP 1: load the simulation and conditional-AAD engine (every code cell of
# the engine notebook).
# ---------------------------------------------------------------------------
print("Loading simulation engine...", flush=True)
nb = nbformat.read(NB_PATH, as_version=4)
ns = {}
t0 = time.time()
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    exec(compile(c.source, f"<engine cell {i}>", "exec"), ns)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

# Override the inherited generic 300bps LAMBDA_JUMP_ASSUMED with Citigroup's
# own 2008-2009 financial-crisis CDS spread average, for consistency with the
# rest of the Citigroup-specific calibration. run_aad_sensitivities_conditional
# falls back to the global LAMBDA_JUMP_ASSUMED (looked up in ns, its __globals__)
# whenever lambda_jump_assumed isn't passed explicitly, so overwriting ns here
# is sufficient -- Python resolves globals at call time, not def time.
df_crisis_lj = pd.read_excel("../4_market_data/licensed_not_for_upload/5Y citygroup CDS 2007 - 2015.xlsx")
df_crisis_lj.columns = ["Date", "Spread"]
df_crisis_lj["Spread"] = pd.to_numeric(df_crisis_lj["Spread"], errors="coerce")
df_crisis_lj = df_crisis_lj.dropna(subset=["Date", "Spread"]).sort_values("Date")
crisis_window_lj = df_crisis_lj[(df_crisis_lj["Date"] >= "2008-01-01") & (df_crisis_lj["Date"] <= "2009-12-31")]
S_CDS_ASSUMED_NEW = crisis_window_lj["Spread"].mean() / 10000
LAMBDA_JUMP_ASSUMED_NEW = S_CDS_ASSUMED_NEW / (1.0 - ns["recovery_rate"])
ns["S_CDS_ASSUMED"] = S_CDS_ASSUMED_NEW
ns["LAMBDA_JUMP_ASSUMED"] = LAMBDA_JUMP_ASSUMED_NEW
print(f"  Overrode LAMBDA_JUMP_ASSUMED: {S_CDS_ASSUMED_NEW*10000:.2f}bps (Citigroup 2008-2009 crisis avg) "
      f"-> lambda_jump={LAMBDA_JUMP_ASSUMED_NEW:.6f}  (was 300bps/0.05)", flush=True)

run_aad_sensitivities_conditional = ns["run_aad_sensitivities_conditional"]
build_time_grid = ns["build_time_grid"]
generate_dual_wwr_randoms = ns["generate_dual_wwr_randoms"]
simulate_forward_zcb_paths = ns["simulate_forward_zcb_paths"]
simulate_hazard_paths = ns["simulate_hazard_paths"]
T_pay = ns["T_pay"]
notional = ns["notional"]
swap_config = ns["swap_config"]
max_years = ns["max_years"]

# ---------------------------------------------------------------------------
# STEP 2: rebuild rates (new SOFR data) and credit (Citigroup, Exact MLE, no
# Q-measure refit) -- identical methodology to citi_final_calibration.py.
# ---------------------------------------------------------------------------
print("Rebuilding Citigroup calibration...", flush=True)

df_sofr_new = pd.read_excel("../4_market_data/sofr/SOFR 2023 -2026.xlsx")
df_sofr_new.columns = ["Date", "SOFR"]
df_sofr_new["SOFR"] = pd.to_numeric(df_sofr_new["SOFR"], errors="coerce") / 100
df_sofr_new = df_sofr_new.dropna(subset=["Date", "SOFR"]).sort_values("Date")
sofr_series_new = df_sofr_new.set_index("Date")["SOFR"]
a_new, sigma_r_new = ns["calibrate_hw"](sofr_series_new)

df_ois = pd.read_excel("../4_market_data/licensed_not_for_upload/SOFR 8_11.xlsx", sheet_name="Curve1 - Realtime", header=None)
df_ois.columns = ["Maturity", "Bid", "Yield", "Ask", "BenchmarkSpread", "NetChange"]
df_ois = df_ois.iloc[2:].copy()
df_ois["Bid"] = pd.to_numeric(df_ois["Bid"], errors="coerce")
ois_map = dict(zip(df_ois["Maturity"], df_ois["Bid"]))
sofr_market_quotes_new = {t: ois_map[t] / 100 for t in ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]}

_, _, sofr_yield_curve_new_prelim = ns["get_ql_br_t"](sofr_market_quotes_new, a_new, sigma_r_new)
yield_curve_handle_new = ql.RelinkableYieldTermStructureHandle(sofr_yield_curve_new_prelim)
index_new = ql.Sofr(yield_curve_handle_new)
helpers_new = []
for expiry, tenor, vol in ns["swaption_vols"]:
    vol_quote = ql.QuoteHandle(ql.SimpleQuote(vol / 10000.0))
    helper = ql.SwaptionHelper(ql.Period(int(expiry * 12), ql.Months), ql.Period(int(tenor), ql.Years),
        vol_quote, index_new, ql.Period(1, ql.Years), ql.Actual360(), ql.Actual360(),
        yield_curve_handle_new, ql.BlackCalibrationHelper.RelativePriceError, ql.nullDouble(), 1.0, 1)
    helpers_new.append(helper)
model_new = ql.HullWhite(yield_curve_handle_new, a_new, sigma_r_new)
engine_new = ql.JamshidianSwaptionEngine(model_new)
for h in helpers_new:
    h.setPricingEngine(engine_new)
model_new.calibrate(helpers_new, ql.LevenbergMarquardt(), ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8))
a_calibrated = model_new.params()[0]
sigma_r_calibrated = model_new.params()[1]
t_axis_cal_new, br_curve_cal_new, sofr_yield_curve_calibrated = ns["get_ql_br_t"](
    sofr_market_quotes_new, a_calibrated, sigma_r_calibrated)

df_citi = pd.read_excel("../4_market_data/licensed_not_for_upload/Citigroup 5Y CDS 2024 - 2026.xlsx")
df_citi.columns = ["Date", "Spread5Y"]
df_citi["Spread5Y"] = pd.to_numeric(df_citi["Spread5Y"], errors="coerce") / 10000
recovery_rate = 0.4
df_citi["lambda"] = df_citi["Spread5Y"] / (1 - recovery_rate)
df_citi = df_citi.dropna(subset=["Date", "lambda"]).sort_values("Date")
lambda_series_citi = df_citi.set_index("Date")["lambda"]


def calibrate_cir_mle(lam_series, dt=1 / 252):
    lam = lam_series[lam_series > 1e-8].values
    lam_t, lam_tp1 = lam[:-1], lam[1:]

    def neg_loglik(params):
        kappa, lambda_bar, sigma = params
        if kappa <= 0 or lambda_bar <= 0 or sigma <= 0:
            return 1e12
        c = sigma ** 2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
        if c <= 0:
            return 1e12
        df = 4 * kappa * lambda_bar / sigma ** 2
        nc = 4 * kappa * np.exp(-kappa * dt) * lam_t / (sigma ** 2 * (1 - np.exp(-kappa * dt)))
        x = lam_tp1 / c
        pdf_vals = np.clip(ncx2.pdf(x, df, nc) / c, 1e-300, None)
        return -np.sum(np.log(pdf_vals))

    k0, lb0, s0 = ns["calibrate_cir_regression"](lam_series, dt)
    init = [max(k0, 0.1), max(lb0, 1e-4), max(s0, 1e-4)]
    opt_res = minimize(neg_loglik, init, method="Nelder-Mead",
                        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000})
    return opt_res.x[0], opt_res.x[1], opt_res.x[2]


kappa_mle, lambda_bar_mle, sigma_mle_raw = calibrate_cir_mle(lambda_series_citi)
sigma_mle_final = max(sigma_mle_raw, 0.02)

cds_market_data = {
    "6M": 24.550 / 10000, "1Y": 26.757 / 10000, "2Y": 32.761 / 10000,
    "3Y": 37.253 / 10000, "5Y": 53.331 / 10000, "7Y": 69.857 / 10000, "10Y": 90.838 / 10000,
}
b_lambda_prelim = ns["get_ql_blambda_t"](
    cds_market_data, sofr_yield_curve_calibrated, kappa_mle, lambda_bar_mle, sigma_mle_final,
    x0=lambda_bar_mle)
x_lambda_0 = b_lambda_prelim["fm_curve"](b_lambda_prelim["t_axis_l"])[0]
b_lambda_calibration = ns["get_ql_blambda_t"](
    cds_market_data, sofr_yield_curve_calibrated, kappa_mle, lambda_bar_mle, sigma_mle_final,
    x0=x_lambda_0)
bl_func_calibration = b_lambda_calibration["bl_func"]
fm_curve_calibration = b_lambda_calibration["fm_curve"]

kappa_calibration = kappa_mle
lambda_bar_calibration = lambda_bar_mle
sigma_lambda_calibration = sigma_mle_final

print(f"  rates: a_calibrated={a_calibrated:.6f}  sigma_r_calibrated={sigma_r_calibrated:.6f}")
print(f"  credit: kappa={kappa_calibration:.4f}  lambda_bar={lambda_bar_calibration:.6f}  "
      f"sigma_lambda={sigma_lambda_calibration:.6f}  x_lambda_0={x_lambda_0:.6f}", flush=True)

# ---------------------------------------------------------------------------
# STEP 3: reconstruct the 10,000 outer MC paths' own (x_r, x_lambda) states
# (NumPy engine, Citigroup-calibrated) to derive the per-date percentile
# pilot grids.
# ---------------------------------------------------------------------------
n_paths_run = N_PILOT_PATHS
seed_run = 1234
t_axis, dt_grid = build_time_grid(max_years, steps_per_year=52)
T_m = T_pay[-1]


def reconstruct_paths_joint(rho_r_lambda):
    randoms_j = generate_dual_wwr_randoms(t_axis, n_paths_run, rho_r_lambda=rho_r_lambda, seed=seed_run)
    dW_r_j, dW_lambda_j = randoms_j["dW_r"], randoms_j["dW_lambda"]
    n_paths_j = dW_r_j.shape[1]
    x_r_path_j = np.zeros((len(t_axis), n_paths_j))
    for k in range(len(t_axis) - 1):
        t_prev = t_axis[k]
        sigma_P_Tm_k = (sigma_r_calibrated / a_calibrated) * (1.0 - np.exp(-a_calibrated * (T_m - t_prev)))
        drift = (-a_calibrated * x_r_path_j[k, :] - sigma_r_calibrated * sigma_P_Tm_k) * dt_grid[k]
        x_r_path_j[k + 1, :] = x_r_path_j[k, :] + drift + sigma_r_calibrated * dW_r_j[k, :]
    lambda_diff_j = simulate_hazard_paths(
        kappa_calibration, lambda_bar_calibration, sigma_lambda_calibration, x_lambda_0,
        a_calibrated, sigma_r_calibrated, rho_r_lambda, bl_func_calibration, dW_lambda_j, t_axis, T_m,
    )
    h_j = np.sqrt(kappa_calibration ** 2 + 2.0 * sigma_lambda_calibration ** 2)
    exp_ht_j = np.exp(h_j * t_axis)
    denom_j = 2.0 * h_j + (kappa_calibration + h_j) * (exp_ht_j - 1.0)
    fCIR_j = (2.0 * kappa_calibration * lambda_bar_calibration * (exp_ht_j - 1.0)) / denom_j + \
        x_lambda_0 * (4.0 * h_j ** 2 * exp_ht_j) / denom_j ** 2
    fM_j = fm_curve_calibration(t_axis)
    b_lambda_j = fM_j - fCIR_j
    x_lambda_path_j = np.clip(lambda_diff_j - b_lambda_j[:, np.newaxis], 0.0, None)
    return x_r_path_j, x_lambda_path_j


print("Reconstructing outer paths for pilot-grid percentiles...", flush=True)
xr_path, xl_path = reconstruct_paths_joint(RHO_R_LAMBDA)

grid_idx = np.arange(0, len(t_axis), REBAL_STEP)
n_dates = len(grid_idx)
grid_dates = t_axis[grid_idx]
print(f"  {n_dates} monthly rebalancing dates, t in [{grid_dates[0]:.3f}, {grid_dates[-1]:.3f}]", flush=True)

# ---------------------------------------------------------------------------
# STEP 4: build (or resume) the 9x9 joint conditional-AAD grid per date.
# ---------------------------------------------------------------------------
import os

if os.path.exists(OUT_PATH):
    ckpt = np.load(OUT_PATH, allow_pickle=True)
    cva_grid = ckpt["cva_grid"]
    delta_R_grid = ckpt["delta_R_grid"]
    delta_L_grid = ckpt["delta_L_grid"]
    vega_grid = ckpt["vega_grid"]
    xr_pilot_grid = ckpt["xr_pilot_grid"]
    xl_pilot_grid = ckpt["xl_pilot_grid"]
    done_mask = ckpt["done_mask"]
    print(f"Resuming from checkpoint: {done_mask.sum()}/{n_dates} dates already done", flush=True)
else:
    cva_grid = np.zeros((n_dates, 9, 9))
    delta_R_grid = np.zeros((n_dates, 9, 9))
    delta_L_grid = np.zeros((n_dates, 9, 9))
    vega_grid = np.zeros((n_dates, 9, 9))
    xr_pilot_grid = np.zeros((n_dates, 9))
    xl_pilot_grid = np.zeros((n_dates, 9))
    done_mask = np.zeros(n_dates, dtype=bool)


def save_checkpoint():
    np.savez(
        OUT_PATH,
        cva_grid=cva_grid, delta_R_grid=delta_R_grid, delta_L_grid=delta_L_grid, vega_grid=vega_grid,
        xr_pilot_grid=xr_pilot_grid, xl_pilot_grid=xl_pilot_grid, done_mask=done_mask,
        t_axis_dates=grid_dates, rho_r_lambda=RHO_R_LAMBDA, rho_copula=RHO_COPULA,
    )


t_start_all = time.time()
for di in range(n_dates):
    if done_mask[di]:
        continue
    k_full = grid_idx[di]
    t_k = t_axis[k_full]

    xr_vals = xr_path[k_full, :]
    xl_vals = xl_path[k_full, :]
    xr_pilot = np.percentile(xr_vals, PCTS)
    xl_pilot = np.percentile(xl_vals, PCTS)
    xr_pilot_grid[di, :] = xr_pilot
    xl_pilot_grid[di, :] = xl_pilot

    t_date0 = time.time()
    for ii in range(9):
        for jj in range(9):
            res = run_aad_sensitivities_conditional(
                t_start=t_k, x_r_start=float(xr_pilot[ii]), x_lambda_start=float(max(xl_pilot[jj], 0.0)),
                yield_curve=sofr_yield_curve_calibrated, fm_curve_calibration=fm_curve_calibration,
                sofr_market_quotes=sofr_market_quotes_new, cds_market_data=cds_market_data,
                a0=a_calibrated, sigma_r0=sigma_r_calibrated,
                kappa0=kappa_calibration, lambda_bar0=lambda_bar_calibration,
                sigma_lambda0=sigma_lambda_calibration,
                rho_r_lambda0=RHO_R_LAMBDA, rho_copula0=RHO_COPULA,
                payment_dates=T_pay, notional=notional, position_sign=swap_config["delta"],
                recovery_rate=recovery_rate, T_max=max_years, n_paths=N_AAD_PATHS, seed=1234,
                steps_per_year=26,
            )
            cva_grid[di, ii, jj] = res["cva"]
            delta_R_grid[di, ii, jj] = res["bump_R"].grad.sum().item()
            delta_L_grid[di, ii, jj] = res["bump_lambda"].grad.sum().item()
            vega_grid[di, ii, jj] = res["sigma_r"].grad.item()

    done_mask[di] = True
    save_checkpoint()
    elapsed = time.time() - t_start_all
    eta = elapsed / (di + 1) * (n_dates - di - 1)
    print(f"  [{SCENARIO}] date {di+1}/{n_dates} (t={t_k:.3f}) done in {time.time()-t_date0:.1f}s "
          f"| elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min", flush=True)

print(f"\n=== {SCENARIO} grid complete: {OUT_PATH} ===", flush=True)
