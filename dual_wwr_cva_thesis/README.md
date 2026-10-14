# Dual Wrong-Way Risk CVA: Hull-White 1F + CIR++ + Gaussian-copula jump channel

Code for an MSc thesis (Tilburg University) on CVA under dual wrong-way risk (WWR) for a
USD 1M-notional 10Y payer IRS against Citigroup (BBB+). Includes the Monte Carlo engine,
PyTorch AAD sensitivities, and the dynamic-hedging analysis built on a 9x9 conditional-AAD grid.

## Model overview

The model prices the unilateral CVA of an interest rate swap under two separate wrong-way risk (WWR) channels, a diffusive spread channel and a jump-to-default channel.

- **Interest rate (Hull-White 1F).** The short rate is $r(t)=x_r(t)+b_r(t)$, where $x_r$ is a zero-mean Ornstein-Uhlenbeck process and the deterministic shift $b_r(t)$ reproduces today's SOFR OIS discount curve. Paths are simulated through forward zero-coupon bond prices under the $T_m$-forward measure.
- **Diffusive credit intensity (CIR++).** $\lambda^{diff}(t)=x_\lambda(t)+b_\lambda(t)$, where $x_\lambda$ follows a CIR process simulated with a Milstein scheme and full truncation, and $b_\lambda(t)$ reproduces the market-implied CDS survival curve. The Brownian motions of the two processes are correlated with $\rho_{r,\lambda}$ (the diffusive spread channel).
- **Jump-to-default channel.** A constant intensity $\lambda^{jump}$, calibrated to Citigroup's 2008-2009 stress-period CDS spread, defines the deterministic margin $u_\tau(t)=Q^{jump}(t)$. A Gaussian copula with parameter $\rho_{copula}$ couples it to each path's exposure rank $u_E=\mathrm{rank}(E)/(M+1)$ and reweights the simulated paths. The copula redistributes existing default probability toward high-exposure paths and does not create new default probability.
- **CVA.** With $M$ paths and $n$ time steps, the CVA is
$$
CVA = -(1-R)P(0,T)\sum_{k=1}^{n}\sum_{j=1}^{M}\frac{E_j(t_k)}{P^{(j)}(t_k,T)}w_j(t_k)\Delta Q_j^{diff}(t_k)
$$
- **Sensitivities (AAD).** A single backward pass in PyTorch yields Market Delta (SOFR and CDS curves), Model Vega, WWR Rho and, with a central difference over AAD, the rate/credit cross-gamma. QuantLib supplies a frozen base curve and a differentiable bump layer (`bump_interp`) carries the gradient.
- **Dynamic hedging.** The CVA is re-hedged monthly (131 dates). At each date a $9\times9$ grid of conditional-AAD valuations is fitted by a quadratic proxy in $(x_r,x_\lambda)$ and evaluated at every path's own state. A naive hedge sized under independence is compared with a hedge sized under the true WWR scenario, using hedge effectiveness, unexplained P&L variance, VaR/ES and the FRTB P&L attribution tests.

## Key results

Uncollateralized CVA of a 10Y payer IRS (USD 1M notional) against Citigroup, 10,000 paths, weekly grid.

| Quantity | Result | Thesis |
|---|---|---|
| Independent CVA | 1528.80 USD (0.1529% of notional) | Table 7.1 |
| Diffusive channel only ($\rho_{r,\lambda}=0.9$) | +2.03% | Table 7.1 |
| Jump-to-default channel only ($\rho_{copula}=0.9$) | +114.9% | Table 7.1 |
| Dual WWR ($\rho_{r,\lambda}=\rho_{copula}=0.9$) | +122.2% | Table 7.1 |
| 10Y Rate Delta, Dual-strong vs. Independent | +86% (19.46 to 36.12 per 1bp) | Table 7.2 |
| Credit Delta profile | hump-shaped, peaking at the 3Y-5Y nodes | Table 7.3 |
| Hedge effectiveness at $(0.8,0.8)$ | naive 84.16% vs. adjusted 96.35% | Table 7.8 |
| Unexplained P&L variance ratio at $(0.8,0.8)$ | 15.8% to 3.6% (-77%) | Table 7.8 |
| ES$_{97.5}$ reduction at $(0.8,0.8)$ | naive 59.9% vs. adjusted 77.9% | Table 7.8 |

The naive hedge stays below the 20% PLA variance-test threshold in every scenario, so the finding is a loss of margin and not a regulatory breach.

The values in this table are reproduced by `1_main_notebook/dual_wwr_cva_citi_calibration.ipynb` (roughly 15 to 25 minutes on a laptop CPU).

## Layout

| Folder | Content |
|---|---|
| `1_main_notebook/` | `dual_wwr_cva_citi_calibration.ipynb`: final Citigroup run (calibration, CVA, WWR decomposition, AAD report, cross-gamma, collateral and netting, hedge effectiveness and FRTB PLA) |
| `2_engine_notebook/` | `dual_wwr_cva_engine.ipynb`: simulation and conditional-AAD engine (definitions only, no data or results). The main notebook and the grid builder execute all of its code cells via `nbformat`, so it must stay here |
| `3_grids_9x9/` | Precomputed Citigroup conditional-AAD grids (131 monthly dates x 9x9 percentile pilots of (x_r, x_lambda)), one `.npz` per WWR scenario, plus `build_citi_joint_grid.py`, the script that generates them |
| `4_market_data/` | `sofr/`: public SOFR history. `licensed_not_for_upload/`: vendor-licensed CDS and OIS-curve quotes, excluded from git (see `4_market_data/README.md`) |
| `results/` | Figures and result `.npz` files written by the notebooks |

## Running

1. `pip install -r requirements.txt` (versions pinned to the environment the results were
   reproduced in: Python 3.12.7, macOS, CPU)
2. Place the licensed data files in `4_market_data/licensed_not_for_upload/` (see its README).
3. Open `1_main_notebook/dual_wwr_cva_citi_calibration.ipynb` and run from that folder
   (all paths are relative to the notebook's own directory).
4. To regenerate a grid instead of using the shipped one:
   `cd 3_grids_9x9 && python build_citi_joint_grid.py wwr` (about 3-4 hours per scenario).
   Pilot percentiles are taken from 10,000 outer paths; each of the 81 nodes per date is
   re-simulated with 1,000 fresh sub-paths.

## Notes

- The evaluation date is 2026-08-11 and is set inside the engine functions `get_ql_br_t` and `get_ql_blambda_t`.
- The licensed files are not distributed. The calibration cells of the main notebook and the grid
  builder read them directly, so the three files listed in `4_market_data/README.md` are needed to
  run them end to end.
