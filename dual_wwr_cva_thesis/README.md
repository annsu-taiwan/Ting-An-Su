# Dual Wrong-Way Risk CVA: Hull-White 1F + CIR++ + Gaussian-copula jump channel

Code for an MSc thesis (Tilburg University) on CVA under dual wrong-way risk (WWR) for a
USD 1M-notional 10Y payer IRS against Citigroup (BBB+). Includes the Monte Carlo engine,
PyTorch AAD sensitivities, and the dynamic-hedging analysis built on a 9x9 conditional-AAD grid.

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
