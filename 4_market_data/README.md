# Market data

## `sofr/` (included)

| File | Content |
|---|---|
| `SOFR 2023 -2026.xlsx` | Daily SOFR history (Federal Reserve Bank). Columns `Date`, `SOFR` (in %) |

## `licensed_not_for_upload/` (excluded from git via `.gitignore`)

These files come from a licensed vendor (Refinitiv; CDS RIC `C5YUSAX=R`) and are not
redistributed. To reproduce the results, place the following files in this folder:

| File | Columns | Content |
|---|---|---|
| `SOFR 8_11.xlsx` | sheet `Curve1 - Realtime`, columns `Maturity`, `Bid`, `Yield`, `Ask`, `BenchmarkSpread`, `NetChange` | SOFR OIS curve quotes as of 2026-08-11 for 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y (two header rows, quotes in %) |
| `Citigroup 5Y CDS 2024 - 2026.xlsx` | `Date`, `C5YUSAX=R (PAR_MID1)` | Citigroup 5Y CDS par spread, 2024-08-12 to 2026-08-10 (CIR++ MLE calibration) |
| `5Y citygroup CDS 2007 - 2015.xlsx` | `Date`, `C5YUSAX=R (PAR_MID1)` | Citigroup 5Y CDS, 2007-2015 (the 2008-2009 average sets the jump-to-default intensity) |

CDS spreads are in basis points; any equivalent daily 5Y series with the same columns works.
