# Probabilistic day-ahead price forecasting for DK1

## Goal

Forecast hourly day-ahead electricity prices for Western Denmark (DK1), with prediction intervals as well as point forecasts. The pre-gate experiment targets the information available before the requested 12:00 CET bidding cutoff on the preceding day.

## Data

The pipeline downloads prices, load forecasts and wind/solar forecasts from ENTSO-E. Archived ECMWF forecasts come through Open-Meteo and provide wind speed, solar radiation and temperature at representative DK1 sites. Raw data, processed datasets and forecast caches stay under the ignored `data/` directory and are not distributed in this repository.

For the pre-gate variant, the weather builder selects the preceding day's midnight UTC run, with an earlier run as a fallback. Later ENTSO-E renewables forecasts are used as a comparison information set, rather than assumed to be available at the bidding cutoff. The code uses local Danish delivery dates and averages subhourly prices to hourly observations. It averages repeated daylight-saving hours and interpolates the missing spring hour.

Run selection enforces the intended weather timing. The cached ENTSO-E data do not preserve publication vintages, so the historical availability of every input cannot be independently established from these files alone.

## Method

- A calendar-based naive benchmark uses historical prices at the same hour.
- LEAR fits Lasso autoregressions for individual delivery hours over rolling training windows.
- LightGBM fits quantile regressions with price lags, forecast load, renewables and calendar/weather features.
- The weather-to-power proxy transforms wind speed through generic turbine power curves, then uses a monthly linear regression on past weather indices and ENTSO-E renewables forecasts. Missing proxy values after its start remain missing.
- Empirical LEAR error quantiles and conformalized LightGBM quantiles form an equal-weight ensemble. Calibration uses earlier out-of-sample errors.
- One-sided Diebold-Mariano tests compare daily losses with a Newey-West variance estimate. Their p-values describe predictive-loss differences, rather than trading profitability.

Forecasting and modelling source files are copied unchanged from the local project. `scripts/verify_saved_results.py` recomputes the reported metrics from saved forecasts without refitting models.

## Results

The cleanup audit reproduced these values from the local cached forecasts. Exact values, evaluation periods and input file hashes are saved in [results/verified_metrics.json](results/verified_metrics.json).

The information-set comparison uses **577 common complete evaluation days** within **2025-01-01 through 2026-08-31**, rather than every calendar day in that interval:

| Ensemble renewables input | MAE (€/MWh) | Pinball loss (€/MWh) | Coverage of nominal 90% interval |
|---|---:|---:|---:|
| ENTSO-E forecast comparison | 16.756 | 5.099 | 91.176% |
| Pre-gate ECMWF proxy and weather | 16.219 | 4.935 | 91.197% |
| No delivery-day renewables forecast | 19.270 | 5.889 | 91.226% |

The naive benchmark has MAE **31.223 €/MWh** on those same days. The pre-gate ensemble reduces MAE by **48.054%**. Its pinball loss is lower than the ENTSO-E comparison ensemble's: one-sided DM statistic **−4.645**, **p = 0.00000170**.

The separate interval-calibration comparison uses **649 common evaluation days** within **2024-10-27 through 2026-08-31** and the ENTSO-E information set. Raw LightGBM nominal 90% intervals cover **75.847%** of observations; conformal adjustment raises coverage to **89.779%**. These are empirical coverages, not a guarantee for future prices.

The rolling weather-to-power proxy has **R² = 0.8805** and MAE **255.629 MW** against the **ENTSO-E renewables forecast**, evaluated within **2024-06-01 through 2026-08-31**. This measures agreement with that forecast, not accuracy against observed generation. The significance result above concerns the downstream price ensemble's pinball loss.

## Limitations

- Results were recomputed from existing forecasts; the cleanup did not refit the entire model history or redownload the data.
- Publication vintages are not retained for all inputs. Strict historical information availability needs a separate source-vintage audit.
- Weather coverage limits the evaluation period, and requiring all compared models to be present removes some days.
- The weather-to-power mapping is a simple proxy fitted to a forecast target, with generic turbine curves.
- Hourly aggregation and daylight-saving interpolation remove some intrahour structure.
- The comparison covers one bidding zone. Fixed LightGBM settings and rolling conformal calibration can perform differently under a new price regime.
- These forecast-error results do not establish profitable trading after execution costs.

## How to run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Put your own ENTSO-E API key in `.env`, then export it before downloading. The application does not load `.env` automatically:

```bash
set -a
source .env
set +a
python -m src.data --zone DK_1 --start 2019-01-01 --end 2026-09-01
python -m src.features --zone DK_1
python -m src.evaluate --zone DK_1
python -m src.evaluate_prob --zone DK_1
python -m src.leakage_check --zone DK_1
python -m src.weather --start 2024-03-14 --end 2026-08-30
python -m src.weather --build
python -m src.pregate --zone DK_1
python -m src.dm --zone DK_1
```

Downloads require access to the source services and an ENTSO-E API key. Training can take substantial time; the default LEAR evaluation refits daily. Commands write intermediate data and results, so preserve any existing experiment outputs before rerunning them.

To audit already-generated forecasts:

```bash
PYTHONPATH=. python scripts/verify_saved_results.py
python test.py
python -m compileall -q src scripts ensemble.py test.py
```

The audit needs `data/processed/DK_1_dataset.parquet`, `data/processed/DK_1_weather.parquet`, the naive forecast and the LEAR/LightGBM forecasts for the base, `_pregate` and `_nores` variants. Those local datasets are excluded from Git. `test.py` is a saved-forecast diagnostic, not a unit-test suite; this project currently has no automated unit-test suite.

## Project status

The day-ahead forecasting study has saved forecasts and a reproduced metric audit. The **imbalance versus day-ahead spread study is in progress**. Related modules include `imbalance`, `revisions`, `infosets`, `spread_models` and `backtest_spread`; their saved reports are exploratory and are not presented here as a validated trading result.

## Data credits and author

Electricity data: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). Weather data: [Open-Meteo](https://open-meteo.com/), based on ECMWF forecasts. Consult the providers' terms before redistributing data.

Vaibhav Chahar · [GitHub](https://github.com/VChahar1) · [LinkedIn](https://www.linkedin.com/in/vaibhav-chahar-89654a1a5)
