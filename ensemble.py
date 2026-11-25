import pandas as pd
from src.evaluate_prob import wide_to_long, lear_eq, report
from src.lgbm import load_long

y = load_long("data/processed/DK_1_dataset.parquet")["price"]
fc = pd.read_parquet("data/forecasts/DK_1_lgbm_q.parquet")
cq = pd.read_parquet("data/forecasts/DK_1_lgbm_cqr.parquet")
lear = wide_to_long(pd.read_parquet("data/forecasts/DK_1_lear.parquet"), "lear")
naive = wide_to_long(pd.read_parquet("data/forecasts/DK_1_naive.parquet"), "naive")
eq = lear_eq(lear, y.reindex(lear.index), 56)

idx = eq.dropna().index.intersection(cq.dropna().index)
ens = (eq.loc[idx] + cq.loc[idx]) / 2          # quantile averaging

report(y.loc[fc.index],
       {"naive": naive, "lear": lear, "lgbm_median": fc["q0.50"], "ens_median": ens["q0.50"]},
       {"LEAR-EQ": eq, "LGBM": fc, "LGBM-CQR": cq, "ENS": ens})