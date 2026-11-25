import pandas as pd
for m in ["naive", "lear"]:
    f = pd.read_parquet(f"data/forecasts/DK_1_{m}.parquet")
    print(m, list(f.index[f.isna().any(axis=1)].date))