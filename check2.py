import os
import pandas as pd

base = "datasets"

for f in os.listdir(base):
    if f.startswith("MBKMEANS_") and f.endswith(".csv"):

        path = os.path.join(base, f)
        df = pd.read_csv(path)

        # second last col = y, last col = MBKMEANS
        cols = list(df.columns)

        y_col = cols[-2]
        mb_col = cols[-1]

        new_cols = cols[:-2] + [mb_col, y_col]

        df = df[new_cols]
        df.to_csv(path, index=False)
        print(f"Fixed → {f}")
