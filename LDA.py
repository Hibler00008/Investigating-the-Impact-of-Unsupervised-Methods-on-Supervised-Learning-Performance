import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

base_path = "./datasets/"

for filename in os.listdir(base_path):
    if filename.lower().endswith(".csv") and not filename.startswith("LDA_"):
        input_path = os.path.join(base_path, filename)
        output_path = os.path.join(base_path, f"LDA_{filename}")

        df = pd.read_csv(input_path, low_memory=False)
        if "y" not in df.columns:
            continue

        X = df.drop(columns=["y"])
        y = df["y"]

        X = X.apply(pd.to_numeric, errors="coerce")
        df_clean = pd.concat([X, y], axis=1).dropna()
        X = df_clean.drop(columns=["y"])
        y = df_clean["y"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X_scaled, y)

        lda_columns = [f"LD{i+1}" for i in range(X_lda.shape[1])]
        df_lda = pd.DataFrame(X_lda, columns=lda_columns)
        df_lda["y"] = y.to_numpy()

        df_lda.to_csv(output_path, index=False)
        print("Saved:", output_path)
