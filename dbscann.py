import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

DATA_DIR = "datasets"

def choose_best_gmm(X_scaled, min_components=1, max_components=10):
    lowest_bic = np.inf
    best_gmm = None
    best_labels = None
    best_n = None

    for n in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=0)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)

        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            best_labels = gmm.predict(X_scaled)
            best_n = n

    return best_labels, best_n, best_gmm


def process_file(path):
    print(f"\nProcessing: {path}")

    df = pd.read_csv(path)

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    # Extract y if present
    if "y" in df.columns:
        X = df.drop(columns=["y"])
        has_y = True
    else:
        X = df.copy()
        has_y = False

    # Impute missing values in X
    X = X.fillna(X.mean())

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running GMM (auto-tuning components using BIC)...")
    labels, n_components, model = choose_best_gmm(X_scaled)
    print(f"Best number of components (clusters): {n_components}")

    # Top-3 probability clusters
    probs = model.predict_proba(X_scaled)
    top1, top2, top3 = [], [], []

    for row in probs:
        sorted_idx = np.argsort(row)[::-1]
        s = sorted_idx.tolist()
        while len(s) < 3:
            s.append(-1)
        top1.append(s[0])
        top2.append(s[1])
        top3.append(s[2])

    # Add columns
    df["gmm_id"] = labels
    df["gmm_top1"] = top1
    df["gmm_top2"] = top2
    df["gmm_top3"] = top3

    # Place before y if needed
    if has_y:
        y_col = df["y"]
        df = df.drop(columns=["y"])
        df["y"] = y_col

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} points")

    # Save output
    filename = os.path.basename(path)
    outname = os.path.join(DATA_DIR, f"GMM_{filename}")
    df.to_csv(outname, index=False)
    print(f"\nSaved: {outname}")


def main():
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
        and not (
            f.startswith("KMEANS_")
            or f.startswith("MBKMEANS_")
            or f.startswith("LDA_")
            or f.startswith("DBSCAN_")
            or f.startswith("GMM_")
        )
    ]

    if not files:
        print("No files to process.")
        return

    for f in files:
        process_file(os.path.join(DATA_DIR, f))


if __name__ == "__main__":
    main()
