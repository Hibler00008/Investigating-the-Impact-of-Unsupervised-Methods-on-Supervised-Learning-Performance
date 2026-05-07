import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import math


# ==============================
# USER CONFIG
# ==============================
DATASET_DIR = "datasets"
MAX_K = 40
KMEANS_THREADS = 10


# ==============================
# CPU FUNCTIONS
# ==============================
def kmeans_inertia(k, X):
    km = KMeans(n_clusters=k, n_init="auto", algorithm="elkan", random_state=42)
    km.fit(X)
    return float(km.inertia_)


# ==============================
# PROCESS ONE FILE
# ==============================
def process_file(path, filename):

    print(f"\n\n========== PROCESSING: {filename} ==========")

    df = pd.read_csv(path, low_memory=False)
    has_y = "y" in df.columns

    # Separate y if exists
    y = df["y"] if has_y else None
    Xdf = df.drop(columns=["y"]) if has_y else df.copy()

    # ===========================================
    # CLEANING: Convert invalid strings to NaN
    # ===========================================
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    # Now safe to convert to float32
    X = Xdf.values.astype(np.float32)
    n, d = X.shape
    print(f"Samples={n}, Features={d}")

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # -------- KMEANS ----------
    print("Running CPU KMeans sweep...")
    Ks = list(range(1, MAX_K + 1))

    inertias = Parallel(n_jobs=KMEANS_THREADS, prefer="threads")(
        delayed(kmeans_inertia)(k, Xs) for k in Ks
    )

    # Knee detection
    try:
        knee = KneeLocator(Ks, inertias, S=1.0, curve="convex", direction="decreasing")
        Kopt = knee.knee if knee.knee else int(math.sqrt(n))
    except:
        Kopt = int(math.sqrt(n))

    print("Best K =", Kopt)

    km = KMeans(n_clusters=Kopt, n_init="auto", algorithm="elkan", random_state=42)
    klabels = km.fit_predict(Xs)

    # ==============================
    # Cluster Summary
    # ==============================
    print("\n===== KMEANS SUMMARY =====")
    unique_clusters = np.unique(klabels)
    num_clusters = len(unique_clusters)
    print("Number of clusters:", num_clusters)

    cluster_sizes = []
    for cid in unique_clusters:
        count = np.sum(klabels == cid)
        cluster_sizes.append((cid, count))
        print(f"Cluster {cid}: {count} points")
    print("==========================\n")

    # Save main output
    outfile = os.path.join(DATASET_DIR, f"KMEANS_{filename}")
    out = Xdf.copy()
    out["kmeans_id"] = klabels.astype(int)
    if has_y:
        out["y"] = y
    out.to_csv(outfile, index=False)
    print("Saved:", outfile)

    # Save summary
    summary_file = os.path.join(DATASET_DIR, f"KMEANS_{filename}_summary.csv")
    pd.DataFrame(cluster_sizes, columns=["cluster_id", "count"]).to_csv(summary_file, index=False)
    print("Saved summary:", summary_file)


# ==============================
# MAIN
# ==============================
def main():

    files = [
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".csv")
        and not f.startswith("LDA_")
        and not f.startswith("KMEANS_")
    ]

    print("\nFiles to process:", files)

    for fname in files:
        fpath = os.path.join(DATASET_DIR, fname)
        process_file(fpath, fname)


if __name__ == "__main__":
    main()
