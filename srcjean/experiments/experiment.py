import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .kmeans_cost import kmeans_cost

K_VALUES = [2, 4, 8, 16, 32, 64]
Q_FRACTIONS = [0.0025, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50]
N_TRIALS = 5


def coreset_size(n, k, frac):
    target = int(round(frac * n))
    return max(k + 1, target)


def fit_full_kmeans(P, k, seed=0):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(P)
    return km.cluster_centers_


def make_coreset(P, AlgoClass, m, seed=0):
    np.random.seed(seed)
    algo = AlgoClass(m)
    Q = algo.generate(P)
    return Q, algo.weights


def fit_centers(Q, w, k, seed=0):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(Q, sample_weight=w)
    return km.cluster_centers_


def run_grid(P, algorithms, k_values=K_VALUES, q_fractions=Q_FRACTIONS, n_trials=N_TRIALS):
    n = len(P)
    rows = []
    full_centers = {}
    coreset_centers = []

    for k in k_values:
        print(f"  k={k}", flush=True)
        C = fit_full_kmeans(P, k, seed=0)
        full_centers[k] = C
        cost_P_C = kmeans_cost(P, C)

        for frac in q_fractions:
            m = coreset_size(n, k, frac)
            print(f"    frac={frac:.4f} m={m}", flush=True)

            for algo_name, AlgoClass in algorithms.items():
                for seed in range(n_trials):
                    Q, w = make_coreset(P, AlgoClass, m, seed=seed)
                    Cp = fit_centers(Q, w, k, seed=seed)

                    cost_Q_C = kmeans_cost(Q, C, w)
                    cost_Q_Cp = kmeans_cost(Q, Cp, w)
                    cost_P_Cp = kmeans_cost(P, Cp)

                    r1 = cost_Q_C / cost_P_C
                    r2 = cost_Q_Cp / cost_P_Cp
                    r3 = cost_P_Cp / cost_P_C

                    row = {
                        "algo": algo_name,
                        "k": k,
                        "frac": frac,
                        "m": m,
                        "seed": seed,
                        "r1": r1,
                        "r2": r2,
                        "r3": r3,
                    }
                    rows.append(row)

                    entry = {
                        "algo": algo_name,
                        "k": k,
                        "frac": frac,
                        "m": m,
                        "seed": seed,
                        "C": Cp,
                    }
                    coreset_centers.append(entry)

    df = pd.DataFrame(rows)
    centers = {"full": full_centers, "coreset": coreset_centers}
    return df, centers


def _center_rows(C_orig, meta, feature_names):
    rows = []
    for i in range(len(C_orig)):
        row = dict(meta)
        row["center_idx"] = i
        for j, name in enumerate(feature_names):
            row[name] = C_orig[i, j]
        rows.append(row)
    return rows


def centers_to_dataframe(centers, scaler, feature_names):
    rows = []

    for k, C in centers["full"].items():
        C_orig = scaler.inverse_transform(C)
        meta = {"algo": "full", "k": k, "frac": None, "m": None, "seed": None}
        rows.extend(_center_rows(C_orig, meta, feature_names))

    for entry in centers["coreset"]:
        C_orig = scaler.inverse_transform(entry["C"])
        meta = {
            "algo": entry["algo"],
            "k": entry["k"],
            "frac": entry["frac"],
            "m": entry["m"],
            "seed": entry["seed"],
        }
        rows.extend(_center_rows(C_orig, meta, feature_names))

    return pd.DataFrame(rows)
