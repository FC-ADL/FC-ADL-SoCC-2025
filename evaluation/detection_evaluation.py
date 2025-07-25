import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import dsbcorr
import fnmatch
from collections import Counter, defaultdict
import random
import scipy.sparse.linalg as spla
from tqdm import tqdm
from copy import deepcopy
import hdbscan
from sklearn.cluster import Birch
from baro.anomaly_detection import bocpd, nsigma
from baro.root_cause_analysis import robust_scorer
from functools import partial
import scipy.linalg as la
import ads_evt as spot
import time
import json
import sys
import warnings

warnings.filterwarnings('ignore')


def detect_spot(df, q=1e-2, depth=180, init_len=360, end=1080):
    """
    Run dSPOT on each column’s window and return a single list of 0/1
    labels of length (end - init_len)=720.

    0 = no change, 1 = change detected (from the earliest column alarm).
    """
    start = time.time()
    truncated = df.iloc[:end]

    stream_len = end - init_len

    stream_alarms = []
    for col in truncated.columns:
        series = truncated[col].values.astype(float)
        init_series = series[:init_len]
        stream = series[init_len:end]

        model = spot.dSPOT(q=q, depth=depth)
        model.fit(init_data=init_series, data=stream)
        model.initialize()
        alarms = model.run().get("alarms", [])

        if alarms:
            stream_alarms.append(alarms[0])

    if not stream_alarms:
        return [0] * stream_len

    first_cp_in_stream = min(stream_alarms)

    res = [
        0 if i < first_cp_in_stream else 1
        for i in range(stream_len)
    ]
    end = time.time()

    total_t = end - start
    return res, total_t


def detect_nsigma(df):
    start = time.time()
    df['time'] = np.arange(len(df))
    res = nsigma(df, startsfrom=360)

    if not res:
        end = time.time()
        total_t = end - start
        res = [0] * (len(df) - 360)
        return res, total_t

    cp_rel = res[0] - 360
    res = [0] * cp_rel + [1] * (len(df) - cp_rel - 360)

    end = time.time()

    total_t = end - start

    return res, total_t


def detect_birch(df):
    """
    BARO’s published BIRCH baseline:
      - threshold=0.5, branching_factor=50, n_clusters=3
      - Flag an anomaly at t if cluster_label[t] != cluster_label[t-1]
      - Once the first anomaly appears, everything after is anomalous.
    """
    start = time.time()

    X_train = df.iloc[:360].values
    X_pred = df.iloc[360:].values

    clf = Birch(threshold=0.5,
                branching_factor=50,
                n_clusters=3)
    clf.fit(X_train)

    labels_train = clf.predict(X_train)
    labels_pred = clf.predict(X_pred)
    labels_all = np.concatenate([labels_train, labels_pred])

    flips = np.concatenate([[False], labels_all[1:] != labels_all[:-1]])
    raw_flags = flips[len(X_train):].astype(int)
    step_preds = np.maximum.accumulate(raw_flags)
    res = step_preds.tolist()

    end = time.time()
    total_t = end - start

    return res, total_t


def detect_baro(df):
    start = time.time()
    res = bocpd(df.iloc[360:])

    if not res:
        end = time.time()
        total_t = end - start
        bin_res = [0] * (len(df) - 360)
        return bin_res, total_t

    bin_res = [0] * res[0] + [1] * (len(df.iloc[360:]) - res[0])

    end = time.time()
    total_t = end - start

    return bin_res, total_t


def detect_fcadl(df):
    start = time.time()
    win = 360
    step = 1
    thr = 0.1

    df_zero_na = df.replace(0, np.nan)
    df_interpolated = df_zero_na.interpolate(method='linear', axis=0, limit_direction='both')
    df_imputed = df_interpolated.fillna(method='ffill').fillna(method='bfill')
    df = df_imputed.diff()
    df = df.iloc[1:]

    data_arr = np.array(df)
    data_arr[np.where(data_arr < 1e-3)] = 0

    r_dict = dsbcorr.rolling_window(
        data_arr, win, "tapered", step, thr, False, "pearsons"
    )

    r_pos = {}
    for corr in r_dict:
        r_pos[corr] = deepcopy(r_dict[corr])
        r_pos[corr] = r_pos[corr] + 1

    eps = 1e-2

    def compute_affinity_matrix(A, eps=eps):
        """
        A: (n×n) numpy array, nonnegative adjacency
        returns S = (I + eps^2 D − eps A)^{-1}
        """
        n = A.shape[0]
        # degree diag
        deg = A.sum(axis=1)
        D = sp.diags(deg)
        I = sp.eye(n, format='csr')
        M = I + (eps ** 2) * D - eps * sp.csr_matrix(A)
        # factorize and invert via n solves
        factor = spla.factorized(M)
        S = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            S[:, i] = factor(e)
        return S

    # compute once per window
    S_dict = {w: compute_affinity_matrix(A) for w, A in r_pos.items()}
    all_windows = list(r_pos.keys())

    S_sqrt = np.stack([np.sqrt(S_dict[w]) for w in all_windows], axis=0)
    n, m, _ = S_sqrt.shape
    S_flat = S_sqrt.reshape(n, m * m)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(5, len(r_pos) // 5),
        min_samples=5,
        metric='euclidean',
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    labels = clusterer.fit_predict(S_flat)
    arr = np.array(labels)
    # both neighbors must be non -1, and different
    mask = (arr[:-1] != -1) & (arr[1:] != -1)
    idxs = np.where((arr[:-1] != arr[1:]) & mask)[0]
    changepoint = int(idxs[0]) if idxs.size else None
    if changepoint is not None:
        result = ([0] * changepoint) + ([1] * (len(labels) - changepoint))
    else:
        result = ([0] * len(labels))

    end = time.time()
    total_t = end - start

    return result, total_t


def run_detection(method, condition, n, run_iteration):
    os.makedirs("../results", exist_ok=True)

    df = pd.read_csv(f"../data/{condition}_{n}/cpu_usage.csv")
    df = df.iloc[:-360]
    df_zero_na = df.replace(0, np.nan)
    df_interpolated = df_zero_na.interpolate(method='linear', axis=0, limit_direction='both')
    df = df_interpolated.fillna(method='ffill').fillna(method='bfill')
    df.dropna(axis=1, how='all', inplace=True)
    with open(f"../data/{condition}_{n}/fault_inject.txt", "r") as f:
        fault_t = int(f.read())

    if 'time' in df.columns:
        df = df.drop('time', axis=1)
        misc_patterns = ["nginx*", "jaeger*", "socialnetwork-elasticsearch*"]
        keep_cols = [
            c for c in df.columns
            if not any(fnmatch.fnmatchcase(c, pat) for pat in misc_patterns)
        ]
        df = df[keep_cols]

        bases = ["-".join(c.split("-")[:-2]) for c in df.columns]
        totals = Counter(bases)

        indices = defaultdict(int)
        new_cols = []
        for base in bases:
            if totals[base] > 1:
                indices[base] += 1
                new_cols.append(f"{base}-{indices[base]}")
            else:
                new_cols.append(base)

        df.columns = new_cols

        def split_base(name):
            parts = name.split("-")
            return parts[:-1] + [int(parts[-1]) if parts[-1].isdigit() else parts[-1]]

        df = df.reindex(columns=sorted(df.columns, key=split_base))

    non_zero_var_cols = df.var() != 0
    df_clean = df.loc[:, non_zero_var_cols]

    normal_data = df_clean.iloc[5:fault_t]
    abnormal_data = df_clean.iloc[fault_t:int(fault_t + len(normal_data))]

    s_list = df_clean.columns.tolist()

    if method == 'fcadl':
        return detect_fcadl(df)
    elif method == 'nsigma':
        return detect_nsigma(df_clean)
    elif method == 'spot':
        return detect_spot(df_clean)
    elif method == 'birch':
        return detect_birch(df_clean)
    elif method == 'baro':
        return detect_baro(df_clean)


if __name__ == "__main__":
    run_iteration = sys.argv[1]
    services = ['cp', 'hts', 'uts', 'ums']
    faults = ['cpuhog', 'memhog', 'memhog_redo', 'netdelay']

    methods = ['spot', 'fcadl', 'nsigma', 'birch', 'baro']
    repeats = 5
    k = 5

    root_cause_map = {
        'cp': 'compose-post-service',
        'hts': 'home-timeline-service',
        'uts': 'user-timeline-service',
        'ums': 'user-mention-service'
    }

    fault_conditions = [f'{s}_{f}_sin' for s in services for f in faults]
    prec_dict = {
        fault: {
            root_cause_map[service]: {method: [] for method in methods}
            for service in services
        }
        for fault in faults
    }
    recall_dict = {
        fault: {
            root_cause_map[service]: {method: [] for method in methods}
            for service in services
        }
        for fault in faults
    }

    f1_dict = {
        fault: {
            root_cause_map[service]: {method: [] for method in methods}
            for service in services
        }
        for fault in faults
    }

    t_dict = {
        fault: {
            root_cause_map[service]: {method: [] for method in methods}
            for service in services
        }
        for fault in faults
    }

    for condition in tqdm(fault_conditions):
        fault = "_".join(condition.split('_')[1:-1])
        root_cause = root_cause_map[condition.split('_')[0]]
        for method in methods:
            exist = 0
            total_f1 = 0
            total_t = 0
            print(f"Doing method: {method}")
            for n in range(0, 4):
                if os.path.isdir(f"../data/{condition}_{n}"):
                    exist += 1
                    pred, t = run_detection(method, condition, n, run_iteration)
                    with open(f"../data/{condition}_{n}/fault_inject.txt",
                              "r") as f:
                        fault_t = int(f.read())
                    # pred = pred[:720]
                    true = ([0] * (fault_t - 360)) + ([1] * (720 - (fault_t - 360)))
                    TP = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
                    FP = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
                    FN = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
                    TN = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)

                    precision = TP / (TP + FP) if (TP + FP) else 0
                    recall = TP / (TP + FN) if (TP + FN) else 0

                    if precision + recall:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    total_f1 += f1
                    total_t += t

                    prec_dict[fault][root_cause][method].append(precision)
                    recall_dict[fault][root_cause][method].append(recall)
                    f1_dict[fault][root_cause][method].append(f1)
                    t_dict[fault][root_cause][method].append(t)

            if exist != 0:
                total_f1 = total_f1 / exist
                total_t = total_t / exist
                print(f"F1: {total_f1}")
                print(f"Time: {total_t}")
                print(f"{method} {condition} Overall F1-score = {total_f1:.3f}, Time taken = {total_t:.3f}")

    with open(f"../results/ms-fault-detection/f1_{run_iteration}.json", "w",
              encoding="utf-8") as f:
        json.dump(f1_dict, f, ensure_ascii=False, indent=2)

    with open(f"../results/ms-fault-detection/prec_{run_iteration}.json", "w",
              encoding="utf-8") as f:
        json.dump(prec_dict, f, ensure_ascii=False, indent=2)

    with open(f"../results/ms-fault-detection/recall_{run_iteration}.json", "w",
              encoding="utf-8") as f:
        json.dump(recall_dict, f, ensure_ascii=False, indent=2)

    with open(f"../results/ms-fault-detection/t_{run_iteration}.json", "w",
              encoding="utf-8") as f:
        json.dump(t_dict, f, ensure_ascii=False, indent=2)


