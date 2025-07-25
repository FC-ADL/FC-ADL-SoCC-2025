from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis
from pyrca.analyzers.random_walk import RandomWalk
from pyrca.analyzers.ht import HT
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from baro.anomaly_detection import bocpd
from baro.anomaly_detection import nsigma as detect_nsigma
from baro.root_cause_analysis import robust_scorer
from baro.root_cause_analysis import nsigma as rca_nsigma
import json
from tqdm import trange
import torch.optim as optim
from torch.optim import lr_scheduler
from sknetwork.ranking import PageRank
from sklearn.preprocessing import normalize
import dsbcorr
import fnmatch
from collections import Counter, defaultdict
import scipy.sparse.linalg as spla
from tqdm import tqdm
from copy import deepcopy
import hdbscan
import time
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import sys
import os

warnings.filterwarnings('ignore')

def run_epsilon(normal_data, abnormal_data, top_k=5):
    # abnormal and normal data length must be the same
    n_L = normal_data.shape[0]
    a_L = abnormal_data.shape[0]

    if n_L < a_L:
        abnormal_data = abnormal_data.iloc[:n_L]
    elif a_L < n_L:
        normal_data = normal_data.iloc[:a_L]

    start = time.time()
    model = EpsilonDiagnosis(config=EpsilonDiagnosis.config_class(alpha=0.05, root_cause_top_k=top_k))
    model.train(normal_data)
    res = model.find_root_causes(abnormal_data)
    end = time.time()
    total_t = end - start
    return [rc[0] for rc in res.root_cause_nodes], total_t


def run_dl_pr(abnormal_data, pr_alpha=0.85, top_k=5):
    start = time.time()
    # Absolute correlations for MicroDiag-style weightings
    corr = abnormal_data.corr().abs()

    # Get DirectLiNGAM causal graph and inverse edges
    model = lingam.DirectLiNGAM()
    model.fit(abnormal_data.values)
    B = model.adjacency_matrix_
    n = B.shape[0]
    metrics = list(abnormal_data.columns)
    inv_mask = (B != 0).T

    # Create weighted inverse directed graph
    G = nx.DiGraph()
    G.add_nodes_from(metrics)
    for i in range(n):
        for j in range(n):
            if inv_mask[i, j]:
                w = corr.iat[i, j]
                if w > 0:
                    G.add_edge(metrics[i], metrics[j], weight=w)

    # Normalize edge‐weights to probabilities
    for u in G:
        total = sum(d['weight'] for _, _, d in G.out_edges(u, data=True))
        if total > 0:
            for _, v, d in G.out_edges(u, data=True):
                d['weight'] /= total

    pr_scores = nx.pagerank(G, alpha=pr_alpha, weight='weight')

    top = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    end = time.time()
    total_t = end - start
    return [node for node, score in top], total_t


def run_pc_pr(abnormal_data, pc_alpha=0.05, pr_alpha=0.85, top_k=5):
    start = time.time()
    # Absolute correlations for MicroDiag-style weightings
    corr = abnormal_data.corr().abs()

    # Get PC causal graph and inverse edges
    cg = pc(np.array(abnormal_data), alpha=pc_alpha, show_progress=False)
    raw = np.array(cg.G.graph)
    n = raw.shape[0]

    # make an empty binary adjacency
    bin_adj = np.zeros((n, n), dtype=int)

    # loop over each pair i, j
    for i in range(n):
        for j in range(n):
            # directed: cg.G.graph[j,i]==1 & cg.G.graph[i,j]==-1 means i → j
            if raw[j, i] == 1 and raw[i, j] == -1:
                bin_adj[i, j] = 1

            # undirected: both == –1  ⇒ treat as bidirectional
            elif raw[i, j] == -1 and raw[j, i] == -1:
                bin_adj[i, j] = 1
                bin_adj[j, i] = 1

            # confounded (bidirected): both == +1 ⇒ also bidirectional
            elif raw[i, j] == 1 and raw[j, i] == 1:
                bin_adj[i, j] = 1
                bin_adj[j, i] = 1

    adj = bin_adj
    n = adj.shape[0]
    metrics = list(abnormal_data.columns)
    inv_mask = (adj.T != 0)

    # Create weighted inverse directed graph
    G = nx.DiGraph()
    G.add_nodes_from(metrics)
    for i in range(n):
        for j in range(n):
            if inv_mask[i, j]:
                weight = corr.iat[i, j]
                if weight > 0:
                    G.add_edge(metrics[i], metrics[j], weight=weight)

    # Normalize edge‐weights to probabilities
    for u in G:
        total = sum(d['weight'] for _, _, d in G.out_edges(u, data=True))
        if total > 0:
            for _, v, d in G.out_edges(u, data=True):
                d['weight'] /= total

    # To dataframe adjacency matrix for service labels and input to random walker
    pr_scores = nx.pagerank(G, alpha=pr_alpha, weight='weight')
    top = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    end = time.time()
    total_t = end - start
    return [node for node, score in top], total_t


def run_circa(normal_data, abnormal_data, s_list, alpha=0.05, top_k=5):
    start = time.time()
    cg = pc(np.array(normal_data), alpha=alpha, show_progress=False)
    raw = np.array(cg.G.graph)
    n = raw.shape[0]

    # make an empty binary adjacency
    bin_adj = np.zeros((n, n), dtype=int)

    # loop over each pair i, j
    for i in range(n):
        for j in range(n):
            # directed: cg.G.graph[j,i]==1 & cg.G.graph[i,j]==-1 means i → j
            if raw[j, i] == 1 and raw[i, j] == -1:
                bin_adj[i, j] = 1

            # undirected: both == –1  ⇒ treat as bidirectional
            elif raw[i, j] == -1 and raw[j, i] == -1:
                bin_adj[i, j] = 1
                bin_adj[j, i] = 1

            # confounded (bidirected): both == +1 ⇒ also bidirectional
            elif raw[i, j] == 1 and raw[j, i] == 1:
                bin_adj[i, j] = 1
                bin_adj[j, i] = 1

    graph_df = pd.DataFrame(bin_adj, s_list, s_list)

    model = HT(
        config=HT.config_class(
            graph=graph_df,
            root_cause_top_k=top_k
        )
    )

    model.train(normal_data)

    res = model.find_root_causes(abnormal_data)
    end = time.time()
    total_t = end - start
    return [rc[0] for rc in res.root_cause_nodes], total_t


def run_fcadl(df, win=360, step=1, thr=0.1, top_k=5, gt=False):
    df_zero_na = df.replace(0, np.nan)
    df_interpolated = df_zero_na.interpolate(method='linear', axis=0, limit_direction='both')
    df_imputed = df_interpolated.fillna(method='ffill').fillna(method='bfill')
    df = df_imputed.diff()
    df = df.iloc[1:]

    var_map = {var: i for var, i in enumerate(df.columns)}

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
    S_flat = S_sqrt.reshape(n, m * m)  # now shape == (n_windows, m*m)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(5, len(r_pos) // 5),
        min_samples=5,
        metric='euclidean',
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    labels = clusterer.fit_predict(S_flat)

    start = time.time()

    is_outlier = labels == -1
    valid = ~is_outlier
    changes = np.where((labels[:-1] != labels[1:]) & valid[:-1] & valid[1:])[0]

    if changes.size == 0:
        end = time.time()
        total_t = end - start
        return [], total_t

    changepoint = changes[0] + 1
    new_labels = np.zeros_like(labels)
    new_labels[changepoint:] = 1
    new_labels[is_outlier] = -1

    labels=new_labels

    normal_lbl = 0
    abnorm_lbl = 1

    all_windows = list(r_pos.keys())

    normal_idxs = np.where(labels == normal_lbl)[0]
    abn_idxs = np.where(labels == abnorm_lbl)[0]

    # now average their adjacency matrices
    def cluster_centroid(idxs):
        M_sum = np.zeros_like(r_dict[all_windows[0]])
        for i in idxs:
            win_key = all_windows[i]
            M_sum += r_dict[win_key]
        return M_sum / len(idxs)

    normal_mat = cluster_centroid(normal_idxs)
    abnormal_mat = cluster_centroid(abn_idxs)

    normal_mat = normal_mat / np.max(normal_mat)
    normal_mat[np.where((normal_mat < thr) & (normal_mat > -thr))] = 0

    abnormal_mat = abnormal_mat / np.max(abnormal_mat)
    abnormal_mat[np.where((abnormal_mat < thr) & (abnormal_mat > -thr))] = 0

    sub_G = abnormal_mat - normal_mat
    G = nx.from_numpy_array(sub_G)

    node_characteristics = dsbcorr.characterize_nodes_weighted(G, walk_length=10, num_walks=10, weight='weight')

    total_f = defaultdict(float)
    for i, characteristics in node_characteristics.items():
        for node in characteristics:
            total_f[node] += characteristics[node]

    combined_frequencies = defaultdict(float)
    for service_id, frequency in total_f.items():
        service_type = var_map.get(service_id, 'unknown')
        combined_frequencies[service_type] += frequency

    # Convert the combined frequencies dictionary to a pandas dataframe
    combined_frequencies_df = pd.DataFrame(
        list(combined_frequencies.items()), columns=["Service Type", "Combined Frequency"]
    )
    end = time.time()
    total_t = end - start
    return combined_frequencies_df.nlargest(top_k, "Combined Frequency")["Service Type"].tolist(), total_t


def run_baro(df, top_k=5):
    res = bocpd(df)

    start = time.time()
    top_5 = robust_scorer(df, anomalies=res)["ranks"][:top_k]
    end = time.time()
    total_t = end - start

    return top_5, total_t


def run_nsigma(df, top_k=5):
    df['time'] = np.arange(len(df))
    detect_res = detect_nsigma(df, startsfrom=0)

    start = time.time()
    rca_res = rca_nsigma(df, detect_res[0])
    end = time.time()
    total_t = end - start

    return rca_res["ranks"][:top_k], total_t


def run_rca(rca_method, condition, n, k):
    df = pd.read_csv(f"../data/{condition}_{n}/cpu_usage.csv")
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

        # 2) Compute “base” names (everything except the last two dash-fields)
        bases = ["-".join(c.split("-")[:-2]) for c in df.columns]
        totals = Counter(bases)

        # 3) If a base occurs multiple times, append a running index
        indices = defaultdict(int)
        new_cols = []
        for base in bases:
            if totals[base] > 1:
                indices[base] += 1
                new_cols.append(f"{base}-{indices[base]}")
            else:
                new_cols.append(base)

        df.columns = new_cols

        # 4) Finally, sort the columns by your custom split_base key
        def split_base(name):
            # your existing logic; e.g. split on dash and return a tuple for sorting
            parts = name.split("-")
            return parts[:-1] + [int(parts[-1]) if parts[-1].isdigit() else parts[-1]]

        df = df.reindex(columns=sorted(df.columns, key=split_base))

    non_zero_var_cols = df.var() != 0
    df_clean = df.loc[:, non_zero_var_cols]
    fault_t = fault_t
    normal_data = df_clean.iloc[5:fault_t]
    abnormal_data = df_clean.iloc[fault_t:int(fault_t + len(normal_data))]
    s_list = df_clean.columns.tolist()

    if rca_method == 'epsilon':
        return run_epsilon(normal_data, abnormal_data)

    if rca_method == 'PC+PR':
        return run_pc_pr(df_clean)

    if rca_method == 'DL+PR':
        return run_dl_pr(df_clean)

    if rca_method == 'CIRCA':
        return run_circa(normal_data, abnormal_data, s_list)

    if rca_method == 'BARO':
        return run_baro(df_clean, k)

    if rca_method == 'nsigma':
        return run_nsigma(df_clean, k)

    if rca_method == 'fcadl':
        return run_fcadl(df, top_k=5)


if __name__ == "__main__":
    run_iteration = sys.argv[1]
    os.makedirs("../results", exist_ok=True)
    
    services = ['cp', 'hts', 'uts', 'ums']
    faults = ['cpuhog','memhog', 'memhog_redo', 'netdelay']
    methods = ['epsilon', 'PC+PR', 'DL+PR', 'CIRCA', 'nsigma', 'fcadl']

    repeats = 5
    k = 5

    root_cause_map = {
        'cp': 'compose-post-service',
        'hts': 'home-timeline-service',
        'uts': 'user-timeline-service',
        'ums': 'user-mention-service'
    }

    fault_conditions = [f'{s}_{f}_sin' for s in services for f in faults]

    top_1 = {}
    top_3 = {}
    top_5 = {}
    t_tot = {}

    for condition in tqdm(fault_conditions):
        top_1[condition] = {}
        top_3[condition] = {}
        top_5[condition] = {}
        t_tot[condition] = {}
        root_cause = root_cause_map[condition.split('_')[0]]
        pbar = tqdm(methods, desc="Starting...")
        for method in pbar:
            print(f"{method}")
            pbar.set_description(f"Running {method}", refresh=False)
            correct_1 = 0
            correct_3 = 0
            correct_5 = 0
            total_time = 0
            exist = 0
            for n in range(0, 4):
                if os.path.isdir(f"../data/{condition}_{n}"):
                    exist += 1
                    result, t = run_rca(method, condition, n, k)
                    total_time += t
                    if root_cause in result[:1]:
                        correct_1 += 1

                    if root_cause in result[:3]:
                        correct_3 += 1

                    if root_cause in result[:5]:
                        correct_5 += 1

            acc_1 = correct_1 / exist
            acc_3 = correct_3 / exist
            acc_5 = correct_5 / exist

            top_1[condition][method] = acc_1
            top_3[condition][method] = acc_3
            top_5[condition][method] = acc_5

            t_tot[condition][method] = total_time / exist

    with open(f"../results/ms-rca/top1_{run_iteration}.json", "w", encoding="utf-8") as f:
        json.dump(top_1, f, ensure_ascii=False, indent=2)

    with open(f"../results/ms-rca/top3_{run_iteration}.json", "w", encoding="utf-8") as f:
        json.dump(top_3, f, ensure_ascii=False, indent=2)

    with open(f"../ms-rca/top5_{run_iteration}.json", "w", encoding="utf-8") as f:
        json.dump(top_5, f, ensure_ascii=False, indent=2)

    with open(f"../ms-rca/t_{run_iteration}.json", "w", encoding="utf-8") as f:
        json.dump(t_tot, f, ensure_ascii=False, indent=2)

