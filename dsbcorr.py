from copy import deepcopy
import networkx as nx
from tqdm import trange
import scipy as sp
from scipy.sparse.csgraph import laplacian
import random
import topcorr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lobpcg
from scipy.stats import spearmanr, median_abs_deviation
from astropy.stats import biweight_midcorrelation
import dcor
from joblib import Parallel, delayed
import numpy as np


def _process_window(w, X, w_size, w_shape, step, thr, type_thr, corr):
    block = X[w:w + w_size]
    data_len, n_vars = X.shape

    # 1) Compute chosen correlation matrix
    if corr == "pearsons":
        # Assume weightedcorrs returns an (n_vars,n_vars) array
        corr_mat = weightedcorrs(block, win_shape(w_size, w_shape))

    elif corr == "spearmans":
        corr_mat = np.eye(n_vars)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                rho, _ = spearmanr(block[:, i], block[:, j])
                corr_mat[i, j] = corr_mat[j, i] = rho

    elif corr == "bicor":
        # Biweight midcorrelation is median-based and robust to outliers
        corr_mat = np.eye(n_vars)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                bic = biweight_midcorrelation(block[:, i], block[:, j])
                corr_mat[i, j] = corr_mat[j, i] = bic

    elif corr == "dcor":
        # Distance correlation detects any dependence (zero iff independent)
        corr_mat = np.eye(n_vars)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                dcor_val = dcor.distance_correlation(block[:, i], block[:, j])
                corr_mat[i, j] = corr_mat[j, i] = dcor_val

    elif corr == "dcca":
        corr_mat = topcorr.dcca(block, 25)

    else:
        raise ValueError(f"Unknown corr method: {corr}")

    # 2) Clean NaNs & enforce self-correlation = 1
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    np.fill_diagonal(corr_mat, 1.0)

    # 3) Thresholding
    if type_thr is False:
        # simple absolute threshold
        mask = (np.abs(corr_mat) < thr)
        corr_mat[mask] = 0.0

    elif type_thr == "mad":
        # robust MAD-based threshold: keep edges where
        # |value – median| >= thr * MAD
        flat = corr_mat[np.triu_indices(n_vars, k=1)]
        med = np.median(flat)  # median of off-diagonals
        mad = median_abs_deviation(flat, scale='normal')  # scaled to resemble σ :contentReference[oaicite:3]{index=3}
        lower = med - thr * mad
        upper = med + thr * mad
        # zero out weak edges
        mask = ((corr_mat < lower) | (corr_mat > upper))
        corr_mat[mask == False] = 0.0

    elif type_thr == "neg":
        corr_mat[corr_mat > -thr] = 0.0

    elif type_thr == "pos":
        corr_mat[corr_mat < thr] = 0.0

    elif type_thr == "proportional":
        corr_mat = proportional_thr(corr_mat, thr)

    elif type_thr == "pmfg":
        if thr != 0:
            corr_mat[corr_mat > -thr] = 0.0
        corr_mat = nx.to_numpy_array(topcorr.pmfg(corr_mat))

    elif type_thr == "tmfg":
        corr_mat = nx.to_numpy_array(topcorr.tmfg(corr_mat, absolute=True))

    else:
        raise ValueError(f"Unknown threshold type: {type_thr}")
    return w, corr_mat

def rolling_window_parallel(X, w_size, w_shape, step, thr=0, type_thr=False, corr="pearsons", n_jobs=-1):
    data_len, n_vars = X.shape
    windows = list(range(0, data_len - w_size, step))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_window)(w, X, w_size, w_shape, step, thr, type_thr, corr)
        for w in windows
    )
    return dict(results)


def win_shape(w, w_shape):
    if w_shape == "square":
        return np.ones((w, 1)) / w

    elif w_shape == "tapered":
        theta = np.round(w / 3)
        w0 = (1 - np.exp(-1 / theta)) / (1 - np.exp(-w / theta))
        return ((w0 * np.exp((np.array(range(-w + 1, 1)) / theta))).T).reshape(w, 1)


def weightedcorrs(X, w):
    dt, N = np.shape(X)
    temp = X - np.tile(np.dot(w.T, X), (dt, 1))
    temp = np.dot(temp.T, (temp * np.tile(w, (1, N))))
    temp = 0.5 * (temp + temp.T)
    R = np.diag(temp)
    R = R.reshape(len(R), 1)
    R = temp / np.sqrt(np.dot(R, R.T))
    return R


def rolling_window(X, w_size, w_shape, step, thr=0, type_thr=False, corr="pearsons"):
    """
    Compute a series of correlation matrices over sliding windows of X,
    with options for Pearson, Spearman, biweight midcorrelation, distance correlation,
    DCCA, and various thresholding schemes including MAD-based robust thresholding.
    """
    data_len, n_vars = X.shape
    r_dict = {}

    for w in trange(0, int(data_len - w_size), step, desc="Windows"):
        block = X[w:w + w_size, :]

        # 1) Compute chosen correlation matrix
        if corr == "pearsons":
            # Assume weightedcorrs returns an (n_vars,n_vars) array
            corr_mat = weightedcorrs(block, win_shape(w_size, w_shape))

        elif corr == "spearmans":
            corr_mat = np.eye(n_vars)
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    rho, _ = spearmanr(block[:, i], block[:, j])
                    corr_mat[i, j] = corr_mat[j, i] = rho

        elif corr == "bicor":
            # Biweight midcorrelation is median-based and robust to outliers
            corr_mat = np.eye(n_vars)
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    bic = biweight_midcorrelation(block[:, i], block[:, j])
                    corr_mat[i, j] = corr_mat[j, i] = bic

        elif corr == "dcor":
            # Distance correlation detects any dependence (zero iff independent)
            corr_mat = np.eye(n_vars)
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    dcor_val = dcor.distance_correlation(block[:, i], block[:, j])
                    corr_mat[i, j] = corr_mat[j, i] = dcor_val

        elif corr == "dcca":
            corr_mat = topcorr.dcca(block, 25)

        else:
            raise ValueError(f"Unknown corr method: {corr}")

        # 2) Clean NaNs & enforce self-correlation = 1
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        np.fill_diagonal(corr_mat, 1.0)

        # 3) Thresholding
        if type_thr is False:
            # simple absolute threshold
            mask = (np.abs(corr_mat) < thr)
            corr_mat[mask] = 0.0

        elif type_thr == "mad":
            # robust MAD-based threshold: keep edges where
            # |value – median| >= thr * MAD
            flat = corr_mat[np.triu_indices(n_vars, k=1)]
            med = np.median(flat)  # median of off-diagonals
            mad = median_abs_deviation(flat, scale='normal')  # scaled to resemble σ :contentReference[oaicite:3]{index=3}
            lower = med - thr * mad
            upper = med + thr * mad
            # zero out weak edges
            mask = ((corr_mat < lower) | (corr_mat > upper))
            corr_mat[mask == False] = 0.0

        elif type_thr == "neg":
            corr_mat[corr_mat > -thr] = 0.0

        elif type_thr == "pos":
            corr_mat[corr_mat < thr] = 0.0

        elif type_thr == "proportional":
            corr_mat = proportional_thr(corr_mat, thr)

        elif type_thr == "pmfg":
            if thr != 0:
                corr_mat[corr_mat > -thr] = 0.0
            corr_mat = nx.to_numpy_array(topcorr.pmfg(corr_mat))

        elif type_thr == "tmfg":
            corr_mat = nx.to_numpy_array(topcorr.tmfg(corr_mat, absolute=True))

        else:
            raise ValueError(f"Unknown threshold type: {type_thr}")

        r_dict[w] = corr_mat

    return r_dict


def claplacian(M, norm=True):
    if norm == True:
        L = np.diag(sum(M)) - M
        v = 1. / np.sqrt(sum(M))
        return np.diag(v) * L * np.diag(v)
    else:
        return np.diag(sum(M)) - M


def eigenspectrum(L):
    eigvals = np.real(np.linalg.eig(L)[0])
    return -np.sort(-eigvals)


def all_spectra_lobpcg(A_dict, norm=True):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((9, len(dict_keys)))
    i = 0
    for key in dict_keys:
        L = laplacian(A_dict[key], norm)
        L_sparse = csr_matrix(L)
        X = np.random.normal(size=(L_sparse.shape[0], 9))
        eigenspectrums[:, i], _ = lobpcg(L_sparse, X, largest=False)
        i += 1
    return eigenspectrums


def all_spectra(A_dict, norm=True):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i = 0
    for key in dict_keys:
        L = laplacian(A_dict[key], norm)
        eigenspectrums[:, i] = eigenspectrum(L)
        i += 1
    return eigenspectrums


def all_spectra_signed(A_dict, norm=True):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i=0
    for r in A_dict:
        R = A_dict[r]
        A_pos = np.maximum(0, R)
        A_neg = -np.minimum(0, R)

        D_pos = np.diag(np.sum(A_pos, axis=1))
        D_neg = np.diag(np.sum(A_neg, axis=1))

        D_pos_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_pos)))
        D_neg_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_neg)))

        D_pos_inv_sqrt[np.isinf(D_pos_inv_sqrt)] = 0
        D_neg_inv_sqrt[np.isinf(D_neg_inv_sqrt)] = 0

        L_pos = np.eye(R.shape[0]) - np.dot(np.dot(D_pos_inv_sqrt, A_pos), D_pos_inv_sqrt)
        L_neg = np.eye(R.shape[0]) - np.dot(np.dot(D_neg_inv_sqrt, A_neg), D_neg_inv_sqrt)

        L_signed = L_pos + L_neg

        eigenvalues, eigenvectors = np.linalg.eigh(L_signed)

        eigenspectrums[:,i] = eigenvalues

        i+=1

    return eigenspectrums


def snapshot_dist(eigenspectrums, norm=True):
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N, N))
    for i in trange(N):
        for j in range(N):
            dist[i, j] = np.sqrt(np.sum(np.power((eigenspectrums[:, i] - eigenspectrums[:, j]), 2)))
            if norm == True:
                if max(max(eigenspectrums[:, i]), max(eigenspectrums[:, j])) > 1e-10:
                    dist[i, j] = dist[i, j] / np.sqrt(
                        max((np.sum(np.power(eigenspectrums[:, i], 2))), (np.sum(np.power(eigenspectrums[:, j], 2)))))
                else:
                    dist[i, j] = 0

    return dist


def landmark_snapshot_dist(eigenspectrums, lm_inds, norm=True):
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N, len(lm_inds)))
    for i in trange(N):
        for j in range(len(lm_inds)):
            k = lm_inds[j]
            dist[i, j] = np.sqrt(np.sum(np.power((eigenspectrums[:, i] - eigenspectrums[:, k]), 2)))
            if norm == True:
                if max(max(eigenspectrums[:, i]), max(eigenspectrums[:, k])) > 1e-10:
                    dist[i, j] = dist[i, j] / np.sqrt(
                        max((np.sum(np.power(eigenspectrums[:, i], 2))), (np.sum(np.power(eigenspectrums[:, k], 2)))))
                else:
                    dist[i, j] = 0
    return dist


def LMDS(D, lands, dim):
    Dl = D[:, lands]
    n = len(Dl)

    # Centering matrix
    H = - np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    # YY^T
    H = -H.dot(Dl ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    if dim:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            print('Error: Not enough positive eigenvalues for the selected dim.')
            return []
    if w.size == 0:
        print('Error: matrix is negative definite.')
        return []

    V = evecs[:, w]
    N = D.shape[1]
    Lh = V.dot(np.diag(1. / np.sqrt(evals[w]))).T
    Dm = D - np.tile(np.mean(Dl, axis=1), (N, 1)).T
    dim = w.size
    X = -Lh.dot(Dm) / 2.
    X -= np.tile(np.mean(X, axis=1), (N, 1)).T

    _, evecs = sp.linalg.eigh(X.dot(X.T))

    return (evecs[:, ::-1].T.dot(X)).T


def r_thr(r, thr, signed="pos"):
    if signed == False:
        r[np.where((r < thr) & (r > -thr))] = 0
    elif signed == "neg":
        r[np.where(r > -thr)] = 0
    else:
        r[np.where(r < thr)] = 0
    return r


def proportional_thr(r, p):
    thr_r = np.zeros((r.shape))
    r[np.where((-0.05 < r) & (r < 0.05))] = 0
    ut = np.triu(r, k=1)
    n = ((len(r) ** 2) / 2) - len(r)
    n = int(n * p)
    if len(np.where((0 < ut) | (ut < 0))[0]) < n:
        return r
    else:
        elem = [[x, y] for x, y in zip(np.where((0 < ut) | (ut < 0))[0], np.where((0 < ut) | (ut < 0))[1])]
        vals = ut[np.where((0 < ut) | (ut < 0))]
        vals = abs(vals)
        ind = np.argpartition(vals, -n)[-n:]
        for i in ind:
            thr_r[elem[i][0], elem[i][1]] = r[elem[i][0], elem[i][1]]
            thr_r[elem[i][1], elem[i][0]] = r[elem[i][0], elem[i][1]]
        # thr = min(vals[ind])
        np.fill_diagonal(thr_r, 1)
        return thr_r


def pmfg(corr_mat):
    """Construct a Planar Maximally Filtered Graph from a correlation matrix."""
    corr_mat[np.where(np.isnan(corr_mat) == True)] = 0
    n = corr_mat.shape[0]
    edges = [(corr_mat[i, j], i, j) for i in range(n) for j in range(i + 1, n) if corr_mat[i, j] != 0]
    # Sort edges based on weight in descending order
    edges.sort(reverse=True, key=lambda x: x[0])

    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    for _, i, j in edges:
        G.add_edge(i, j)
        if not nx.check_planarity(G)[0]:  # Check if the graph remains planar
            G.remove_edge(i, j)  # Remove the edge if adding it violates planarity

    return nx.to_numpy_array(G)


def sub_sample_zeros(X):
    zeros = list(np.where(X == 0)[0][1:])
    ind = 0
    while ind != len(zeros):
        intr = []
        curr = zeros[ind]
        intr.append(zeros[ind])
        ind += 1
        if ind != len(zeros):
            while zeros[ind] == curr:
                ind += 1
                curr = zeros[ind]
                intr.append(zeros[ind])
        if intr[-1] + 1 < X.shape[0]:
            st_val = X[intr[0] - 1]
            nd_val = X[intr[-1] + 1]
            n_val = len(intr)
            step = (nd_val - st_val) / (n_val + 1)
            for i in range(intr[0], intr[-1] + 1):
                X[i] = X[i - 1] + step
    return X


def weighted_random_walk(graph, start_node, walk_length, weight='weight'):
    """
    Perform a single weighted random walk starting from a given node.

    Parameters:
    - graph: NetworkX graph
    - start_node: Node to start the random walk from
    - walk_length: Length of the random walk
    - weight: Edge attribute to use as weight (default is 'weight')
    """
    path = [start_node]
    for _ in range(walk_length - 1):
        current_node = path[-1]
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:  # If no neighbors, end the walk
            break

        # Get weights of edges to neighbors
        weights = np.array([graph[current_node][neighbor].get(weight, 1) for neighbor in neighbors])
        probabilities = weights / weights.sum()  # Normalize to get probabilities

        # Choose the next node based on the probabilities
        next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
        path.append(next_node)

    return path


def characterize_nodes_weighted(graph, walk_length=10, num_walks=100, weight='weight'):
    """
    Characterize all nodes in the graph using weighted random walks.

    Parameters:
    - graph: NetworkX graph
    - walk_length: Length of each random walk
    - num_walks: Number of random walks per node
    - weight: Edge attribute to use as weight

    Returns:
    - node_characteristics: Dictionary of visit frequencies for each node
    """
    node_characteristics = {}
    for node in graph.nodes:
        # Collect weighted random walk paths starting from this node
        all_walks = [weighted_random_walk(graph, node, walk_length, weight) for _ in range(num_walks)]

        # Compute visit frequencies for each node
        visit_counts = {}
        for walk in all_walks:
            for visited_node in walk:
                visit_counts[visited_node] = visit_counts.get(visited_node, 0) + 1

        # Normalize frequencies
        total_visits = sum(visit_counts.values())
        visit_frequencies = {n: count / total_visits for n, count in visit_counts.items()}

        # Store the characteristics
        node_characteristics[node] = visit_frequencies
    return node_characteristics