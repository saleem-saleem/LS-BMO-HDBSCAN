rom __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
import hdbscan


# ---------------------------
# Utility: Objective function
# ---------------------------

def sse_objective(X: np.ndarray, centroids: np.ndarray) -> float:
    """
    f(X_i) = sum_{n=1..N} min_{1<=j<=k} || x_n - c_j ||^2          (Eq. 6)
    """
    _, dists = pairwise_distances_argmin_min(X, centroids)
    return float(np.sum(dists ** 2))


def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels, _ = pairwise_distances_argmin_min(X, centroids)
    return labels


def compute_entropy_from_centroids(X: np.ndarray, centroids: np.ndarray) -> float:
    """
    H(t) = - sum_{i=1..k} p_i(t) * log(p_i(t)),  p_i(t)=n_i/N       (Eq. 7)
    """
    labels = assign_labels(X, centroids)
    N = X.shape[0]
    k = centroids.shape[0]
    H = 0.0
    for i in range(k):
        p = np.sum(labels == i) / max(N, 1)
        if p > 0:
            H -= p * math.log(p + 1e-12)
    return H


# ----------------------------------------------------
# L-SHADE with archive, success-history & LPSR (Eqs.)
# ----------------------------------------------------

@dataclass
class LSHADEParams:
    k: int
    pop_max: int = 100
    pop_min: int = 20
    F_min: float = 0.3
    F_max: float = 0.9
    CR_min: float = 0.1
    CR_max: float = 0.9
    T_max: int = 300
    eps_div: float = 1e-3
    memory_size: int = 20
    p_best_frac: float = 0.2   # Top-p% for pbest (Eq. 8)
    random_state: Optional[int] = 42


class LSHADE:
    """
    L-SHADE tailored for centroid optimization in R^{k*d}.
    Implements:
      - Population init (Eq. 5)
      - Archive A, mutation/crossover/selection (Eqs. 8-12)
      - Fuzzy scaling based on entropy H(t) (custom step per spec)
      - Linear population size reduction (Eq. 13)
    """

    def __init__(self, params: LSHADEParams):
        self.p = params
        if self.p.random_state is not None:
            np.random.seed(self.p.random_state)
        self.F_hist = np.full(self.p.memory_size, 0.5)
        self.CR_hist = np.full(self.p.memory_size, 0.5)
        self.mem_idx = 0

    # ---------- Initialization (Eq. 5) ----------
    def _init_population(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        k = self.p.k
        pop = []
        for _ in range(self.p.pop_max):
            idx = np.random.choice(N, k, replace=False)
            centroids = X[idx].copy()
            pop.append(centroids)
        return np.array(pop)  # shape: (P, k, D)

    # ---------- Fuzzy-based scaling factor from entropy (spec) ----------
    def _fuzzy_scale_F(self, H: float, k: int) -> float:
        """
        Map entropy H in [0, H_max=log(k)] to a fuzzy blend of (exploit, balanced, explore) Fs.
        - Low entropy (clusters imbalanced) -> larger F (explore)
        - Mid entropy -> medium F
        - High entropy (well-spread) -> smaller F (exploit)
        """
        H_max = math.log(max(k, 2))
        h = np.clip(H / (H_max + 1e-12), 0.0, 1.0)

        # Triangular memberships
        mu_low = np.clip(1.0 - h * 2.0, 0.0, 1.0)
        mu_mid = np.clip(1.0 - np.abs(h - 0.5) * 2.0, 0.0, 1.0)
        mu_high = np.clip((h - 0.5) * 2.0, 0.0, 1.0)

        F_explore = self.p.F_max
        F_mid = (self.p.F_min + self.p.F_max) / 2.0
        F_exploit = self.p.F_min

        F = (mu_low * F_explore + mu_mid * F_mid + mu_high * F_exploit) / (mu_low + mu_mid + mu_high + 1e-12)

        # Add small jitter and clip
        F += np.random.normal(0, 0.05)
        return float(np.clip(F, self.p.F_min, self.p.F_max))

    def _sample_CR(self) -> float:
        CR = np.random.uniform(self.p.CR_min, self.p.CR_max)
        return float(np.clip(CR, 0.0, 1.0))

    # ---------- Mutation (current-to-pbest/1 with archive) (Eq. 9) ----------
    def _mutate(self, Xi: np.ndarray, Xpbest: np.ndarray, Xr: np.ndarray, Xs: np.ndarray, F: float) -> np.ndarray:
        # V_i = X_i + F (X_pbest - X_i) + F (X_r - X_s)
        return Xi + F * (Xpbest - Xi) + F * (Xr - Xs)

    # ---------- Crossover (binomial) (Eq. 10) ----------
    def _crossover(self, Xi: np.ndarray, Vi: np.ndarray, CR: float) -> np.ndarray:
        K, D = Xi.shape
        trial = Xi.copy()
        size = K * D
        j_rand = np.random.randint(size)
        Xi_flat = Xi.ravel()
        Vi_flat = Vi.ravel()
        Tr_flat = trial.ravel()

        for j in range(size):
            if (np.random.rand() < CR) or (j == j_rand):
                Tr_flat[j] = Vi_flat[j]
            else:
                Tr_flat[j] = Xi_flat[j]
        return Tr_flat.reshape(K, D)

    def run(self, X: np.ndarray) -> np.ndarray:
        P_max, P_min = self.p.pop_max, self.p.pop_min
        T_max, eps_div = self.p.T_max, self.p.eps_div
        k = self.p.k

        population = self._init_population(X)               # (P, k, d)
        P = population.shape[0]
        archive: List[np.ndarray] = []                     # A = ∅
        fitness = np.array([sse_objective(X, ind) for ind in population])

        t = 0
        best_idx = int(np.argmin(fitness))
        best = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        while (t < T_max):
            # Entropy of current best centroids
            H_t = compute_entropy_from_centroids(X, best)
            if H_t <= eps_div:
                break

            # Success memories
            suc_F: List[float] = []
            suc_CR: List[float] = []

            # Build union P_t ∪ A for random picks
            if len(archive) > 0:
                union = np.concatenate([population, np.array(archive)], axis=0)
            else:
                union = population

            # Top p% for pbest (Eq. 8)
            pbest_count = max(1, int(self.p.p_best_frac * P))
            order = np.argsort(fitness)
            pbest_set = population[order[:pbest_count]]

            new_pop = []
            new_fit = []
            replaced_indices = []

            # Derived per-iteration adaptive parameters (fuzzy-based F, sampled CR)
            F_iter = self._fuzzy_scale_F(H_t, k)
            CR_iter = self._sample_CR()

            for i in range(P):
                Xi = population[i]
                # pick X_pbest, X_r, X_s
                Xpbest = pbest_set[np.random.randint(0, pbest_set.shape[0])]
                # ensure distinct indices for r, s, i
                choices = list(range(union.shape[0]))
                # We'll tolerate potential i overlap because union != population[i] index spaces;
                # but sample r != s
                r, s = np.random.choice(choices, 2, replace=False)
                Xr, Xs = union[r], union[s]

                # Mutation (Eq. 9)
                Vi = self._mutate(Xi, Xpbest, Xr, Xs, F_iter)

                # Crossover (Eq. 10)
                Ui = self._crossover(Xi, Vi, CR_iter)

                # Selection (Eq. 11)
                f_Ui = sse_objective(X, Ui)
                if f_Ui < fitness[i]:
                    new_pop.append(Ui)
                    new_fit.append(f_Ui)
                    suc_F.append(F_iter)
                    suc_CR.append(CR_iter)
                    # Archive stores replaced Xi (Eq. 12)
                    archive.append(Xi.copy())
                    replaced_indices.append(i)
                else:
                    new_pop.append(Xi)
                    new_fit.append(fitness[i])

            population = np.array(new_pop)
            fitness = np.array(new_fit)

            # Update success history (mean-of-success)
            if len(suc_F) > 0:
                self.F_hist[self.mem_idx] = np.mean(suc_F)
            if len(suc_CR) > 0:
                self.CR_hist[self.mem_idx] = np.mean(suc_CR)
            self.mem_idx = (self.mem_idx + 1) % self.p.memory_size

            # Maintain a bounded archive (size <= P_max)
            if len(archive) > P_max:
                # remove oldest overflow
                archive = archive[-P_max:]

            # Linear population size reduction (Eq. 13)
            P_target = int(P_max - (t / max(T_max, 1)) * (P_max - P_min))
            if population.shape[0] > P_target:
                ord2 = np.argsort(fitness)
                keep = ord2[:P_target]
                population = population[keep]
                fitness = fitness[keep]
                P = population.shape[0]

            # Track best
            bidx = int(np.argmin(fitness))
            if fitness[bidx] < best_fit:
                best_fit = float(fitness[bidx])
                best = population[bidx].copy()

            t += 1  # (Eq. 14)

        # Return X* = argmin f(X_i)
        return best


# ---------------------------------------------------
# BMO: chemotaxis, swim, memetic weighted update (Eq)
# ---------------------------------------------------

@dataclass
class BMOParams:
    num_bacteria: int = 40
    chemotactic_steps: int = 10
    swim_length: int = 6
    reproduction_steps: int = 3
    step_scale: float = 0.05
    ed_prob: float = 0.05
    memetic_steps: int = 2
    outlier_eps: float = 1e-3
    random_state: Optional[int] = 123


class BMO:
    """
    Implements:
      - Initialize B around X* (from LSHADE)
      - Chemotaxis: b_j <- b_j + eta * d_j,  d_j ~ N(0, 1)         (Eq. 15)
      - Swim: continue move while improving; else tumble
      - Memetic local search with weighted updates (Eq. 16)
    """

    def __init__(self, params: BMOParams):
        self.p = params
        if self.p.random_state is not None:
            np.random.seed(self.p.random_state)

    def _init_population(self, X: np.ndarray, X_star: np.ndarray) -> np.ndarray:
        K, D = X_star.shape
        data_min, data_max = X.min(axis=0), X.max(axis=0)
        scale = (data_max - data_min + 1e-12)
        pop = []
        for _ in range(self.p.num_bacteria):
            jitter = np.random.normal(0.0, self.p.step_scale, size=(K, D)) * scale
            cand = X_star + jitter
            cand = np.clip(cand, data_min, data_max)
            pop.append(cand)
        return np.array(pop)

    def _chemotaxis_and_swim(self, X: np.ndarray, start: np.ndarray) -> np.ndarray:
        """
        One chemotaxis (tumble + swim) episode returning improved bacterium.
        """
        data_min, data_max = X.min(axis=0), X.max(axis=0)
        scale = (data_max - data_min + 1e-12)

        def step_once(b: np.ndarray, step_size: float, direction: Optional[np.ndarray] = None) -> np.ndarray:
            K, D = b.shape
            if direction is None:
                direction = np.random.normal(size=(K, D))
                direction /= (np.linalg.norm(direction) + 1e-12)
            cand = b + step_size * direction * scale
            return np.clip(cand, data_min, data_max)

        best = start.copy()
        best_fit = sse_objective(X, best)

        # Tumble
        direction = np.random.normal(size=best.shape)
        direction /= (np.linalg.norm(direction) + 1e-12)

        # Swim while improving
        step = self.p.step_scale
        for _ in range(self.p.swim_length):
            cand = step_once(best, step, direction)
            fit = sse_objective(X, cand)
            if fit < best_fit:
                best, best_fit = cand, fit
                step *= 1.15  # small acceleration
            else:
                break

        return best

    def _memetic_weighted_update(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Weighted centroid update (Eq. 16):
           μ_j = sum w(x_i) x_i / sum w(x_i),
           w(x_i) = exp(- ||x_i - μ_j||^2 / (2 σ^2)).
        Points with w(x_i) < eps are treated as potential outliers (downweighted naturally).
        """
        C = centroids.copy()
        labels = assign_labels(X, C)
        K, D = C.shape
        for j in range(K):
            mask = labels == j
            if not np.any(mask):
                # reseed to random point to avoid empty cluster
                C[j] = X[np.random.randint(0, X.shape[0])]
                continue
            cluster_pts = X[mask]
            # estimate sigma from cluster spread
            if cluster_pts.shape[0] > 1:
                sigma = np.std(cluster_pts, axis=0).mean() + 1e-9
            else:
                sigma = 1.0
            diff = cluster_pts - C[j]
            dist2 = np.sum(diff * diff, axis=1)
            w = np.exp(-dist2 / (2.0 * sigma * sigma))
            # Outlier threshold (points with low weight are naturally reduced)
            w = np.where(w < self.p.outlier_eps, 0.0, w)
            if np.sum(w) > 0:
                C[j] = np.average(cluster_pts, axis=0, weights=w)
            else:
                # fallback to mean
                C[j] = cluster_pts.mean(axis=0)
        return C

    def run(self, X: np.ndarray, X_star: np.ndarray) -> np.ndarray:
        pop = self._init_population(X, X_star)
        fitness = np.array([sse_objective(X, b) for b in pop])

        best_idx = int(np.argmin(fitness))
        gbest = pop[best_idx].copy()
        gbest_fit = float(fitness[best_idx])

        for rep in range(self.p.reproduction_steps):
            for _ in range(self.p.chemotactic_steps):
                for i in range(pop.shape[0]):
                    # Chemotaxis + swim
                    b_new = self._chemotaxis_and_swim(X, pop[i])
                    # Memetic local search
                    for _ in range(self.p.memetic_steps):
                        b_new = self._memetic_weighted_update(X, b_new)
                    f_new = sse_objective(X, b_new)

                    if f_new < fitness[i]:
                        pop[i] = b_new
                        fitness[i] = f_new

                        if f_new < gbest_fit:
                            gbest_fit = f_new
                            gbest = b_new.copy()

            # Reproduction: keep best half and duplicate
            order = np.argsort(fitness)
            half = pop.shape[0] // 2
            survivors = pop[order[:half]]
            survivors_fit = fitness[order[:half]]

            pop[:half] = survivors
            pop[half:] = survivors.copy()
            fitness[:half] = survivors_fit
            fitness[half:] = survivors_fit.copy()

            # Elimination-dispersal
            data_min, data_max = X.min(axis=0), X.max(axis=0)
            scale = (data_max - data_min + 1e-12)
            for i in range(pop.shape[0]):
                if np.random.rand() < self.p.ed_prob:
                    jitter = np.random.normal(0.0, self.p.step_scale, size=gbest.shape) * scale
                    newcomer = np.clip(gbest + jitter, data_min, data_max)
                    pop[i] = newcomer
                    fitness[i] = sse_objective(X, newcomer)

        return gbest


# ----------------------------------------------------------
# HDBSCAN: core distance, mutual reachability, MST (Eqs 18-20)
# (We compute core distances & MST for transparency, then use
#  hdbscan to extract condensed tree + stable clusters.)
# ----------------------------------------------------------

@dataclass
class KHDBSCANParams:
    min_cluster_size: int = 15
    min_samples: int = 5
    metric: str = "euclidean"
    reassign_noise_to_kmeans: bool = True
    random_state: Optional[int] = 7


def compute_core_distances(X: np.ndarray, s: int, metric: str = "euclidean") -> np.ndarray:
    """
    core_k(x_i) = distance to s-th nearest neighbour                    (Eq. 18)
    """
    s = max(1, int(s))
    nn = NearestNeighbors(n_neighbors=s + 1, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)  # includes self at column 0
    core = dists[:, s]  # s-th neighbor distance
    return core


def compute_mutual_reachability(X: np.ndarray, core: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    d_mreach(x_i, x_j) = max(core_k(x_i), core_k(x_j), ||x_i - x_j||)  (Eq. 19)
    Returns dense matrix (may be large for big N; for large data rely directly on hdbscan).
    """
    # We'll use pairwise distances in batches to avoid memory blow-up for very large N
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, metric=metric)
    N = X.shape[0]
    M = np.maximum(D, core.reshape(N, 1))
    M = np.maximum(M, core.reshape(1, N))
    return M


def run_khdbscan_with_explanation(X: np.ndarray,
                                  init_centroids: np.ndarray,
                                  params: KHDBSCANParams) -> np.ndarray:
    """
    (1) KMeans init with provided centroids
    (2) Compute core distances & mutual reachability + MST (explanatory)
    (3) Use hdbscan to extract condensed tree + stable clusters (Eq. 20)
    (4) Optionally reassign noise to nearest KMeans cluster
    """
    k = init_centroids.shape[0]
    km = KMeans(n_clusters=k, init=init_centroids, n_init=1, random_state=params.random_state)
    km.fit(X)
    km_labels = km.labels_

    # Core distances and mutual reachability (for transparency; not strictly needed to run HDBSCAN lib)
    core = compute_core_distances(X, s=params.min_samples, metric=params.metric)
    M = compute_mutual_reachability(X, core, metric=params.metric)
    # Minimum Spanning Tree over mutual reachability graph
    _mst = minimum_spanning_tree(M)  # sparse CSR

    # HDBSCAN clustering proper: condensed tree & stability maximization (Eq. 20)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=params.min_cluster_size,
                                min_samples=params.min_samples,
                                metric=params.metric,
                                prediction_data=False)
    clusterer.fit(X)
    labels = clusterer.labels_

    # Optionally reassign noise (-1) to nearest KMeans cluster
    if params.reassign_noise_to_kmeans and np.any(labels == -1):
        noise_idx = np.where(labels == -1)[0]
        labels[noise_idx] = km_labels[noise_idx]

    return labels


# --------------------------------------
# Full pipeline wrapper (LS → BMO → KHDB)
# --------------------------------------

@dataclass
class PipelineParams:
    lshade: LSHADEParams
    bmo: BMOParams
    khdb: KHDBSCANParams


def ls_bmo_hdbscan(X: np.ndarray, pipe: PipelineParams) -> Dict[str, object]:
    """
    Returns:
      {
        "centroids_lshade": ...,
        "centroids_bmo": ...,
        "labels": ...,
        "metrics": {"silhouette": ..., "dbi": ...}
      }
    """
    # 1) LSHADE
    lshade = LSHADE(pipe.lshade)
    X_star = lshade.run(X)

    # 2) BMO refine
    bmo = BMO(pipe.bmo)
    C_star = bmo.run(X, X_star)

    # 3) K-HDBSCAN
    labels = run_khdbscan_with_explanation(X, C_star, pipe.khdb)

    # Basic internal metrics (if >1 cluster)
    metrics = {}
    if len(set(labels)) > 1:
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["silhouette"] = None
        try:
            metrics["dbi"] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics["dbi"] = None
    else:
        metrics["silhouette"] = None
        metrics["dbi"] = None

    return {
        "centroids_lshade": X_star,
        "centroids_bmo": C_star,
        "labels": labels,
        "metrics": metrics,
    }


# -----------------
# Minimal run demo
# -----------------
if __name__ == "__main__":
    # Small synthetic demo
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=800, centers=4, n_features=2, cluster_std=0.80, random_state=10)

    pipe = PipelineParams(
        lshade=LSHADEParams(k=4, pop_max=60, pop_min=15, T_max=120, p_best_frac=0.2, random_state=1),
        bmo=BMOParams(num_bacteria=30, chemotactic_steps=8, swim_length=5, reproduction_steps=2,
                      step_scale=0.07, memetic_steps=2, random_state=2),
        khdb=KHDBSCANParams(min_cluster_size=20, min_samples=5, reassign_noise_to_kmeans=True, random_state=3)
    )

    out = ls_bmo_hdbscan(X, pipe)
    print("\n--- Pipeline Results ---")
    print("Centroids (LSHADE):\n", out["centroids_lshade"])
    print("\nCentroids (BMO refined):\n", out["centroids_bmo"])
    print("\nLabels (unique):", np.unique(out["labels"]))
    print("Metrics:", out["metrics"])
