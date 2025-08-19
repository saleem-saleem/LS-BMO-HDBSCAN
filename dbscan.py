import numpy as np
from sklearn.cluster import KMeans
import hdbscan


def run_khdbscan(X: np.ndarray,
                 init_centroids: np.ndarray,
                 min_cluster_size: int = 15,
                 min_samples: int = 5,
                 metric: str = "euclidean") -> np.ndarray:
    """
    Run K-means initialized HDBSCAN clustering.

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n_samples, n_features).
    init_centroids : np.ndarray
        Initial centroids (K x D) from LS-SHADE / BMO.
    min_cluster_size : int
        Minimum size of clusters in HDBSCAN.
    min_samples : int
        Controls how conservative HDBSCAN is (higher = more conservative).
    metric : str
        Distance metric (default: 'euclidean').

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (-1 means noise).
    """
    n_clusters = init_centroids.shape[0]

    # Step 1: KMeans with given centroids
    print(f"[K-HDBSCAN] Running KMeans initialization with k={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, random_state=42)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_

    # Step 2: HDBSCAN clustering
    print("[K-HDBSCAN] Running HDBSCAN refinement...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=metric)
    clusterer.fit(X)

    labels = clusterer.labels_

    # Step 3: Handle noise (-1 labels)
    if np.any(labels == -1):
        print(f"[K-HDBSCAN] Found {np.sum(labels == -1)} noise points, reassigning to nearest KMeans cluster...")
        noise_idx = np.where(labels == -1)[0]
        for idx in noise_idx:
            labels[idx] = kmeans_labels[idx]

    print(f"[K-HDBSCAN] Final clusters: {len(set(labels))}")
    return labels
