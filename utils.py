import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, jaccard_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, y_true: np.ndarray | None = None) -> dict:
    """
    Evaluate clustering performance using internal & external metrics.

    Parameters
    ----------
    X : np.ndarray
        Dataset (n_samples x n_features).
    labels : np.ndarray
        Cluster labels predicted (-1 = noise).
    y_true : np.ndarray or None
        Ground truth labels (optional).

    Returns
    -------
    results : dict
        Dictionary of evaluation scores.
    """
    results = {}

    # Internal metrics
    if len(set(labels)) > 1:
        results['Silhouette'] = silhouette_score(X, labels)
        results['DBI'] = davies_bouldin_score(X, labels)
    else:
        results['Silhouette'] = -1
        results['DBI'] = np.inf

    # External metrics (if ground truth provided)
    if y_true is not None:
        results['ARI'] = adjusted_rand_score(y_true, labels)
        try:
            results['Jaccard'] = jaccard_score(y_true, labels, average="macro")
        except Exception:
            results['Jaccard'] = None

    # Print results
    print("\n[Evaluation Results]")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")

    return results


def plot_clusters(X: np.ndarray, labels: np.ndarray, method: str = "pca") -> None:
    """
    Plot clusters using PCA or t-SNE.

    Parameters
    ----------
    X : np.ndarray
        Dataset (n_samples x n_features).
    labels : np.ndarray
        Cluster labels (-1 = noise).
    method : str
        Dimensionality reduction method ('pca' or 'tsne').
    """
    if X.shape[1] > 2:
        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        X_red = reducer.fit_transform(X)
    else:
        X_red = X

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for color, label in zip(colors, unique_labels):
        mask = labels == label
        if label == -1:
            plt.scatter(X_red[mask, 0], X_red[mask, 1], c="k", marker="x", s=40, label="Noise")
        else:
            plt.scatter(X_red[mask, 0], X_red[mask, 1], c=[color], s=40, label=f"Cluster {label}")

    plt.title(f"Cluster Visualization ({method.upper()})")
    plt.legend()
    plt.show()


def save_labels(labels: np.ndarray, filename: str = "results/labels.csv") -> None:
    """
    Save cluster labels to CSV file.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels array.
    filename : str
        Output file path.
    """
    import os
    import pandas as pd

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame({"Cluster": labels}).to_csv(filename, index=False)
    print(f"[INFO] Labels saved to {filename}")
