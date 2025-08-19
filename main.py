import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ls_shade import LS_SHADE
from bmo import BMO
from khdbscan import run_khdbscan
from utils import evaluate_clustering, plot_clusters


def main(args):
    # Step 1: Load dataset
    print(f"[INFO] Loading dataset: {args.dataset}")
    data = pd.read_csv(args.dataset)
    X = data.values

    # Step 2: Run L-SHADE (Global Optimization)
    print("[INFO] Running L-SHADE optimization...")
    ls = LS_SHADE(pop_size=150, max_gen=500)
    centroids = ls.run(X, num_clusters=args.k)
    print("[INFO] L-SHADE completed.")

    # Step 3: Run BMO (Local Exploitation)
    print("[INFO] Running BMO refinement...")
    bmo = BMO(num_bacteria=50, swim_length=6)
    refined_centroids = bmo.run(X, centroids)
    print("[INFO] BMO refinement completed.")

    # Step 4: Run K-HDBSCAN (Density-based clustering)
    print("[INFO] Running K-HDBSCAN clustering...")
    labels = run_khdbscan(X, refined_centroids,
                          min_cluster_size=args.min_cluster_size,
                          min_samples=args.min_samples)
    print("[INFO] K-HDBSCAN clustering completed.")

    # Step 5: Evaluate results
    print("[INFO] Evaluating clustering results...")
    evaluate_clustering(X, labels)

    # Step 6: Visualization
    if args.plot:
        print("[INFO] Plotting clusters...")
        plot_clusters(X, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LS-BMO-HDBSCAN clustering")

    # Input arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--k", type=int, default=3, help="Approximate number of clusters for initialization")
    parser.add_argument("--min_cluster_size", type=int, default=15, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for core point detection")
    parser.add_argument("--plot", action="store_true", help="Enable visualization of clusters")

    args = parser.parse_args()
    main(args)
