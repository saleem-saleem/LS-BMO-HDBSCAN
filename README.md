# LS-BMO-HDBSCAN
<p>This repository provides the implementation of LS-BMO-HDBSCAN, a novel hybrid clustering framework that integrates L-SHADE (Success-History based Adaptive Differential Evolution with Linear Population Reduction), Bacterial Memetic Optimization (BMO), and K-means initialized HDBSCAN. </p>   
<p> It is designed to improve clustering accuracy, noise resilience and convergence speed for high-dimensional, sparse, and noisy datasets</p>

## Key Features
- **L-SHADE**: Provides adaptive mutation control, dynamic population reduction, and fast convergence.  
- **BMO (Bacterial Memetic Optimization)**: Balances exploration and exploitation with memetic local learning.  
- **K-means initialized HDBSCAN**: Performs density-based clustering, detects arbitrary-shaped clusters, and handles noise/outliers.  
- **Multi-objective Optimization**: Minimizes Sum of Squared Errors (SSE), Davies-Bouldin Index, and maximizes Silhouette Score.  
- **Scalability & Robustness**: Works efficiently with large and complex real-world datasets (healthcare, bioinformatics, education, etc.).

## Evaluation Metrics

- **Davies-Bouldin Index**
- **Adjusted Rand Index**
- **Jaccard Index**
- **Sum of Squared Errors (SSE)**
- **Beta Index**


## Benchmark Datasets

- **Artificial datasets (D1, D2)**
- **Balance Scale**
- **CMC**
- **Car Evaluation**
- **Air Quality**
- **Breast Cancer**
- **Energy Efficiency**
- **Wine**
- **Steel Plate**
- **Yeast**

## Usage
Run clustering on a dataset:
<p> python main.py --dataset data/sample.csv --min_cluster_size 15 --min_samples 5 </p>
<p> Arguments:
<br>--dataset: Path to input dataset (CSV format).</br>
<br>--min_cluster_size: Minimum size of clusters (default=15).</br>
<br>--min_samples: Minimum samples for core point detection (default=5).</br></p>






