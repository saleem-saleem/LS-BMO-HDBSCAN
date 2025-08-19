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
  ## Steps to Run LS-BMO-HDBSCAN

<p> 1. Clone the Repository
<br>If you already pushed your repo to GitHub:</br>
<br>git clone https://github.com/YOUR_USERNAME/LS-BMO-HDBSCAN.git</br>
<br>cd LS-BMO-HDBSCAN</br></p>

<p> 2. Create Virtual Environment (Recommended)

<br>Keeps dependencies clean:</br>
<br>python -m venv venv</br>
<br>source venv/bin/activate   # for Linux/Mac</br>
<br>venv\Scripts\activate      # for Windows</br></p>

<p> 3. Install Dependencies
<br>Install all required Python libraries</br>
<br>pip install -r requirements.txt</br></p>
<p> 4. Prepare Dataset
<br>Put your dataset in the data/ folder (e.g., data/sample.csv)</br>
<br>Ensure it’s in CSV format with rows = samples, columns = feature</br>
(No labels needed since it’s unsupervised clustering).</br></p>
<p> 5. Run the Algorithm
<br>Run from terminal:</br>
<br>python main.py --dataset data/sample.csv --min_cluster_size 15 --min_samples 5</br>
<br>Arguments</br>
<br>--dataset → Path to dataset</br>
<br>--min_cluster_size → Minimum cluster size for HDBSCAN (default=15)</br>
<br>--min_samples → Minimum samples for core points (default=5)</br></p>
<p> 6. Outputs
<br>After execution, you’ll get:</br>
<br>Cluster labels for each point</br>
<br>Evaluation metrics (Silhouette, DBI, ARI, Jaccard, SSE, Beta Index)</br>
<br>Plots (cluster visualization using PCA or t-SNE)</br>
<br>Results saved in results/ folder</p></br></p>







