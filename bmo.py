from __future__ import annotations
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def sse_fitness(X: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of Squared Errors (SSE) for clustering assignment."""
    _, dists = pairwise_distances_argmin_min(X, centroids)
    return float(np.sum(dists ** 2))


def clamp_centroids(centroids: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    """Keep centroids within the data hyper-rectangle."""
    return np.clip(centroids, data_min, data_max)


def memetic_local_refine(X: np.ndarray, centroids: np.ndarray, steps: int = 2) -> np.ndarray:
    """
    Lightweight k-means-style refinement:
    - Assign points to nearest centroid
    - Recompute centroids
    - Repeat for `steps`
    This is intentionally tiny to keep BMO fast.
    """
    C = centroids.copy()
    for _ in range(steps):
        labels = assign_labels(X, C)
        # Recompute; if a cluster becomes empty, nudge the centroid to a random point
        for k in range(C.shape[0]):
            mask = labels == k
            if np.any(mask):
                C[k] = X[mask].mean(axis=0)
            else:
                # re-seed empty cluster at a random data point
                C[k] = X[np.random.randint(0, X.shape[0])]
    return C


def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment."""
    labels, _ = pairwise_distances_argmin_min(X, centroids)
    return labels


class BMO:
    """
    Bacterial Memetic Optimization for centroid refinement.

    Each bacterium encodes a full centroid set (K x D).
    Algorithm loop:
        for epoch in range(chemotactic_steps):
            for each bacterium:
                tumble -> evaluate
                swim while improving (up to swim_length)
                memetic local refinement (tiny k-means)
            reproduction (keep best half, duplicate)
            elimination-dispersal (with probability ed_prob)

    Parameters
    ----------
    num_bacteria : int
        Population size (number of candidate centroid sets).
    swim_length : int
        Max consecutive swim steps if improvement continues.
    chemotactic_steps : int
        Number of chemotaxis iterations per reproduction cycle.
    reproduction_steps : int
        Number of reproduction cycles.
    step_scale : float
        Base step size scale for tumbles (relative to data range).
    ed_prob : float
        Elimination-dispersal probability after each reproduction step.
    memetic_steps : int
        Number of tiny k-means updates per chemotactic iteration.
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_bacteria: int = 50,
        swim_length: int = 6,
        chemotactic_steps: int = 12,
        reproduction_steps: int = 4,
        step_scale: float = 0.05,
        ed_prob: float = 0.05,
        memetic_steps: int = 1,
        random_state: int | None = None,
    ):
        self.num_bacteria = int(num_bacteria)
        self.swim_length = int(swim_length)
        self.chemotactic_steps = int(chemotactic_steps)
        self.reproduction_steps = int(reproduction_steps)
        self.step_scale = float(step_scale)
        self.ed_prob = float(ed_prob)
        self.memetic_steps = int(memetic_steps)
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def _init_population(self, X: np.ndarray, init_centroids: np.ndarray) -> np.ndarray:
        """
        Initialize population by jittering the provided centroids.
        Shape: (num_bacteria, K, D)
        """
        K, D = init_centroids.shape
        data_min, data_max = X.min(axis=0), X.max(axis=0)
        scale = (data_max - data_min + 1e-12)
        pop = np.empty((self.num_bacteria, K, D), dtype=float)
        for i in range(self.num_bacteria):
            jitter = np.random.normal(loc=0.0, scale=self.step_scale, size=(K, D)) * scale
            cand = init_centroids + jitter
            pop[i] = clamp_centroids(cand, data_min, data_max)
        return pop

    def _tumble(self, bacterium: np.ndarray, data_min: np.ndarray, data_max: np.ndarray, step_scale: float) -> np.ndarray:
        """
        Random directional move (Gaussian), scaled by data range.
        """
        K, D = bacterium.shape
        scale = (data_max - data_min + 1e-12)
        direction = np.random.normal(size=(K, D))
        direction /= (np.linalg.norm(direction) + 1e-12)  # unit direction
        step = np.random.normal(loc=step_scale, scale=step_scale * 0.3)  # slightly noisy step
        candidate = bacterium + step * direction * scale
        return clamp_centroids(candidate, data_min, data_max)

    def _swim(self, X: np.ndarray, start: np.ndarray, start_fit: float,
              data_min: np.ndarray, data_max: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Swim in the same tumble direction while improving fitness.
        """
        best = start
        best_fit = start_fit

        # Estimate the last move direction; if we don't have it, generate a fresh direction
        K, D = start.shape
        scale = (data_max - data_min + 1e-12)
        direction = np.random.normal(size=(K, D))
        direction /= (np.linalg.norm(direction) + 1e-12)

        step = self.step_scale
        for _ in range(self.swim_length):
            candidate = best + step * direction * scale
            candidate = clamp_centroids(candidate, data_min, data_max)
            fit = sse_fitness(X, candidate)
            if fit < best_fit:
                best, best_fit = candidate, fit
                # slight step growth to ride the slope
                step *= 1.1
            else:
                break
        return best, best_fit

    def run(self, X: np.ndarray, init_centroids: np.ndarray) -> np.ndarray:
        """
        Execute BMO to refine the initial centroids.
        Returns the best refined centroids (K x D).
        """
        assert X.ndim == 2
        assert init_centroids.ndim == 2
        K, D = init_centroids.shape

        data_min, data_max = X.min(axis=0), X.max(axis=0)

        # Initialize population around given centroids
        population = self._init_population(X, init_centroids)
        fitness = np.array([sse_fitness(X, c) for c in population])

        best_idx = int(np.argmin(fitness))
        global_best = population[best_idx].copy()
        global_best_fit = float(fitness[best_idx])

        for rep in range(self.reproduction_steps):
            # Chemotaxis
            for chem in range(self.chemotactic_steps):
                for i in range(self.num_bacteria):
                    bacterium = population[i]

                    # Tumble
                    candidate = self._tumble(bacterium, data_min, data_max, self.step_scale)
                    cand_fit = sse_fitness(X, candidate)

                    # If improved, swim; else keep original
                    if cand_fit < fitness[i]:
                        candidate, cand_fit = self._swim(X, candidate, cand_fit, data_min, data_max)

                    # Memetic tiny k-means refinement
                    candidate = memetic_local_refine(X, candidate, steps=self.memetic_steps)
                    candidate = clamp_centroids(candidate, data_min, data_max)
                    cand_fit = sse_fitness(X, candidate)

                    # Selection
                    if cand_fit < fitness[i]:
                        population[i] = candidate
                        fitness[i] = cand_fit

                # Track global best
                local_best_idx = int(np.argmin(fitness))
                if fitness[local_best_idx] < global_best_fit:
                    global_best_fit = float(fitness[local_best_idx])
                    global_best = population[local_best_idx].copy()

            # Reproduction: keep best half, duplicate
            order = np.argsort(fitness)
            half = self.num_bacteria // 2
            survivors = population[order[:half]]
            survivors_fit = fitness[order[:half]]

            population[:half] = survivors
            population[half:] = survivors.copy()  # duplicate the best
            fitness[:half] = survivors_fit
            fitness[half:] = survivors_fit.copy()

            # Elimination-Dispersal
            for i in range(self.num_bacteria):
                if np.random.rand() < self.ed_prob:
                    # Re-seed around global best with some noise
                    jitter = np.random.normal(loc=0.0, scale=self.step_scale, size=(K, D)) * (data_max - data_min + 1e-12)
                    newcomer = global_best + jitter
                    newcomer = clamp_centroids(newcomer, data_min, data_max)
                    population[i] = newcomer
                    fitness[i] = sse_fitness(X, newcomer)

            # Optional: slight annealing of step size
            self.step_scale *= 0.95

            print(f"[BMO] Reproduction {rep+1}/{self.reproduction_steps} | Best SSE: {global_best_fit:.4f}")

        return global_best
