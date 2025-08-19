import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


class LS_SHADE:
    def __init__(self, pop_size=150, max_gen=500, memory_size=20):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.memory_size = memory_size
        self.F_history = np.full(memory_size, 0.5)  # scaling factor memory
        self.CR_history = np.full(memory_size, 0.5)  # crossover rate memory

    def initialize_population(self, X, num_clusters):
        """Initialize population of candidate centroids"""
        n_samples, n_features = X.shape
        population = []
        for _ in range(self.pop_size):
            indices = np.random.choice(n_samples, num_clusters, replace=False)
            centroids = X[indices]
            population.append(centroids)
        return np.array(population)

    def fitness(self, X, centroids):
        """Objective function: Sum of Squared Errors (SSE)"""
        labels, distances = pairwise_distances_argmin_min(X, centroids)
        return np.sum(distances ** 2)

    def mutate(self, population, idx, F):
        """DE/rand/1 mutation strategy"""
        ids = list(range(len(population)))
        ids.remove(idx)
        a, b, c = population[np.random.choice(ids, 3, replace=False)]
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant, CR):
        """Binomial crossover"""
        trial = np.copy(target)
        n_features = target.shape[0] * target.shape[1]
        rand_j = np.random.randint(n_features)
        for j in range(n_features):
            if np.random.rand() < CR or j == rand_j:
                trial.ravel()[j] = mutant.ravel()[j]
        return trial

    def run(self, X, num_clusters=3):
        """Run L-SHADE optimization"""
        n_samples, n_features = X.shape
        population = self.initialize_population(X, num_clusters)
        fitness_vals = np.array([self.fitness(X, ind) for ind in population])

        memory_index = 0

        for gen in range(1, self.max_gen + 1):
            new_population = []
            new_fitness = []

            for i in range(len(population)):
                # Adaptive parameters
                F = np.clip(np.random.normal(self.F_history[memory_index], 0.1), 0, 1)
                CR = np.clip(np.random.normal(self.CR_history[memory_index], 0.1), 0, 1)

                # Mutation + Crossover
                mutant = self.mutate(population, i, F)
                trial = self.crossover(population[i], mutant, CR)

                # Evaluate
                trial_fit = self.fitness(X, trial)

                # Selection
                if trial_fit < fitness_vals[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fit)
                    self.F_history[memory_index] = (self.F_history[memory_index] + F) / 2
                    self.CR_history[memory_index] = (self.CR_history[memory_index] + CR) / 2
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness_vals[i])

            # Update population
            population = np.array(new_population)
            fitness_vals = np.array(new_fitness)

            # Population size reduction (Linear)
            new_size = int(self.pop_size - (self.pop_size - 20) * (gen / self.max_gen))
            if len(population) > new_size:
                sorted_idx = np.argsort(fitness_vals)
                population = population[sorted_idx[:new_size]]
                fitness_vals = fitness_vals[sorted_idx[:new_size]]

            memory_index = (memory_index + 1) % self.memory_size

            if gen % 50 == 0:
                print(f"[GEN {gen}] Best SSE: {fitness_vals.min():.4f}")

        # Return best solution
        best_idx = np.argmin(fitness_vals)
        return population[best_idx]
