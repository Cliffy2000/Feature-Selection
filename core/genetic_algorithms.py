import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

try:
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class NumericFeaturesGA:
    """
    DEPRECATED
    """
    def __init__(self, X, y, population_size, generations, elitism_ratio, crossover_rate, mutation_rate, fitness_evaluator):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_select = self.n_features / 2
        self.population_size = population_size
        self.generations = generations
        self.elitism_count = int(self.population_size * elitism_ratio)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.evaluator = fitness_evaluator

        self.population = self.initialize_population()
        self.history = []

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.randint(-1, self.n_features, size=self.n_features)
            population.append({
                'chromosome': chromosome,
                'fitness': 0
            })

        return population

    def evaluate_population(self):
        for indv in self.population:
            indv['fitness'] = self.evaluator(indv['chromosome'])

    def evaluate_batch(self, batch):
        for indv in batch:
            indv['fitness'] = self.evaluator(indv['chromosome'])

    def select(self):
        fitness_values = np.array([indv['fitness'] for indv in self.population])
        probabilities = fitness_values / np.sum(fitness_values)
        selected_indices = np.random.choice(len(self.population), size=2, p=probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.n_features)

            child1_chromosome = np.concatenate([
                parent1['chromosome'][:crossover_point],
                parent2['chromosome'][crossover_point:]
            ])
            child2_chromosome = np.concatenate([
                parent2['chromosome'][:crossover_point],
                parent1['chromosome'][crossover_point:]
            ])

            child1 = {'chromosome': child1_chromosome, 'fitness': 0}
            child2 = {'chromosome': child2_chromosome, 'fitness': 0}
        else:
            child1 = {'chromosome': parent1['chromosome'].copy(), 'fitness': 0}
            child2 = {'chromosome': parent2['chromosome'].copy(), 'fitness': 0}

        return child1, child2

    def mutate(self, indv):
        for i in range(len(indv['chromosome'])):
            if np.random.random() < self.mutation_rate:
                indv['chromosome'][i] = np.random.randint(-1, self.n_features)
        return indv

    def evolve(self):
        self.evaluate_population()

        for generation in range(self.generations):
            new_population = self.population[:self.elitism_count]
            new_indvs = []

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select()
                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_indvs.extend([child1, child2])

            self.evaluate_batch(new_indvs)
            new_population.extend(new_indvs)
            self.population = new_population[:self.population_size]

            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            self.history.append(self.population[0])
            print(f"Generation {generation}: Best fitness = {self.population[0]['fitness']:.4f}")


class BaseGA:
    def __init__(self, X, y, population_size, generations, elitism_ratio, crossover_rate, mutation_rate, gpu=False):
        # dataset params
        self.X = X
        self.y = y
        self.n_features = X.shape[1]

        # GA hyper-parameters
        self.population_size = population_size
        self.generations = generations
        self.elitism_ratio = elitism_ratio
        self.elitism_count = int(self.population_size * self.elitism_ratio)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # GA variables
        self.population = self.initialize_population()
        self.history = []

        # Env Settings
        self.gpu = False

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.rand(self.n_features + 1)    # additional value for threshold
            population.append({
                'chromosome': chromosome,
                'fitness': 0
            })

        return population

    def select(self):
        """
        Roulette wheel selection
        :return: parent1, parent2
        """
        fitness_values = np.array([indv['fitness'] for indv in self.population])
        probabilities = fitness_values / np.sum(fitness_values)
        selected_indices = np.random.choice(len(self.population), size=2, p=probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        """
        One point crossover
        :return: child1, child2 as new individual objects in the population
        """
        if np.random.random() >= self.crossover_rate:
            return {'chromosome': parent1['chromosome'].copy(), 'fitness': 0}, \
                {'chromosome': parent2['chromosome'].copy(), 'fitness': 0}

        point = np.random.randint(1, len(parent1['chromosome']))
        child1_chr = np.concatenate([parent1['chromosome'][:point], parent2['chromosome'][point:]])
        child2_chr = np.concatenate([parent2['chromosome'][:point], parent1['chromosome'][point:]])

        return {'chromosome': child1_chr, 'fitness': 0}, {'chromosome': child2_chr, 'fitness': 0}

    def mutate(self, indv):
        """
        Randomly generated new allele value for feature and adds random normal value for the threshold
        """
        mask = np.random.random(len(indv['chromosome'])) < self.mutation_rate

        # Features: complete random reset when mutating
        for i in range(self.n_features):
            if mask[i]:
                indv['chromosome'][i] = np.random.random()

        # Threshold: larger gaussian perturbation
        if mask[-1]:
            indv['chromosome'][-1] += np.random.normal(0, 0.05)
            indv['chromosome'][-1] = np.clip(indv['chromosome'][-1], 0, 1)

        return indv

    def decode(self, chromosome):
        """
        Returns either boolean mask or weight array for features, threshold should be kept in the chromosome
        """
        raise NotImplementedError("Each GA variant must implement decode()")

    def fitness_knn(self, chromosome, k=5, n_samples=2000, n_trials=3):
        decoded = self.decode(chromosome)

        if decoded.dtype == bool:
            X_transformed = self.X[:, decoded[:-1]]
            if X_transformed.shape[1] == 0:
                return 0.0
        else:
            X_transformed = self.X * decoded[:-1]

        scores = []
        for _ in range(n_trials):
            idx = np.random.choice(len(self.X), n_samples, replace=True)
            X_sample = X_transformed[idx]
            y_sample = self.y[idx]

            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=None
            )

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            scores.append(knn.score(X_test, y_test))

        return np.mean(scores)

    def evaluate_batch_gpu(self, batch):
        print(f"GPU batch evaluation starting for {len(batch)} individuals...")
        start = time.time()

        # Move data to GPU once
        X_gpu = cp.asarray(self.X, dtype=cp.float32)
        y_gpu = cp.asarray(self.y, dtype=cp.int32)
        print(f"Data moved to GPU: {time.time() - start:.2f}s")

        # Prepare all chromosomes
        all_decoded = [self.decode(indv['chromosome']) for indv in batch]

        # Create GPU streams for parallel execution
        streams = [cp.cuda.Stream() for _ in batch]
        fitness_scores = []

        for decoded, stream in zip(all_decoded, streams):
            with stream:
                if decoded.dtype == bool:
                    X_transformed = X_gpu[:, decoded[:-1]]
                    if X_transformed.shape[1] == 0:
                        fitness_scores.append(0.0)
                        continue
                else:
                    X_transformed = X_gpu * decoded[:-1]

                scores = []
                for _ in range(3):
                    idx = cp.random.choice(len(X_gpu), 5000, replace=True)
                    X_sample = cp.asnumpy(X_transformed[idx])
                    y_sample = cp.asnumpy(y_gpu[idx])

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_sample, y_sample, test_size=0.3, random_state=None
                    )

                    knn = cuKNN(n_neighbors=5)
                    knn.fit(X_train, y_train)
                    scores.append(float(knn.score(X_test, y_test)))

                fitness_scores.append(np.mean(scores))

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()

        # Assign fitness values
        for indv, fitness in zip(batch, fitness_scores):
            indv['fitness'] = fitness

    def evolve(self):
        print("Evolution starting:")

        if self.gpu:
            self.evaluate_batch_gpu(self.population)
        else:
            for indv in self.population:
                indv['fitness'] = self.fitness_knn(indv['chromosome'])

        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        print("Initial population completed.")

        for generation in range(self.generations):
            new_population = self.population[:self.elitism_count].copy()

            offspring = []
            while len(new_population) + len(offspring) < self.population_size:
                parent1, parent2 = self.select()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])

            if self.gpu:
                self.evaluate_batch_gpu(offspring)
            else:
                for indv in offspring:
                    indv['fitness'] = self.fitness_knn(indv['chromosome'])

            new_population.extend(offspring)
            new_population = new_population[:self.population_size]

            new_population.sort(key=lambda x: x['fitness'], reverse=True)
            self.population = new_population

            best = self.population[0]
            fitness_values = [indv['fitness'] for indv in self.population]
            chromosomes = np.array([indv['chromosome'] for indv in self.population])
            self.history.append({
                'generation': generation,
                'best_fitness': fitness_values[0],
                'worst_fitness': fitness_values[-1],
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'diversity': np.mean(np.std(chromosomes, axis=0)),
                'best_chromosome': self.population[0]['chromosome'].copy()
            })

            if generation % 5 == 0:
                print(f"Gen {generation}: Best = {best['fitness']:.4f}")

            if fitness_values[0] > 0.999:
                print(f"Early stop at generation {generation}: Perfect fitness achieved")
                break

        print(f"\nEvolution completed at generation {self.history[-1]['generation']}")
        print(f"Final best fitness: {self.population[0]['fitness']:.4f}")
        print(f"Final best individual: {self.population[0]['chromosome']}")

        return self.population[0]



class ThresholdDecodingGA(BaseGA):
    """
    This decoding method considers the value of each allele and the threshold. Features are turned on if their corresponding allele has a value larger than the threshold.
    """
    def decode(self, chromosome):
        threshold = chromosome[-1]
        mask = chromosome[:-1] > threshold
        return np.append(mask, threshold)


class StochasticDecodingGA(BaseGA):
    """
    This decoding method only uses the value of each allele. During evaluation, features are used by treating the corresponding allele values as probabilities, and aims to naturally split the allele values.
    """
    def decode(self, chromosome):
        probabilities = chromosome[:-1]
        sampled = np.random.random(len(probabilities)) < probabilities
        return np.append(sampled, chromosome[-1])


class RankingDecodingGA(BaseGA):
    """
    This decoding method uses the allele values and the threshold. The allele values are considered ranking order and features with higher allele values are picked first, with the threshold determining the number of features used.
    """
    def decode(self, chromosome):
        threshold = chromosome[-1]
        n_select = int(self.n_features * threshold)

        if n_select == 0:
            mask = np.zeros(self.n_features, dtype=bool)
        else:
            top_indices = np.argsort(chromosome[:-1])[-n_select:]
            mask = np.zeros(self.n_features, dtype=bool)
            mask[top_indices] = True

        return np.append(mask, threshold)


class WeightedFeaturesGA(BaseGA):
    """
    This decoding method only uses the allele values. All features are used during evaluation, but they are scaled by their corresponding allele values as an importance scaler, fading the noise features with near 0 genomes.
    """
    def decode(self, chromosome):
        weights = chromosome[:-1]
        return np.append(weights, chromosome[-1])
