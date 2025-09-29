import time
import torch
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
        self.gpu = gpu

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

    def fitness_knn(self, chromosome, k=25, n_samples=2000, n_trials=3):
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

    def evaluate_knn_pytorch(self, batch, k=25, n_samples=5000, n_trials=3):
        X_torch = torch.tensor(self.X, dtype=torch.float32).cuda()
        y_torch = torch.tensor(self.y, dtype=torch.long).cuda()

        all_scores = []

        for trial in range(n_trials):
            idx = torch.randperm(len(X_torch))[:n_samples].cuda()
            X_sampled = X_torch[idx]
            y_sampled = y_torch[idx]

            train_size = int(0.7 * n_samples)
            X_train = X_sampled[:train_size]
            y_train = y_sampled[:train_size]
            X_test = X_sampled[train_size:]
            y_test = y_sampled[train_size:]

            decoded_batch = torch.stack([
                torch.tensor(self.decode(indv['chromosome'])[:-1], dtype=torch.float32).cuda()
                for indv in batch
            ])

            X_train_batch = X_train.unsqueeze(0) * decoded_batch.unsqueeze(1)
            X_test_batch = X_test.unsqueeze(0) * decoded_batch.unsqueeze(1)

            dists = torch.cdist(X_test_batch, X_train_batch)
            _, indices = dists.topk(k, dim=2, largest=False)

            y_train_expanded = y_train.unsqueeze(0).expand(len(batch), -1)
            neighbors = torch.gather(y_train_expanded.unsqueeze(1).expand(-1, len(X_test), -1), 2, indices)
            predictions = neighbors.mode(dim=2).values

            accuracies = (predictions == y_test.unsqueeze(0)).float().mean(dim=1)
            all_scores.append(accuracies.cpu().numpy())

        for i, indv in enumerate(batch):
            indv['fitness'] = float(np.mean([scores[i] for scores in all_scores]))

    def evolve(self):
        print("Evolution starting:")

        if self.gpu:
            self.evaluate_knn_pytorch(self.population)
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
                self.evaluate_knn_pytorch(offspring)
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
        weights = (chromosome[:-1] > chromosome[-1]).astype(np.float32)
        return np.append(weights, chromosome[-1])


class StochasticDecodingGA(BaseGA):
    """
    This decoding method only uses the value of each allele. During evaluation, features are used by treating the corresponding allele values as probabilities, and aims to naturally split the allele values.
    """
    def decode(self, chromosome):
        weights = (np.random.random(self.n_features) < chromosome[:-1]).astype(np.float32)
        return np.append(weights, chromosome[-1])


class RankingDecodingGA(BaseGA):
    """
    This decoding method uses the allele values and the threshold. The allele values are considered ranking order and features with higher allele values are picked first, with the threshold determining the number of features used.
    """
    def decode(self, chromosome):
        threshold = chromosome[-1]
        n_select = int(self.n_features * threshold)
        weights = np.zeros(self.n_features, dtype=np.float32)
        if n_select > 0:
            weights[np.argsort(chromosome[:-1])[-n_select:]] = 1.0
        return np.append(weights, threshold)


class WeightedFeaturesGA(BaseGA):
    """
    This decoding method only uses the allele values. All features are used during evaluation, but they are scaled by their corresponding allele values as an importance scaler, fading the noise features with near 0 genomes.
    """
    def decode(self, chromosome):
        return chromosome.astype(np.float32)
