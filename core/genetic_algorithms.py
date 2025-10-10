import random
import time
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool, cpu_count


class BaseGA:
    def __init__(self, X, y, population_size, generations, elitism_ratio, crossover_rate, mutation_rate, knn_k=5, gpu=False):
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
        self.knn_k = knn_k

        # GA variables
        self.population = self.initialize_population()
        self.history = []

        # Env Settings
        self.gpu = gpu

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.rand(self.n_features + 1)  # additional value for threshold
            population.append({
                'chromosome': chromosome,
                'fitness': 0
            })

        return population

    def select_roulette(self):
        """
        Roulette wheel selection
        :return: parent1, parent2
        """
        fitness_values = np.array([indv['fitness'] for indv in self.population])
        probabilities = fitness_values / np.sum(fitness_values)
        selected_indices = np.random.choice(len(self.population), size=1, p=probabilities)
        return self.population[selected_indices[0]]

    def select_tourn(self):
        tournament_size = 3
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        winner = max(tournament, key=lambda x: x['fitness'])
        return winner

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

    def fitness_knn(self, chromosome, n_trials=3):
        k = self.knn_k
        decoded = self.decode(chromosome)

        if decoded.dtype == bool:
            X_transformed = self.X[:, decoded[:-1]]
            if X_transformed.shape[1] == 0:
                return 0.0
        else:
            X_transformed = self.X * decoded[:-1]

        scores = []
        for _ in range(n_trials):
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, self.y, test_size=0.25
            )

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            scores.append(knn.score(X_test, y_test))

        return np.mean(scores)

    """
    def evaluate_population_knn_parallel(self, population, k=25, n_samples=70000, n_trials=3):
        def evaluate_single(indv_data):
            indv, X, y, k, n_samples, n_trials = indv_data
            decoded = self.decode(indv['chromosome'])

            if decoded.dtype == bool:
                X_transformed = X[:, decoded[:-1]]
                if X_transformed.shape[1] == 0:
                    return 0.0
            else:
                X_transformed = X * decoded[:-1]

            scores = []
            for _ in range(n_trials):
                idx = np.random.choice(len(X_transformed), n_samples, replace=True)
                X_sample = X_transformed[idx]
                y_sample = y[idx]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_sample, y_sample, test_size=0.3
                )

                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                scores.append(knn.score(X_test, y_test))

            return np.mean(scores)

        with Pool(processes=cpu_count()) as pool:
            args = [(indv, self.X, self.y, k, n_samples, n_trials) for indv in population]
            fitness_values = pool.map(evaluate_single, args)

        for indv, fitness in zip(population, fitness_values):
            indv['fitness'] = fitness

    def evaluate_population_lr(self, batch, n_trials=1):
        # Use full dataset with fixed split(s)
        all_scores = []

        for trial in range(n_trials):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.3, stratify=self.y
            )

            trial_scores = []
            for indv in batch:
                # Decode chromosome to get feature weights
                decoded = self.decode(indv['chromosome'])
                weights = decoded[:-1]

                # Apply feature selection
                X_train_selected = X_train * weights
                X_test_selected = X_test * weights

                # Train logistic regression with L1 penalty (good against noise)
                lr = LogisticRegression(
                    penalty='l1',
                    solver='saga',  # Supports L1
                    C=1.0,  # Regularization strength (lower = more penalty)
                    max_iter=200
                )

                lr.fit(X_train_selected, y_train)
                score = lr.score(X_test_selected, y_test)
                trial_scores.append(score)

            all_scores.append(trial_scores)

        # Average across trials and assign to individuals
        for i, indv in enumerate(batch):
            indv['fitness'] = float(np.mean([scores[i] for scores in all_scores]))
    """

    def evolve(self):
        print("Evolution starting:")

        for indv in self.population:
            indv['fitness'] = self.fitness_knn(indv['chromosome'])

        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        print("Initial population completed.")

        for generation in range(self.generations):
            new_population = []

            # Keep elites but they'll be re-evaluated
            elite_chromosomes = [self.population[i]['chromosome'].copy() for i in range(self.elitism_count)]

            for chromosome in elite_chromosomes:
                new_population.append({'chromosome': chromosome, 'fitness': 0})

            while len(new_population) < self.population_size:
                parent1 = self.select_tourn()
                parent2 = self.select_tourn()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            new_population = new_population[:self.population_size]

            for indv in new_population:
                indv['fitness'] = self.fitness_knn(indv['chromosome'])

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

            '''
            if fitness_values[0] > 0.999:
                print(f"Early stop at generation {generation}: Perfect fitness achieved")
                break
            '''

        print(f"\nEvolution completed at generation {self.history[-1]['generation']}")
        print(f"Final best fitness: {self.population[0]['fitness']:.4f}")
        print(f"Final best individual: {self.population[0]['chromosome']}")

        return self.population[0]


class ThresholdDecodingGA(BaseGA):
    """
    This decoding method considers the value of each allele and the threshold. Features are turned on if their corresponding allele has a value larger than the threshold.
    """

    def decode(self, chromosome):
        weights = chromosome[:-1] > chromosome[-1]
        return np.append(weights, chromosome[-1])


class ThresholdDecodingPenaltyGA(BaseGA):
    """
    This builds on top of the threshold decoding, but adds random chance that the tournament selection aims for least amount of features instead of highest fitness score
    """

    def select_tourn(self):
        tournament_size = 3
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        if random.random() < 0.25:
            winner = min(tournament, key=lambda x: np.sum(x['chromosome'][:-1] > x['chromosome'][-1]))
        else:
            winner = max(tournament, key=lambda x: x['fitness'])
        return winner

    def decode(self, chromosome):
        weights = chromosome[:-1] > chromosome[-1]
        return np.append(weights, chromosome[-1])


class StochasticDecodingGA(BaseGA):
    """
    This decoding method only uses the value of each allele. During evaluation, features are used by treating the corresponding allele values as probabilities, and aims to naturally split the allele values.
    """

    def decode(self, chromosome):
        weights = np.random.random(self.n_features) < chromosome[:-1]
        return np.append(weights, chromosome[-1])


class RankingDecodingGA(BaseGA):
    """
    This decoding method uses the allele values and the threshold. The allele values are considered ranking order and features with higher allele values are picked first, with the threshold determining the number of features used.
    """

    def decode(self, chromosome):
        threshold = chromosome[-1]
        n_select = int(self.n_features * threshold)
        mask = np.zeros(self.n_features, dtype=bool)

        if n_select > 0:
            mask[np.argpartition(chromosome[:-1], -n_select)[-n_select:]] = True

        return np.append(mask, threshold)


class WeightedFeaturesGA(BaseGA):
    """
    This decoding method only uses the allele values. All features are used during evaluation, but they are scaled by their corresponding allele values as an importance scaler, fading the noise features with near 0 genomes.
    """

    def decode(self, chromosome):
        return chromosome.astype(np.float32)
