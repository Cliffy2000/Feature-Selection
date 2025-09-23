import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from core.fitness import *


class NumericFeaturesGA:
    def __init__(self, X, y, population_size, generations, elitism_ratio, crossover_rate, mutation_rate, fitness_evaluator):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        # TODO: potential target feature amount
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
            # TODO: chromosome allele repetition
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

    # TODO: remove order permutation
    # would sorted chromosome affect diversity
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



class ProbabilisticEncodingGA:



class SetEncodingGA:
    

