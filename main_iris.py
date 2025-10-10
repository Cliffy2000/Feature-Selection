import os
import sys
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time

print("Starting GA Feature Selection...")
start = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from data.data_loader import load_clean_iris
from core.genetic_algorithms import *

algorithms = {
    'threshold': ThresholdDecodingGA,
    'thresholdPenalty': ThresholdDecodingPenaltyGA,
    'stochastic': StochasticDecodingGA,
    'ranking': RankingDecodingGA,
    'weighted': WeightedFeaturesGA
}

algorithm_name = sys.argv[1] if len(sys.argv) > 1 else 'thresholdPenalty'

if algorithm_name not in algorithms:
    print(f"Invalid algorithm: {algorithm_name}")
    sys.exit(1)

print(f"Algorithm: {algorithm_name}")
print("Loading data...")

X, y = load_clean_iris()
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Setup time: {time.time()-start:.2f}s")

ga_configs = {
    'population_size': 20,
    'generations': 150,
    'elitism_ratio': 0.05,
    'crossover_rate': 0.7,
    'mutation_rate': 0.2,
    'knn_k': 2,
    'gpu': False
}

print(f"Initializing GA: pop={ga_configs['population_size']}, gens={ga_configs['generations']}")

GA_Class = algorithms[algorithm_name]
ga = GA_Class(X, y, **ga_configs)
best = ga.evolve()

print("Saving results...")

results = {
    'algorithm': algorithm_name,
    'best_fitness': float(best['fitness']),
    'best_chromosome': best['chromosome'].tolist(),
    'history': [{'generation': h['generation'], 'best_fitness': float(h['best_fitness']),
                 'mean_fitness': float(h['mean_fitness']), 'diversity': float(h['diversity'])}
                for h in ga.history]
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(RESULTS_DIR, f"iris_{algorithm_name}_results_{timestamp}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
print(f"Total time: {time.time()-start:.2f}s")