import os
import sys
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import time
start = time.time()
print("Starting GA Feature Selection...")
print("Starting GA Feature Selection...")

try:
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"GPU ENABLED: Using CUDA acceleration")
except ImportError:
    GPU_AVAILABLE = False
    print(f"GPU DISABLED: Using CPU computation")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from data.data_loader import load_data
from core.genetic_algorithms import *

algorithms = {
    'threshold': ThresholdDecodingGA,
    'stochastic': StochasticDecodingGA,
    'ranking': RankingDecodingGA,
    'weighted': WeightedFeaturesGA
}

algorithm_name = sys.argv[1] if len(sys.argv) > 1 else 'threshold'  # Default algorithm

if algorithm_name not in algorithms:
    print("Invalid algorithm name.")
    sys.exit(1)

print(f"Algorithm selected: {algorithm_name}")
print("Loading and preprocessing data...")

scaler = StandardScaler()
print(f"Imports done: {time.time()-start:.2f}s")
X, y = load_data()
print(f"Data loaded: {time.time()-start:.2f}s")
X = scaler.fit_transform(X)
print(f"Data scaled: {time.time()-start:.2f}s")

print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

ga_configs = {
    'population_size': 100,
    'generations': 500,
    'elitism_ratio': 0.05,
    'crossover_rate': 0.7,
    'mutation_rate': 0.2,
    'gpu': GPU_AVAILABLE
}

print(f"Initializing GA with population={ga_configs['population_size']}, generations={ga_configs['generations']}")

GA_Class = algorithms[algorithm_name]
ga = GA_Class(X, y, **ga_configs)
best = ga.evolve()

print("Evolution complete. Saving results...")

results = {
    'algorithm': algorithm_name,
    'best_fitness': float(best['fitness']),
    'best_chromosome': best['chromosome'].tolist(),
    'history': [{'generation': h['generation'], 'best_fitness': float(h['best_fitness']),
                 'mean_fitness': float(h['mean_fitness']), 'diversity': float(h['diversity'])}
                for h in ga.history]
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(RESULTS_DIR, f"{algorithm_name}_results_{timestamp}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")