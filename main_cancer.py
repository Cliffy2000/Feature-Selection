import os
import sys
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import contextlib


print("Starting GA Feature Selection...")
start = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from data.data_loader import load_breast_cancer
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

X, y = load_breast_cancer()
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Setup time: {time.time() - start:.2f}s")

ga_configs = {
    'population_size': 20,
    'generations': 200,
    'elitism_ratio': 0.05,
    'crossover_rate': 0.7,
    'mutation_rate': 0.2,
    'knn_k': 3,
    'gpu': False
}

print(f"Initializing GA: pop={ga_configs['population_size']}, gens={ga_configs['generations']}")

n_runs = 25
all_best_individuals = []
feature_counts = defaultdict(int)
n_features = X.shape[1]

print(f"\nRunning {n_runs} independent trials...")

for run_idx in tqdm(range(n_runs), desc="GA Runs"):
    GA_Class = algorithms[algorithm_name]
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            ga = GA_Class(X, y, **ga_configs)
            best = ga.evolve()

    all_best_individuals.append({
        'run': run_idx,
        'fitness': float(best['fitness']),
        'chromosome': best['chromosome'].tolist(),
        'history': ga.history
    })

    decoded = ga.decode(best['chromosome'])
    if hasattr(decoded, 'dtype') and decoded.dtype == bool:
        active_features = decoded[:-1]
    elif hasattr(decoded, 'dtype') and decoded.dtype == np.float32:
        active_features = decoded[:-1] > 0.5
    else:
        active_features = decoded[:-1] > decoded[-1]

    for feat_idx in range(n_features):
        if active_features[feat_idx]:
            feature_counts[feat_idx] += 1

print("\nSaving results...")

feature_selection_summary = {}
for feat_idx in range(n_features):
    feature_selection_summary[f'feature_{feat_idx}'] = {
        'selected_count': feature_counts[feat_idx],
        'selection_ratio': feature_counts[feat_idx] / n_runs
    }

fitness_values = [ind['fitness'] for ind in all_best_individuals]

results = {
    'algorithm': algorithm_name,
    'n_runs': n_runs,
    'fitness_summary': {
        'best': float(np.max(fitness_values)),
        'worst': float(np.min(fitness_values)),
        'mean': float(np.mean(fitness_values)),
        'std': float(np.std(fitness_values))
    },
    'feature_selection_summary': feature_selection_summary,
    'all_runs': [
        {
            'run': ind['run'],
            'fitness': ind['fitness'],
            'chromosome': ind['chromosome']
        } for ind in all_best_individuals
    ]
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(RESULTS_DIR, f"cancer_{algorithm_name}_25runs_{timestamp}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== RESULTS SUMMARY ===")
print(f"Algorithm: {algorithm_name}")
print(
    f"Fitness - Best: {results['fitness_summary']['best']:.4f}, Mean: {results['fitness_summary']['mean']:.4f} ± {results['fitness_summary']['std']:.4f}")
print(f"\nTop 10 Most Selected Features:")
sorted_features = sorted(feature_selection_summary.items(), key=lambda x: x[1]['selected_count'], reverse=True)
for i, (feat_name, feat_data) in enumerate(sorted_features[:10]):
    print(f"  {feat_name}: {feat_data['selected_count']}/{n_runs} ({feat_data['selection_ratio']:.2%})")

print(f"\nResults saved to {output_file}")
print(f"Total time: {time.time() - start:.2f}s")