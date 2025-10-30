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
from multiprocessing import Pool, cpu_count
import warnings
from data.data_loader import load_breast_cancer
from core.genetic_algorithms import *

# Suppress warnings in child processes
warnings.filterwarnings('ignore')


def run_single_ga(args):
    """Run a single GA trial. This function will be called in parallel."""
    run_idx, algorithm_name, X, y, ga_configs, algorithms = args

    GA_Class = algorithms[algorithm_name]

    # Suppress output for each GA run
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                ga = GA_Class(X, y, **ga_configs)
                best = ga.evolve()

    # Extract results
    result = {
        'run': run_idx,
        'fitness': float(best['fitness']),
        'chromosome': best['chromosome'].tolist(),
        'history': ga.history
    }

    n_features = X.shape[1]

    # Decode features for counting based on algorithm type
    if algorithm_name in ['threshold', 'thresholdPenalty']:
        # Features active when allele > threshold
        active_features = best['chromosome'][:-1] > best['chromosome'][-1]

    elif algorithm_name == 'stochastic':
        # Need multiple samples for stochastic - use expectation
        active_features = best['chromosome'][:-1] > 0.5

    elif algorithm_name in ['ranking', 'rankingPenalty']:
        # Top n_select features based on ranking
        threshold = best['chromosome'][-1]
        n_select = int(n_features * threshold)
        if n_select > 0:
            top_indices = np.argpartition(best['chromosome'][:-1], -n_select)[-n_select:]
            active_features = np.zeros(n_features, dtype=bool)
            active_features[top_indices] = True
        else:
            active_features = np.zeros(n_features, dtype=bool)

    elif algorithm_name == 'weighted':
        # Features with weight > 0.5
        active_features = best['chromosome'][:-1] > 0.5

    # Return which features were selected
    selected_features = []
    for feat_idx in range(n_features):
        if active_features[feat_idx]:
            selected_features.append(feat_idx)

    return result, selected_features


def main():
    print("Starting Parallel GA Feature Selection...")
    start = time.time()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    algorithms = {
        'threshold': ThresholdDecodingGA,
        'thresholdPenalty': ThresholdDecodingPenaltyGA,
        'stochastic': StochasticDecodingGA,
        'ranking': RankingDecodingGA,
        'rankingPenalty': RankingDecodingPenaltyGA,
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

    ga_configs = {
        'population_size': 20,
        'generations': 250,
        'elitism_ratio': 0.05,
        'crossover_rate': 0.7,
        'mutation_rate': 0.2,
        'knn_k': 3,
        'gpu': False
    }

    print(f"GA Config: pop={ga_configs['population_size']}, gens={ga_configs['generations']}")

    n_runs = 100
    n_features = X.shape[1]

    # Determine number of parallel workers
    max_workers = min(cpu_count(), 25)  # Use at most 10 cores
    print(f"Available CPU cores: {cpu_count()}, using: {max_workers}")

    print(f"\nRunning {n_runs} independent trials in parallel...")
    print(f"Setup time: {time.time() - start:.2f}s")

    # Prepare arguments for parallel execution
    run_args = [
        (run_idx, algorithm_name, X, y, ga_configs, algorithms)
        for run_idx in range(n_runs)
    ]

    # Run GA trials in parallel with progress bar
    all_best_individuals = []
    feature_counts = defaultdict(int)

    parallel_start = time.time()

    with Pool(processes=max_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(run_single_ga, run_args),
            total=n_runs,
            desc=f"GA Runs ({max_workers} workers)"
        ))

    # Process results
    for result, selected_features in results:
        all_best_individuals.append(result)
        for feat_idx in selected_features:
            feature_counts[feat_idx] += 1

    parallel_time = time.time() - parallel_start
    print(f"Parallel execution time: {parallel_time:.2f}s")

    print("\nProcessing and saving results...")

    # Create feature selection summary
    feature_selection_summary = {}
    for feat_idx in range(n_features):
        feature_selection_summary[f'feature_{feat_idx}'] = {
            'selected_count': feature_counts[feat_idx],
            'selection_ratio': feature_counts[feat_idx] / n_runs
        }

    # Calculate fitness statistics
    fitness_values = [ind['fitness'] for ind in all_best_individuals]

    results = {
        'algorithm': algorithm_name,
        'n_runs': n_runs,
        'parallel_workers': max_workers,
        'execution_time': parallel_time,
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

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"cancer_{algorithm_name}_{n_runs}runs_{timestamp}.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Parallel workers: {max_workers}")
    print(f"Total runs: {n_runs}")
    print(f"Execution time: {parallel_time:.2f}s")
    print(f"\nFitness Statistics:")
    print(f"  Best:  {results['fitness_summary']['best']:.4f}")
    print(f"  Mean:  {results['fitness_summary']['mean']:.4f} ± {results['fitness_summary']['std']:.4f}")
    print(f"  Worst: {results['fitness_summary']['worst']:.4f}")

    print(f"\nTop 10 Most Selected Features:")
    sorted_features = sorted(
        feature_selection_summary.items(),
        key=lambda x: x[1]['selected_count'],
        reverse=True
    )
    for i, (feat_name, feat_data) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {feat_name}: {feat_data['selected_count']}/{n_runs} ({feat_data['selection_ratio']:.2%})")

    print(f"\nResults saved to: {output_file}")
    print(f"Total time: {time.time() - start:.2f}s")

    # Calculate and display speedup
    estimated_serial_time = (parallel_time / n_runs) * n_runs * max_workers
    speedup = estimated_serial_time / parallel_time
    print(f"\nEstimated speedup: ~{speedup:.1f}x faster than serial execution")


if __name__ == "__main__":
    main()