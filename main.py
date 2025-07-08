import argparse
import os
from benchmarks.pca import analyze_pca_components, run_pca_model_benchmarks


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Feature Selection Research Project')
    parser.add_argument('--benchmark', type=str, choices=['pca', 'pca-models'],
                        help='Run specific benchmark')
    parser.add_argument('--ga', action='store_true',
                        help='Run genetic algorithm feature selection')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing result files')

    args = parser.parse_args()

    # Ensure results directory exists
    ensure_results_dir()

    if args.benchmark == 'pca':
        analyze_pca_components(overwrite=args.overwrite)
    elif args.benchmark == 'pca-models':
        run_pca_model_benchmarks(overwrite=args.overwrite)
    elif args.ga:
        print("GA implementation not yet available")
    else:
        print("No valid option specified. Use --help for options.")


if __name__ == "__main__":
    main()