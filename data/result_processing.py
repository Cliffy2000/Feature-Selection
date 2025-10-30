import json
import pandas as pd
import numpy as np
import sys


def process_ga_results(json_filename):
    with open(json_filename, 'r') as f:
        data = json.load(f)

    n_runs = len(data['all_runs'])
    n_features = len(data['all_runs'][0]['chromosome']) - 1

    # Create dataframe
    rows = []
    for i in range(n_features):
        row = {'Feature': f'feature_{i}'}
        for j in range(n_runs):
            row[f'run {j + 1}'] = data['all_runs'][j]['chromosome'][i]
        rows.append(row)

    # Add threshold row
    threshold_row = {'Feature': 'threshold'}
    for j in range(n_runs):
        threshold_row[f'run {j + 1}'] = data['all_runs'][j]['chromosome'][-1]
    rows.append(threshold_row)

    df = pd.DataFrame(rows)

    # Save to Excel
    output_filename = json_filename.replace('.json', '.xlsx')
    df.to_excel(output_filename, index=False)
    print(f"Saved to {output_filename}")


if __name__ == "__main__":
    process_ga_results('../results/cancer_thresholdPenalty_100runs_20251012_212100.json')