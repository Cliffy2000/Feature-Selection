import pandas as pd


def load_and_preprocess_data(data_path, target_column):
    df = pd.read_csv(data_path)

    # Normalize all columns except target
    for col in df.columns:
        if col != target_column:
            max_val = df[col].max()
            if max_val > 1:
                df[col] = df[col] / max_val

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def print_section(title):
    sep = "=" * len(title)
    print(f"\n\n{title}\n{sep}")
