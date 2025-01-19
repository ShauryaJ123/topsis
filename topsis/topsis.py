import numpy as np
import pandas as pd

def check_if_numeric(data):
    non_numeric = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns: {', '.join(non_numeric)}. Ensure all columns are numbers.")

def topsis(file_path, weights, impacts):
    try:
        if not file_path.endswith(".csv"):
            raise ValueError("Input must be a CSV file.")
        df = pd.read_csv(file_path)
        if df.shape[1] < 3:
            raise ValueError("File must have at least 3 columns.")
        if len(weights) != df.shape[1] - 1:
            raise ValueError("Weights must match number of criteria.")
        if len(impacts) != df.shape[1] - 1:
            raise ValueError("Impacts must match number of criteria.")
        if not all(imp in {"up", "down"} for imp in impacts):
            raise ValueError(f"Impacts must be 'up' or 'down'. Invalid: {', '.join(impacts)}.")

        check_if_numeric(df.iloc[:, 1:])
        norm_df=df.iloc[:, 1:]/np.sqrt((df.iloc[:, 1:] ** 2).sum(axis=0))
        weighted_df = norm_df * weights

        best =[max(col) if imp == 'up' else min(col) for col, imp in zip(weighted_df.T, impacts)]



        worst= [min(col) if imp == 'up' else max(col) for col, imp in zip(weighted_df.T, impacts)]

        dist_best = np.sqrt(((weighted_df - best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_df - worst) ** 2).sum(axis=1))

        scores = dist_worst / (dist_best + dist_worst)
        df["Topsis Score"] = scores
        df["Rank"] = scores.rank(ascending=False).astype(int)

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at '{file_path}'.")
    
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty. Please provide valid data.")

    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
