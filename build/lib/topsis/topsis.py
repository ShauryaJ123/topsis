import numpy as np
import pandas as pd

def check_numeric_data(data):
    """
    Check if all columns in the dataset are numeric. Raise an error if not.
    
    Parameters:
        data (pd.DataFrame): The input dataset to check.
    
    Raises:
        ValueError: If any column contains non-numeric data.
    """
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

    if len(non_numeric_columns) > 0:
        raise ValueError(
            f"The following columns are not numeric: {', '.join(non_numeric_columns)}. "
            "Please ensure all columns contain numeric values."
        )

def topsis(file_path, weights, impacts):
    """
    Perform the TOPSIS method on the given data.

    Parameters:
        file_path (str): Path to the input CSV file containing the decision matrix.
        weights (list): List of weights for each criterion.
        impacts (list): List of 'up' or 'down' for each criterion.

    Returns:
        pd.DataFrame: Original data with an additional 'Topsis Score' and 'Rank' columns.
    
    Raises:
        Exception: For various input and logical errors.
    """
    try:
        # Step 1: Validate file format and read the data
        if not file_path.endswith(".csv"):
            raise ValueError("Input file must be a CSV file.")

        data = pd.read_csv(file_path)

        # Ensure the file has at least three columns: alternatives and criteria
        if data.shape[1] < 3:
            raise ValueError("The input file must contain at least three columns.")

        # Step 2: Validate weights and impacts
        if len(weights) != data.shape[1] - 1:
            raise ValueError("The number of weights must match the number of criteria (excluding the first column).")

        if len(impacts) != data.shape[1] - 1:
            raise ValueError("The number of impacts must match the number of criteria (excluding the first column).")

        valid_impacts = {"up", "down"}
        if not all(impact in valid_impacts for impact in impacts):
            raise ValueError(
                f"Impacts must be 'up' or 'down'. Invalid impacts provided: {', '.join(impacts)}."
            )

        # Step 3: Check if all criteria columns are numeric
        check_numeric_data(data.iloc[:, 1:])

        # Step 4: Normalize the decision matrix
        norm_data = data.iloc[:, 1:] / np.sqrt((data.iloc[:, 1:] ** 2).sum(axis=0))

        # Step 5: Weighted normalized decision matrix
        weighted_data = norm_data * weights

        # Step 6: Determine ideal best and worst
        ideal_best = [max(col) if imp == 'up' else min(col) for col, imp in zip(weighted_data.T, impacts)]
        ideal_worst = [min(col) if imp == 'up' else max(col) for col, imp in zip(weighted_data.T, impacts)]

        # Step 7: Calculate distances from ideal best and worst
        dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

        # Step 8: Calculate the relative closeness to the ideal solution
        scores = dist_worst / (dist_best + dist_worst)

        # Step 9: Add scores and ranks to the original data
        data["Topsis Score"] = scores
        data["Rank"] = scores.rank(ascending=False).astype(int)

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{file_path}' was not found.")
    
    except pd.errors.EmptyDataError:
        raise ValueError("The input file is empty. Please provide a valid CSV file with data.")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

