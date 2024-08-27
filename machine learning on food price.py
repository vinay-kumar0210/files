import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the actual path to your CSV file)
df = pd.read_csv('/content/global_food_price_dataset.csv')

# Dictionary to store results
results = {}

# Check the column names to ensure correctness
print(df.columns)

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
count = 0
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) & (df['adm1_id'] == adm1_id) & (df['cm_id'] == cm_id)]

            if len(subset) > 0:
                # Calculate mean
                mean = sum(subset['mp_price']) / len(subset['mp_price'])

                # Calculate variance (for standard deviation)
                variance = sum((x - mean) ** 2 for x in subset['mp_price']) / len(subset['mp_price'])

                # Calculate covariance
                cov = sum((x - mean) * (y - mean) for x, y in zip(subset['mp_price'], subset['mp_price'])) / len(subset['mp_price'])

                # Add results to dictionary
                results[(adm0_id, adm1_id, cm_id)] = {
                    'mean': mean,
                    'standard_deviation': variance ** 0.5,  # Square root of variance
                    'covariance': cov
                }

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, values in results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"Mean: {values['mean']}")
    print(f"Standard Deviation: {values['standard_deviation']}")
    print(f"Covariance: {values['covariance']}")
    print("="*20)
import pandas as pd
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual path to your CSV file)
df = pd.read_csv('/content/global_food_price_dataset.csv')

# Dictionary to store results
results1 = {}

# Check the column names to ensure correctness
print(df.columns)

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
count = 0
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) & (df['adm1_id'] == adm1_id) & (df['cm_id'] == cm_id)]

            if len(subset) > 1:  # Chi-square requires at least two values
                observed_values = subset['mp_price'].values
                expected_mean = results[(adm0_id, adm1_id, cm_id)]['mean']
                expected_values = np.full(len(observed_values), expected_mean)

                # Calculate Chi-square
                chi_square = np.sum((observed_values - expected_values) ** 2 / expected_values)

                # Add results to dictionary
                results1[(adm0_id, adm1_id, cm_id)] = chi_square

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, value in results1.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"Chi-square: {value}")
    print("="*20)

import pandas as pd
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual path to your CSV file)
df = pd.read_csv('/content/global_food_price_dataset.csv')

# Dictionary to store metrics results
metrics_results = {}

# Check the column names to ensure correctness
print(df.columns)

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
count = 0
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) & (df['adm1_id'] == adm1_id) & (df['cm_id'] == cm_id)]

            if len(subset) > 1:  # At least two values are required for calculations
                observed_values = subset['mp_price'].values
                expected_mean = results[(adm0_id, adm1_id, cm_id)]['mean']
                predicted_values = np.full(len(observed_values), expected_mean)

                # Calculate metrics: MAE, MSE, RMSE
                n = len(observed_values)
                mae = sum([abs(y_observed - y_predicted) for y_observed, y_predicted in zip(observed_values, predicted_values)]) / n
                mse = sum([(y_observed - y_predicted) ** 2 for y_observed, y_predicted in zip(observed_values, predicted_values)]) / n
                rmse = mse ** 0.5

                # Add results to dictionary
                metrics_results[(adm0_id, adm1_id, cm_id)] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse
                }

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, values in metrics_results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"MAE: {values['MAE']}")
    print(f"MSE: {values['MSE']}")
    print(f"RMSE: {values['RMSE']}")
    print("="*20)
import pandas as pd
import csv

# Read CSV file
csv_file = open('/content/global_food_price_dataset.csv')
csv_reader = csv.reader(csv_file)
header = next(csv_reader)

# Convert CSV to pandas DataFrame
df = pd.DataFrame(csv_reader, columns=header)

# Dictionary to store correlation results
correlation_results = {}

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
count = 0
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == str(adm0_id)) &
                        (df['adm1_id'] == str(adm1_id)) &
                        (df['cm_id'] == str(cm_id))]

            if len(subset) > 1:  # Correlation requires at least two values
                # Convert columns to numeric
                subset['mp_year'] = pd.to_numeric(subset['mp_year'])
                subset['mp_month'] = pd.to_numeric(subset['mp_month'])
                subset['mp_price'] = pd.to_numeric(subset['mp_price'])

                # Calculate means
                mean_year = subset['mp_year'].mean()
                mean_month = subset['mp_month'].mean()
                mean_price = subset['mp_price'].mean()

                # Calculate correlation
                numerator = ((subset['mp_year'] - mean_year) *
                             (subset['mp_month'] - mean_month) *
                             (subset['mp_price'] - mean_price)).sum()

                denominator1 = ((subset['mp_year'] - mean_year) ** 2).sum()
                denominator2 = ((subset['mp_month'] - mean_month) ** 2).sum()
                denominator3 = ((subset['mp_price'] - mean_price) ** 2).sum()

                correlation_year_month_price = numerator / (denominator1 * denominator2 * denominator3) ** 0.5

                # Store correlation result
                correlation_results[(adm0_id, adm1_id, cm_id)] = correlation_year_month_price

                count += 1
                if count >= 7:
                    breakimport pandas as pd
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual path to your CSV file)
df = pd.read_csv('/content/global_food_price_dataset.csv')

# Dictionary to store correlation results
correlation_results = {}

# Function to calculate Pearson correlation coefficient
def calculate_correlation(df_subset, col1, col2):
    x_values = df_subset[col1].values
    y_values = df_subset[col2].values

    # Calculate means
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)

    # Calculate numerator and denominators
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator_x = sum((x - x_mean) ** 2 for x in x_values)
    denominator_y = sum((y - y_mean) ** 2 for y in y_values)

    # Calculate Pearson correlation coefficient
    correlation = numerator / (np.sqrt(denominator_x) * np.sqrt(denominator_y))

    return correlation

# Check the column names to ensure correctness
print(df.columns)

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
count = 0
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) & (df['adm1_id'] == adm1_id) & (df['cm_id'] == cm_id)]

            if len(subset) > 1:  # At least two values are required for calculations
                # Calculate correlation between 'mp_year' and 'mp_price'
                correlation_year_price = calculate_correlation(subset, 'mp_year', 'mp_price')

                # Calculate correlation between 'mp_month' and 'mp_price'
                correlation_month_price = calculate_correlation(subset, 'mp_month', 'mp_price')

                # Add results to dictionary
                correlation_results[(adm0_id, adm1_id, cm_id)] = {
                    'correlation_year_price': correlation_year_price,
                    'correlation_month_price': correlation_month_price
                }

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, values in correlation_results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"Correlation between 'mp_year' and 'mp_price': {values['correlation_year_price']}")
    print(f"Correlation between 'mp_month' and 'mp_price': {values['correlation_month_price']}")
    print("="*20)

            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 5 combinations
for keys, value in correlation_results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"Correlation between 'mp_year', 'mp_month', and 'mp_price': {value}")
    print("="*20)

# Close CSV file
csv_file.close()
import pandas as pd
import numpy as np

# Read CSV file with specified encoding
df = pd.read_csv('/content/global_food_price_dataset.csv', encoding='latin1')

# Dictionary to store correlation results
correlation_results = {}

# Function to calculate Pearson correlation coefficient
def calculate_correlation(df_subset, col1, col2):
    x_values = df_subset[col1].values
    y_values = df_subset[col2].values

    # Calculate means
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)

    # Calculate numerator and denominators
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator_x = sum((x - x_mean) ** 2 for x in x_values)
    denominator_y = sum((y - y_mean) ** 2 for y in y_values)

    # Calculate Pearson correlation coefficient
    correlation = numerator / (np.sqrt(denominator_x) * np.sqrt(denominator_y))

    return correlation

# Check the column names to ensure correctness
print(df.columns)
import pandas as pd

# Read the CSV file with specified encoding
df = pd.read_csv('/content/global_food_price_dataset.csv', encoding='latin1')

# Dictionary to store linear regression results
linear_reg_results = {}

# Counter for the number of combinations processed
count = 0

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) &
                        (df['adm1_id'] == adm1_id) &
                        (df['cm_id'] == cm_id)]

            if len(subset) > 1:  # At least two values are required for calculations
                x = subset['mp_month'].values  # Feature: 'mp_month'
                y = subset['mp_price'].values  # Target: 'mp_price'

                # Calculate the mean of x and y
                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)

                # Calculate the slope (m)
                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
                denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
                slope = numerator / denominator

                # Calculate the intercept (b)
                intercept = mean_y - slope * mean_x

                # Calculate predicted price for mp_month = 1
                predicted_price = slope * 1 + intercept

                # Add results to dictionary
                linear_reg_results[(adm0_id, adm1_id, cm_id)] = {
                    'slope': slope,
                    'intercept': intercept,
                    'predicted_price': predicted_price
                }

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, values in linear_reg_results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"Slope: {values['slope']}")
    print(f"Intercept: {values['intercept']}")
    print(f"Predicted 'mp_price' for 'mp_month' = 1: {values['predicted_price']}")
    print("="*20)

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
count = 0
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) & (df['adm1_id'] == adm1_id) & (df['cm_id'] == cm_id)]

            if len(subset) > 1:  # At least two values are required for calculations
                # Calculate correlation between 'mp_year' and 'mp_price'
                correlation_year_price = calculate_correlation(subset, 'mp_year', 'mp_price')

                # Calculate correlation between 'mp_month' and 'mp_price'
                correlation_month_price = calculate_correlation(subset, 'mp_month', 'mp_price')

                # Add results to dictionary
                correlation_results[(adm0_id, adm1_id, cm_id)] = {
                    'correlation_year_price': correlation_year_price,
                    'correlation_month_price': correlation_month_price
                }

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, values in correlation_results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print(f"Correlation between 'mp_year' and 'mp_price': {values['correlation_year_price']}")
    print(f"Correlation between 'mp_month' and 'mp_price': {values['correlation_month_price']}")
    print("="*20)
import pandas as pd
import numpy as np

# Read the CSV file with specified encoding
df = pd.read_csv('/content/global_food_price_dataset.csv', encoding='latin1')

# Euclidean distance calculation
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# k-NN clustering
def k_means_clustering(data, k=5, max_iters=100):
    # Randomly initialize centroids
    n_samples, _ = data.shape
    cluster_labels = np.zeros(n_samples)
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    # Iterate through max_iters
    for _ in range(max_iters):
        # Assign samples to nearest centroid
        for i, sample in enumerate(data):
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            cluster_labels[i] = np.argmin(distances)

        # Update centroids
        for i in range(k):
            cluster_samples = data[cluster_labels == i]
            if len(cluster_samples) > 0:
                centroids[i] = np.mean(cluster_samples, axis=0)

    return cluster_labels

# Dictionary to store clustering results
knn_results = {}

# Check the column names to ensure correctness
print(df.columns)

# Counter for the number of combinations processed
count = 0

# Iterate over unique combinations of 'adm0_id', 'adm1_id', and 'cm_id'
for adm0_id in range(1, 101):
    for adm1_id in range(272, 30001):
        for cm_id in range(55, 100001):
            subset = df[(df['adm0_id'] == adm0_id) &
                        (df['adm1_id'] == adm1_id) &
                        (df['cm_id'] == cm_id)]

            if len(subset) > 1:  # At least two values are required for clustering
                X = subset[['mp_month', 'mp_price']].values

                # Number of clusters (you can adjust this as needed)
                k = 5

                # Perform k-means clustering
                cluster_labels = k_means_clustering(X, k=k)

                # Add cluster labels to the dictionary
                knn_results[(adm0_id, adm1_id, cm_id)] = cluster_labels

                count += 1
                if count >= 7:
                    break
            if count >= 7:
                break
        if count >= 7:
            break

# Display results for the first 100 combinations
for keys, cluster_labels in knn_results.items():
    print(f"adm0_id: {keys[0]}, adm1_id: {keys[1]}, cm_id: {keys[2]}")
    print("Cluster Labels:")
    print(cluster_labels)
    print("="*20)
