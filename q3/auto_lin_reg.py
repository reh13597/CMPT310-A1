import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score
import q3 as q3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# load auto-mpg-regression.tsv, including Keys are the column names, including mpg.
auto_data_all = q3.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q3.standard and q3.one_hot.

features1 = [('cylinders', q3.standard),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

features2 = [('cylinders', q3.one_hot),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = q3.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

print("=" * 80)
print("AUTO MPG PREDICTION - CROSS-VALIDATION ANALYSIS")
print("=" * 80)
print(f"\nDataset size: {auto_data[0].shape[1]} cars")
print(f"Features in features1: {auto_data[0].shape[0]}")
print(f"Features in features2: {auto_data[1].shape[0]}")
print()

# Lambda values to test
lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Number of folds for cross-validation
k_folds = 10

# Store results
results = {}

# Feature set names
feature_names = ['features1 (cylinders: standard)', 'features2 (cylinders: one-hot)']

print("Running cross-validation...")
print(f"Testing {len(lambda_values)} lambda values with {k_folds}-fold cross-validation")
print()

# Test each feature set
for feature_idx, feature_name in enumerate(feature_names):
    print("=" * 80)
    print(f"Testing {feature_name}")
    print("=" * 80)

    X = auto_data[feature_idx]
    y = auto_values

    results[feature_name] = {}

    # Test each lambda value
    for lam in lambda_values:
        # Perform cross-validation
        cv_rmse = q3.xval_learning_alg(X, y, lam, k_folds)

        # Extract scalar value (xval_learning_alg might return array or scalar)
        if isinstance(cv_rmse, np.ndarray):
            cv_rmse = float(cv_rmse.flatten()[0])
        else:
            cv_rmse = float(cv_rmse)

        # Convert back to original MPG units
        # Remember: we standardized y-values, so we need to convert back
        cv_rmse_mpg = cv_rmse * sigma[0]

        results[feature_name][lam] = {
            'cv_rmse_standardized': cv_rmse,
            'cv_rmse_mpg': cv_rmse_mpg
        }

        print(f"  λ = {lam:8.3f} → CV RMSE = {cv_rmse:.4f} (standardized) = {cv_rmse_mpg:.4f} mpg")

    print()

#-------------------------------------------------------------------------------
# Find Best Configuration
#-------------------------------------------------------------------------------

print("=" * 80)
print("FINDING BEST CONFIGURATION")
print("=" * 80)
print()

best_rmse = float('inf')
best_feature_set = None
best_lambda = None

for feature_name in results.keys():
    for lam in lambda_values:
        rmse_mpg = results[feature_name][lam]['cv_rmse_mpg']

        if rmse_mpg < best_rmse:
            best_rmse = rmse_mpg
            best_feature_set = feature_name
            best_lambda = lam

print("RESULTS:")
print("-" * 80)
print(f"1. Best combination:")
print(f"   Feature set: {best_feature_set}")
print(f"   Lambda (λ):  {best_lambda}")
print()
print(f"2. Average cross-validation RMSE:")
print(f"   {best_rmse:.4f} mpg")
print()
print("-" * 80)

#-------------------------------------------------------------------------------
# Detailed Analysis
#-------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DETAILED ANALYSIS - ALL RESULTS")
print("=" * 80)

for feature_name in feature_names:
    print(f"\n{feature_name}")
    print("-" * 80)
    print(f"{'Lambda':>10} {'CV RMSE (mpg)':>15}")
    print("-" * 80)

    for lam in lambda_values:
        rmse_mpg = results[feature_name][lam]['cv_rmse_mpg']
        marker = " ← BEST" if (feature_name == best_feature_set and lam == best_lambda) else ""
        print(f"{lam:10.3f} {rmse_mpg:15.4f}{marker}")

#-------------------------------------------------------------------------------
# Comparison Summary
#-------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

# Best lambda for each feature set
print("\nBest lambda for each feature set:")
print("-" * 80)

for feature_name in feature_names:
    best_lam_for_feature = min(lambda_values,
                                key=lambda l: results[feature_name][l]['cv_rmse_mpg'])
    best_rmse_for_feature = results[feature_name][best_lam_for_feature]['cv_rmse_mpg']

    print(f"{feature_name}:")
    print(f"  Best λ = {best_lam_for_feature}")
    print(f"  CV RMSE = {best_rmse_for_feature:.4f} mpg")
    print()

# Effect of regularization
print("\nEffect of Regularization:")
print("-" * 80)

for feature_name in feature_names:
    no_reg = results[feature_name][0.0]['cv_rmse_mpg']
    with_best = min(results[feature_name][lam]['cv_rmse_mpg'] for lam in lambda_values)
    improvement = no_reg - with_best
    improvement_pct = (improvement / no_reg) * 100

    print(f"{feature_name}:")
    print(f"  Without regularization (λ=0):     {no_reg:.4f} mpg")
    print(f"  With best regularization:         {with_best:.4f} mpg")
    print(f"  Improvement:                      {improvement:.4f} mpg ({improvement_pct:.2f}%)")
    print()
