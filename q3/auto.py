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

#-------------------------------------------------------------------------------
# Convert MPG to different fuel efficiency categories (low, medium, high)
#-------------------------------------------------------------------------------

def mpg_to_categories(mpg_values):
    mpg_flat = mpg_values.flatten()

    medium = np.percentile(mpg_flat, 33.33)
    high = np.percentile(mpg_flat, 66.67)

    categories = np.zeros(len(mpg_flat), dtype=int)
    categories[mpg_flat < medium] = 0 # low fuel efficiency
    categories[(mpg_flat >= medium) & (mpg_flat < high)] = 1 # medium fuel efficiency
    categories[mpg_flat >= high] = 2 # high fuel efficiency

    labels = ['Low', 'Medium', 'High']
    thresholds = [medium, high]

    return categories, labels, thresholds

mpg_categories, category_labels, thresholds = mpg_to_categories(auto_values)

print("=" * 70)
print("KNN Classification for Auto MPG Categories")
print("=" * 70)
print(f"Categories: {category_labels}")
print(f"Thresholds: {[f'{t:.2f}' for t in thresholds]}")
print(f"\nClass distribution:")
for i, label in enumerate(category_labels):
    count = np.sum(mpg_categories == i)
    percentage = (count / len(mpg_categories)) * 100
    print(f"  {label}: {count} samples ({percentage:.1f}%)")
print()

#-------------------------------------------------------------------------------
# 10-fold Cross Validation
#-------------------------------------------------------------------------------

k_values = [1, 3, 5, 7, 10, 15, 20, 25]

results = {}

for feature_idx, feature_name in enumerate(['features1 (cylinders: standard)',
                                            'features2 (cylinders: one-hot)']):
    print(f"\n{'=' * 70}")
    print(f"Testing with {feature_name}")
    print(f"{'=' * 70}")

    # Get the data for this feature set
    X = auto_data[feature_idx].T  # Transpose to get (n_samples, n_features)
    y = mpg_categories             # Use categories instead of raw values

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print()

    results[feature_name] = {}

    # Test different K values
    for k in k_values:
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # Perform 10-fold cross-validation
        # scoring='accuracy' gives us classification accuracy
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

        mean_accuracy = scores.mean()
        std_accuracy = scores.std()

        # Perform 10-fold cross-validation for PRECISION
        # For multi-class: 'weighted' averages precision per class weighted by support
        precision_scorer = make_scorer(precision_score, average='weighted', zero_division=0)
        precision_scores = cross_val_score(knn, X, y, cv=10, scoring=precision_scorer)
        mean_precision = precision_scores.mean()
        std_precision = precision_scores.std()

        results[feature_name][k] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_precision': mean_precision,
            'std_precision': std_precision,
        }

        print(f"K = {k:2d}: Accuracy = {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%) +/- {std_accuracy:.4f} | "
              f"Precision = {mean_precision:.4f} ({mean_precision*100:.2f}%) +/- {std_precision:.4f}")

    # Find best K for this feature set
    best_k = max(results[feature_name].keys(),
                 key=lambda k: results[feature_name][k]['mean_accuracy'])
    best_acc = results[feature_name][best_k]['mean_accuracy']

    print(f"\nBest K for {feature_name}: K = {best_k} "
          f"with Accuracy = {best_acc:.4f} ({best_acc*100:.2f}%)")

#-------------------------------------------------------------------------------
# Summary and Comparison
#-------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Compare best results from each feature set
best_overall = None
best_overall_acc = 0
best_overall_k = None
best_overall_features = None

for feature_name in results.keys():
    best_k = max(results[feature_name].keys(),
                 key=lambda k: results[feature_name][k]['mean_accuracy'])
    best_acc = results[feature_name][best_k]['mean_accuracy']

    print(f"\n{feature_name}:")
    print(f"  Best K: {best_k}")
    print(f"  Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

    if best_acc > best_overall_acc:
        best_overall_acc = best_acc
        best_overall_k = best_k
        best_overall_features = feature_name

print(f"\n{'=' * 70}")
print(f"OVERALL BEST CONFIGURATION:")
print(f"  Features: {best_overall_features}")
print(f"  K: {best_overall_k}")
print(f"  Accuracy: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
print(f"{'=' * 70}")

# Baseline comparison
baseline_accuracy = max([np.sum(mpg_categories == i) for i in range(len(category_labels))]) / len(mpg_categories)
print(f"\nBaseline (always predict most common class): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Improvement over baseline: {(best_overall_acc - baseline_accuracy)*100:.2f}%")