import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import os





version = "base_restricted"
n_cv = 5

# Load dataset (START)
df = pd.read_csv(f"File/ActiveMQ_input_{version}.csv")
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp", "bug"]
# columns_to_drop = ["useless", "bug"]
# columns_to_drop = ["bug"]
df = df.dropna(subset=["npt"])
X = df.drop(columns=columns_to_drop)
y = df["bug"]
# Load dataset (END)



# Define SVM model with linear kernel (no scaling)
svm_model = SVC(kernel='linear', C=1.0)

# Generate learning curve data using the full dataset (START)
train_sizes, train_scores, val_scores = learning_curve(
    svm_model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=n_cv,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)
# Generate learning curve data using the full dataset (END)



# Calculate mean and standard deviation for training and validation scores (START)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
# Calculate mean and standard deviation for training and validation scores (END)



# Plot learning curves (START)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves - SVM (linear kernel, no scaling)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot learning curves (END)



# Compute cross-validated scores for additional metrics (START)
precision = cross_val_score(svm_model, X, y, cv=n_cv, scoring='precision_macro')
recall = cross_val_score(svm_model, X, y, cv=n_cv, scoring='recall_macro')
f1 = cross_val_score(svm_model, X, y, cv=n_cv, scoring='f1_macro')
# Compute cross-validated scores for additional metrics (END)



# Print results (START)
print("Cross-validated metrics (5-fold, macro average):")
print(f"Precision: {precision.mean():.4f} ± {precision.std():.4f}")
print(f"Recall:    {recall.mean():.4f} ± {recall.std():.4f}")
print(f"F1-score:  {f1.mean():.4f} ± {f1.std():.4f}")
# Print results (END)