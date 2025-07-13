from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle





version = "base"
name_try = "adasyn_poly_2"
categorical_cols = ["fix", "nf", "lt", "pd", "exp"]
cols_to_scale = []
n_cv = 5
n_train_sizes = 20

# Load dataset (START)
df = pd.read_csv(f"File/ActiveMQ_input_{version}.csv")
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp", "bug"  ]
# columns_to_drop = ["useless", "bug"]
# columns_to_drop = ["bug"]
df = df.dropna(subset=["npt"])
X = df.drop(columns=columns_to_drop)
y = df["bug"]
# Load dataset (END)



# Initial Split (START)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Initial Split (END)

# Preprocessing setup (START)  
categorical_indices = [i for i, col in enumerate(X.columns) if col in categorical_cols]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), cols_to_scale)
])
# X_train = preprocessor.fit_transform(X_train)
# X_test = preprocessor.transform(X_test)
# Preprocessing setup (END)  


# Pipeline (START)
pipeline = Pipeline([
    ('adasyn', ADASYN(random_state=42)),
    ('model', SVC(C=1.0, kernel="poly", degree=2, random_state=42))
    # ('model', LinearSVC(C=1.0, max_iter=10000, random_state=42))
])
# Pipeline (END)

# Learning Curve (START)
train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, n_train_sizes),
    cv=n_cv,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)
# Learning Curve (END)



os.makedirs(f"File/training/{name_try}", exist_ok=True)



# === 5. Plot Learning Curve ===
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy', color='blue')
plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy', color='green')
plt.title('Learning Curve')
plt.xlabel('Training size')
plt.ylabel('Accuracy')

plt.ylim(0.5, 1)

plt.legend()
plt.grid(True)
plt.savefig(f"File/training/{name_try}/learning_curve.svg", dpi=300, bbox_inches='tight')
plt.savefig(f"File/training/{name_try}/learning_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# === 6. Valutazione finale su test ===
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n EVALUATION OF TEST SET")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall:", recall_score(y_test, y_pred, pos_label=1))
print("F1-score:", f1_score(y_test, y_pred, pos_label=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))



with open(f"File/training/{name_try}/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
with open(f"File/training/{name_try}/X.pkl", "wb") as f:
    pickle.dump(X, f)
with open(f"File/training/{name_try}/y.pkl", "wb") as f:
    pickle.dump(y, f)
with open(f"File/training/{name_try}/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open(f"File/training/{name_try}/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open(f"File/training/{name_try}/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open(f"File/training/{name_try}/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)