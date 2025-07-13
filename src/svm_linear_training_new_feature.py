from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle





version = "base"
name_try = "adasyn_changeIntensity"
categorical_cols = ["fix", "nf", "lt", "pd", "exp"]
n_cv = 5
n_train_sizes = 20



# === Load dataset ===
df = pd.read_csv(f"File/ActiveMQ_input_{version}.csv")
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm", "rexp", "bug",    "nf", "ld", "la"]
df = df.dropna(subset=["npt"])
df["change_intensity"] = (df["ld"] + df["la"]) / df["nf"]
X = df.drop(columns=columns_to_drop)
y = df["bug"]



# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# # === Preprocessing ===
# numeric_cols = [col for col in X.columns if col not in categorical_cols]

# preprocessor = ColumnTransformer([
#     ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
#     ('num', StandardScaler(), numeric_cols)
# ])


# # Fit on training set and transform both
# X_train_transformed = preprocessor.fit_transform(X_train)
# X_test_transformed = preprocessor.transform(X_test)



# === Create the Pipeline with ADASYN ===
pipeline = Pipeline([
    ('adasyn', ADASYN(random_state=42)),
    ('model', LinearSVC(C=1.0, max_iter=10000, random_state=42))
])

# === Learning Curve ===
train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X_train.values, y_train.values,
    train_sizes=np.linspace(0.1, 1.0, n_train_sizes),
    cv=n_cv,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)



os.makedirs(f"File/training/{name_try}", exist_ok=True)



# === Plot ===
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy', color='blue')
plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy', color='green')
plt.title('Learning Curve')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f"File/training/{name_try}/learning_curve.svg", dpi=300, bbox_inches='tight')
plt.savefig(f"File/training/{name_try}/learning_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# === Final Evaluation ===
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n EVALUATION OF TEST SET")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall:", recall_score(y_test, y_pred, pos_label=1))
print("F1-score:", f1_score(y_test, y_pred, pos_label=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# === Save model and datasets ===
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