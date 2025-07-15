from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle





model_name = "svm_base"
file = "Camel"
version = "base"
name_try = "svm_base"
n_cv = 5
n_train_sizes = 20

# Load dataset (START)
df = pd.read_csv(f"File/{file}_input_{version}.csv")
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp", "bug"]
# columns_to_drop = ["useless", "bug"]
# columns_to_drop = ["bug"]
df = df.dropna(subset=["npt"])
X = df.drop(columns=columns_to_drop)
y = df["bug"]
# Load dataset (END)



# === 1. Split iniziale ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 3. Pipeline ===
pipeline = Pipeline([
    # ('model', SVC(kernel='linear', random_state=42))
    ('model', LinearSVC(C=1.0, max_iter=10000, random_state=42))
])

# === 4. Learning Curve (solo su training!) ===
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



# === 5. Plot Learning Curve ===
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

# === 6. Valutazione finale su test ===
pipeline.fit(X_train.values, y_train.values)
y_pred = pipeline.predict(X_test.values)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, digits=3)

print("\nEVALUATION ON TEST SET")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# === Save metrics to file ===
with open(f"File/training/{name_try}/metrics.txt", "w") as f:
    f.write(f"model name: {model_name}\n")
    f.write(f"dataset name: {version}\n")
    f.write("EVALUATION ON TEST SET\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)


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