# 1. Importare le librerie
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib


version = "clean"

# 2. Caricare il dataset Iris
df = pd.read_csv(f"File/ActiveMQ_input_{version}.csv")
# columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp", "bug"]
columns_to_drop = ["useless", "bug"]
df = df.dropna(subset=["npt"])
X = df.drop(columns=columns_to_drop)
y = df["bug"]

print(y)

# 3. Dividere in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Creare il modello SVM con kernel lineare
model = SVC(kernel='linear')

# 5. Addestrare il modello
model.fit(X_train, y_train)

# 6. Fare previsioni sul test set
y_pred = model.predict(X_test)

# 7. Valutare il modello
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza: {accuracy * 100:.2f}%")

# Report più dettagliato
print("\nReport di classificazione:")
print(classification_report(y_test, y_pred))



# Save the model
joblib.dump(model, f"model_svm_{version}.pkl")



# 7. Curva di apprendimento
'''
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# 8. Plot della curva
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation score")
plt.title("Learning Curve (SVM)")
plt.xlabel("Numero di campioni di addestramento")
plt.ylabel("Accuratezza")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.show()
'''