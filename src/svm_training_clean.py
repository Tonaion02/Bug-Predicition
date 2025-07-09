from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib



version = "clean"
# Load dataset
df = pd.read_csv(f"File/ActiveMQ_input_base.csv")
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp"]
df = df.drop(columns=columns_to_drop)
df = df.dropna(subset=["npt"])

print(f"df size at the start: {len(df)}")


# Split dataset
X = df.drop(columns=["bug"])
y = df["bug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remerge X_train, y_train
df_merged = pd.concat([X_train, y_train], axis=1)
print(f"df_merged size before clean: {len(df_merged)}")


# Make clean
columns_to_clean = ["nf" , "entropy" , "la" , "ld" , "lt" , "pd" , "npt" , "exp"]
for column_name in columns_to_clean:
    # Using IRQ(interquartile range) (START)
    data = df_merged[column_name]  # la colonna numerica di interesse

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    print("Report:")
    print(f"lower bound: {lower_bound} upper_bound: {upper_bound}")
    print(f"Number of outliers: {len(outliers)}")

    df_merged = df_merged[(df_merged[column_name] >= lower_bound) & (df_merged[column_name] <= upper_bound)]

print(f"df_merged size after clean: {len(df_merged)}")

# Re-Split
X_train = df_merged.drop(columns=["bug"])
y_train = df_merged["bug"]

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