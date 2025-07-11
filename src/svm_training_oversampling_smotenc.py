from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib





version = "base"
name_of_model = "oversampled_smotenc_before_split"
categorical_features = ["nf", "lt", "fix", "pd", "exp"]
# categorical_features = []

# Load dataset (START)
df = pd.read_csv(f"File/ActiveMQ_input_{version}.csv")
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp", "bug"]
# columns_to_drop = ["useless", "bug"]
# columns_to_drop = ["bug"]
df = df.dropna(subset=["npt"])
X = df.drop(columns=columns_to_drop)
y = df["bug"]
# Load dataset (END)

print(y)

# Split training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Oversampling with SMOTENC (START)
# Encoding of categorical features
categorical_features_indexes = []

for categorical_feature in categorical_features:
    le = LabelEncoder()
    X_train[categorical_feature] = le.fit_transform(X_train[categorical_feature])

    # Specify indexes of categorical features
    categorical_features_indexes.append(X_train.columns.get_loc(categorical_feature))

# SMOTENC: oversampling of minority features
if not categorical_features:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
else: 
    smote_nc = SMOTENC(categorical_features=categorical_features_indexes, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

# Results
print("Original Shape:", X_train.shape, "Distribustion:", y_train.value_counts().to_dict())
print("Shape after SMOTENC:", X_resampled.shape, "Distribustion:", dict(pd.Series(y_resampled).value_counts()))
# Oversampling with SMOTENC (END)





# Train model
model = SVC(kernel='linear')
model.fit(X_resampled, y_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# More detailed report
print("\nReport of classification:")
print(classification_report(y_test, y_pred))



# Save the model
joblib.dump(model, f"model_svm_{name_of_model}.pkl")



# 
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