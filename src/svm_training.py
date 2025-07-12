import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.svm import SVC





# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM model with linear kernel (no scaling)
svm_model = SVC(kernel='linear', C=1.0)

# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    svm_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)

# Calculate mean and standard deviation for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
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


















# from sklearn.model_selection import train_test_split, learning_curve
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import joblib
# import os





# version = "base"
# directory_produced_data = "base"

# # Load dataset (START)
# df = pd.read_csv(f"File/ActiveMQ_input_{version}.csv")
# columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp", "bug"]
# # columns_to_drop = ["useless", "bug"]
# # columns_to_drop = ["bug"]
# df = df.dropna(subset=["npt"])
# X = df.drop(columns=columns_to_drop)
# y = df["bug"]
# # Load dataset (END)

# print(y)



# # Split dataset in training set e test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = SVC(kernel='linear')
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # Report più dettagliato
# print("\nReport classification:")
# print(classification_report(y_test, y_pred))


# # Create directory
# os.mkdir(directory_produced_data)

# # Save the model
# joblib.dump(model, f"model_svm_{version}.pkl")
# X.to_csv(os.path.join(directory_produced_data, "X.csv"), index=False)
# y.to_csv(os.path.join(directory_produced_data, "y.csv"), index=False)
# X_train.to_csv(os.path.join(directory_produced_data, "X_train.csv"), index=False)
# X_test.to_csv(os.path.join(directory_produced_data, "X_test.csv"), index=False)
# y_train.to_csv(os.path.join(directory_produced_data, "y_train.csv"), index=False)
# y_test.to_csv(os.path.join(directory_produced_data, "y_test.csv"), index=False)



# # Make learning curves
# train_sizes, train_scores, test_scores = learning_curve(
#     model,
#     X,
#     y,
#     cv=5,
#     scoring='accuracy',
#     train_sizes=np.linspace(0.1, 1.0, 10),
#     n_jobs=-1
# )



# # WARNING: returned values are negative
# train_loss_mean = -np.mean(train_scores, axis=1)
# test_loss_mean = -np.mean(test_scores, axis=1)

# # Plot of loss curves
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_loss_mean, 'o-', label="Training Loss")
# plt.plot(train_sizes, test_loss_mean, 'o-', label="Validation Loss")
# plt.title("Learning Curve - Loss (SVM)")
# plt.xlabel("Numero di campioni di addestramento")
# plt.ylabel("Log Loss")
# plt.grid(True)
# plt.legend(loc="best")
# plt.tight_layout()
# plt.show()