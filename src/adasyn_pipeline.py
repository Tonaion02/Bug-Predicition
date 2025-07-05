import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN
from collections import Counter





version = "clean"
new_version = "oversampled_adasyn"
path_to_data_set = f"File/ActiveMQ_input_{version}.csv"
path_to_new_data_set = f"File/ActiveMQ_input_{new_version}.csv"

# columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp"]
columns_to_drop = ["useless"]





# Load data set (START)
df = pd.read_csv(path_to_data_set)

# Exclude useless features
df = df.drop(columns=columns_to_drop)

# Eliminate entries with NaN entries
df = df.dropna(subset=["npt"])
# Load data set (END)



# Separate dependent and independent features
dependent_feature_name = "bug"

X = df.drop(dependent_feature_name, axis=1)
y = df[dependent_feature_name]

# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)



# Results
print("Original Shape:", X.shape, "Distribustion:", Counter(y))
print("Shape after ADASYN:", X_resampled.shape, "Distribustion:", Counter(y_resampled))



# Save new data set in another file (START)
new_df = pd.DataFrame(X_resampled, columns=X.columns)
new_df["bug"] = y_resampled
new_df.to_csv(path_to_new_data_set, index=False)
# Save new data set in another file (END)