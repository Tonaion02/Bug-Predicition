import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE






version = "base"
new_version = "oversampled"
path_to_data_set = f"File/ActiveMQ_input_{version}.csv"
path_to_new_data_set = f"File/ActiveMQ_input_{new_version}.csv"

columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp"]
# columns_to_drop = ["useless"]

categorical_features = ["nf", "lt", "fix", "pd", "exp"]
# categorical_features = []





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
# X = df[["npt"]]
y = df[dependent_feature_name]



# Encoding of categorical features

categorical_features_indexes = []

for categorical_feature in categorical_features:
    le = LabelEncoder()
    X[categorical_feature] = le.fit_transform(X[categorical_feature])

    # Specify indexes of categorical features
    categorical_features_indexes.append(X.columns.get_loc(categorical_feature))

# SMOTENC: oversampling of minority features
if not categorical_features:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else: 
    smote_nc = SMOTENC(categorical_features=categorical_features_indexes, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

# Results
print("Original Shape:", X.shape, "Distribustion:", y.value_counts().to_dict())
print("Shape after SMOTENC:", X_resampled.shape, "Distribustion:", dict(pd.Series(y_resampled).value_counts()))



# Save new data set in another file (START)
new_df = pd.DataFrame(X_resampled, columns=X.columns)
new_df["bug"] = y_resampled
new_df.to_csv(path_to_new_data_set, index=False)
# Save new data set in another file (END)