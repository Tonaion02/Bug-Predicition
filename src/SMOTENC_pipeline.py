import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC





version = "base"
new_version = "oversampled"
path_to_data_set = f"File/ActiveMQ_input_{version}.csv"
path_to_new_data_set = f"File/ActiveMQ_input_{new_version}.csv"



# Load data set (START)
df = pd.read_csv(path_to_data_set)

# Exclude useless features
columns_to_drop = ["useless", "transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp"]
df = df.drop(columns=columns_to_drop)

# Exclude other features

# Eliminate entries with NaN entries
df = df.dropna(subset=["npt"])

# Load data set (END)



# Separate dependent and independent features
dependent_feature_name = "bug"
X = df.drop(dependent_feature_name, axis=1)
y = df[dependent_feature_name]



# Encoding of categorial features
categorical_features = ["fix"]
for categorical_feature in categorical_features:
    le = LabelEncoder()
    X[categorical_feature] = le.fit_transform(X[categorical_feature])

# Specify indexes of categorical features
categorical_features_indexes = [X.columns.get_loc(categorical_feature)] 

# SMOTENC: oversampling of minority features
smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

# Results
print("Shape originale:", X.shape, "Distribuzione:", y.value_counts().to_dict())
print("Shape dopo SMOTENC:", X_resampled.shape, "Distribuzione:", dict(pd.Series(y_resampled).value_counts()))



# Save new data set in another file
new_df = pd.DataFrame(X_resampled, columns=X.columns)
new_df["bug"] = y_resampled
new_df.to_csv(path_to_new_data_set, index=False)