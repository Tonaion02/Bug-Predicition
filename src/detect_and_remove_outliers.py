#=========================================================================================
#---------------------------------------------------------------------------------------
#   Detect and Remove outliers
#---------------------------------------------------------------------------------------
#=========================================================================================
import pandas as pd
import numpy as np
import os





file = "Camel"
path_to_data_set = f"File/{file}_input_clean.csv"
path_to_clean_data_set = f"File/{file}_input_clean.csv"
clean_data_set = True
column_name = "npt"



# Load dataset (START)
df = pd.read_csv(path_to_data_set)
df = df.dropna(subset=["npt"])

# Exclude useless features
columns_to_drop = ["transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp"]
# df = df.drop(columns=columns_to_drop)
# Load dataset (END)



# Using IRQ(interquartile range) (START)
data = df[column_name]  # la colonna numerica di interesse

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


if clean_data_set == True:
    # Filtering of outliers
    df_clean = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    # Saving result in a new file
    df_clean.to_csv(path_to_clean_data_set, index=False)
# Using IRQ(interquartile range) (END)