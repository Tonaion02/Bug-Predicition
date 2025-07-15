import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind





project = "Camel"
version = "base"
path_to_dataset = f"File/{project}_input_{version}.csv" 

df = pd.read_csv(path_to_dataset)
column_names = df.columns.tolist()

print(df.info())
print(df["bug"].value_counts(normalize=True))  # distribuzione classi

# Analisi squilibrio
sns.countplot(data=df, x='bug')
plt.ylabel("frequency absolute")
plt.savefig("File/bug_class_frquency_Camel.png")