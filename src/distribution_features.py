import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind





version = "base"
path_to_summary_file = f"File/distributions/summary_{version}.md"
path_to_data_set = f"File/ActiveMQ_input_{version}.csv"
drop_columns = True



# Load dataset (START)
df = pd.read_csv(path_to_data_set)

print(len(df))

# Exclude useless features
columns_to_drop = ["transactionid", "commitdate", "sexp", "ns", "ndev", "nm" , "rexp"]
if drop_columns == True:
    df = df.drop(columns=columns_to_drop)
# Load dataset (END)

column_names = df.columns.tolist()

# Create a directory for images of distributions
os.makedirs("File/distributions", exist_ok=True)



# Summary for all the features (START)
summary = df.describe()

with open(path_to_summary_file, "w") as f:
    for column_name in column_names:
        if column_name not in columns_to_drop:
            f.write(f"\n=== {column_name} ===\n")
            f.write(str(summary[column_name]))
            f.write("\n")
# Summary for all the features (END)

format = "png"

# Distribution of all features (START)

no_histograms = ["fix", "pd"]

x_limits = {"exp": (0, 100), "lt": (0, 20000), "ndev": (0, 100), "nf" : (0, 100), "rexp" : (0, 5)}

for col in df.select_dtypes(include='number').columns:
    if col not in no_histograms:
        x_lim = None
        x_lim = x_limits.get(col)

        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuzione di {col}')

        if x_lim:
            plt.xlim(x_lim)

        # Create the name of the feature
        filename = f"File/distributions/distribuzione_{col}_{version}.{format}".replace(" ", "_")

        # Save the figure
        plt.savefig(filename, format=format, bbox_inches='tight')

        # Show the figure
        # plt.show()

for col in no_histograms:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribuzione di {col}')
        
    # Create the name of the feature
    filename = f"File/distributions/distribuzione_{col}_{version}.{format}".replace(" ", "_")
    
    # Save the figure
    plt.savefig(filename, format=format, bbox_inches='tight')
    
    # Show the figure
    # plt.show()

# Distribution of all features (END)