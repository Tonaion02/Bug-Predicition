import pandas as pd
import os
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency





path_to_data_set_1 = "File/ActiveMQ_input_base.csv"
path_to_data_set_2 = "File/ActiveMQ_input_oversampled.csv"
feature_1 = "fix"
feature_2 = "fix"
# Choose modality between:
#   - KS (Kolmogorov-Smirnov)
#   - AD (Anderson-Darling)
#   - CQ (Chi-Quadro)
modality = "CQ" 



# Load datasets (START)
df_1 = pd.read_csv(path_to_data_set_1)

df_1 = df_1.dropna(subset=["npt"])

df_2 = pd.read_csv(path_to_data_set_2)
# Load datasets (END)



x1 = df_1[feature_1]
x2 = df_2[feature_2]


if modality == "KS":
    stat, p_value = ks_2samp(x1, x2)
    print(f"{feature_1} from dataset:{path_to_data_set_1} and {feature_2} from dataset:{path_to_data_set_2} have:")
    print(f"stat: {stat}")
    print(f"p_value: {p_value}")
    if p_value < 0.05:
        print("Different distributions")
    else:
        print("Similar distributions")

elif modality == "AD":
    pass

elif modality == "CQ":
    # Building contigency table
    contingency = pd.crosstab(pd.Series(x1, name=feature_1),
                              pd.Series(x2, name=feature_2))

    # Test del chi-quadro
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(f"{feature_1} from dataset:{path_to_data_set_1} and {feature_2} from dataset:{path_to_data_set_2} have:")    
    print("Chi-squared statistic:", chi2)
    print("p-value:", p)
    print("Degrees of freedom:", dof)

    # Automatic interpretation
    alpha = 0.05
    if p < alpha:
        print("Distributions are very different")
    else:
        print("Non ci sono prove sufficienti per dire che le distribuzioni siano diverse (non rifiuto H0).")

else:
    print(f"Error not valid value of modality:{modality}")