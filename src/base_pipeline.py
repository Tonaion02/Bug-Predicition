import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)



df = pd.read_csv("File/ActiveMQ_input_base.csv")
column_names = df.columns.tolist()

print(df.info())
print(df["bug"].value_counts(normalize=True))  # distribuzione classi

# Analisi squilibrio
sns.countplot(data=df, x='bug')
plt.ylabel("frequenza assoluta")
plt.savefig("File/bug_class_frquency.png")


# Statistic Analysis
summary = df.describe()
for column_name in column_names:
    if column_name not in ["commitdate"]:
        print(column_name)
        print(summary[column_name])
        # Crea una copia del DataFrame senza la colonna 'commitdate'

#calcolo della correlazione
df_no_commitdate = df.drop(columns=["commitdate"], errors="ignore")
print(df_no_commitdate.corr()["bug"].sort_values(ascending=False))
# Variabile numerica da confrontare con 'bug'
numerical_col = "npt"

# --- 1. T-TEST ---
group0 = df_no_commitdate[df_no_commitdate["bug"] == 0][numerical_col]
group1 = df_no_commitdate[df_no_commitdate["bug"] == 1][numerical_col]

t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
print(f"T-test: t = {t_stat:.4f}, p = {p_val:.4f}")
if p_val < 0.05:
    print("→ Differenza statisticamente significativa (p < 0.05)")
else:
    print("→ Nessuna differenza significativa (p ≥ 0.05)")

# --- 2. ETA SQUARED (manuale) ---
grand_mean = df_no_commitdate[numerical_col].mean()

ss_between = (
    len(group0) * (group0.mean() - grand_mean) ** 2 +
    len(group1) * (group1.mean() - grand_mean) ** 2
)
ss_total = ((df_no_commitdate[numerical_col] - grand_mean) ** 2).sum()
eta_squared = ss_between / ss_total

print(f"Eta Squared (bug vs {numerical_col}): {eta_squared:.4f}")

# --- 3. BOXPLOT ---
sns.stripplot(data=df_no_commitdate, x="bug", y=numerical_col, jitter=0, alpha=0.5)
plt.title(f"Dot Plot: {numerical_col} vs bug")
plt.xlabel("bug (0 = no, 1 = sì)")
plt.ylabel(numerical_col)
plt.ylim([0, 150])
plt.savefig("File/boxplot.png")