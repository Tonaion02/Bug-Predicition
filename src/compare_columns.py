import pandas as pd

version = "base"
file = "Camel"
df = pd.read_csv(f"File/{file}_input_{version}.csv")


# Confronta le colonne sexp e entropy e conta le righe con valori diversi
col1 = "sexp"
col2 = "entropy"
count = (df[col1] != df[col2]).sum()
print(f"Numero di righe in cui {col1} è diverso da {col2}: {count}")

# Confronta le colonne ns e ndev e conta le righe con valori diversi
col1 = "ns"
col2 = "ndev"
count = (df[col1] != df[col2]).sum()
print(f"Numero di righe in cui {col1} è diverso da {col2}: {count}")



# Confronta le colonne ns e exp e conta le righe con valori diversi
col1 = "ns"
col2 = "exp"
count = (df[col1] != df[col2]).sum()
print(f"Numero di righe in cui {col1} è diverso da {col2}: {count}")



# Confronta le colonne nm e rexp e conta le righe con valori diversi
col1 = "nm"
col2 = "rexp"
count = (df[col1] != df[col2]).sum()
print(f"Numero di righe in cui {col1} è diverso da {col2}: {count}")



# Confronta le colonne nm e pd e conta le righe con valori diversi
col1 = "nm"
col2 = "pd"
count = (df[col1] != df[col2]).sum()
print(f"Numero di righe in cui {col1} è diverso da {col2}: {count}")