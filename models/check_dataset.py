import pandas as pd

file_path = "data/reddit/depression_dataset_reddit_cleaned.csv"

df = pd.read_csv(file_path)

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

print("\n--- LABEL DISTRIBUTION ---")
print(df["is_depression"].value_counts())

print("\n--- LABEL PERCENTAGE ---")
print(df["is_depression"].value_counts(normalize=True) * 100)