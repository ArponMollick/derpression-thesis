import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "data/reddit/depression_dataset_reddit_cleaned.csv"
df = pd.read_csv(file_path)

print("Original dataset:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# Remove rows with missing text or labels
df = df.dropna(subset=["clean_text", "is_depression"])

print("After removing missing values:", df.shape)

# Separate features and labels
X = df["clean_text"]
y = df["is_depression"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\n--- FINAL SPLIT ---")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

print("\n--- TRAINING LABELS ---")
print(y_train.value_counts())

print("\n--- VALIDATION LABELS ---")
print(y_val.value_counts())

print("\n--- TESTING LABELS ---")
print(y_test.value_counts())

# Save the datasets
train_df = pd.DataFrame({
    "clean_text": X_train,
    "is_depression": y_train
})

val_df = pd.DataFrame({
    "clean_text": X_val,
    "is_depression": y_val
})

test_df = pd.DataFrame({
    "clean_text": X_test,
    "is_depression": y_test
})

train_df.to_csv("data/reddit/train.csv", index=False)
val_df.to_csv("data/reddit/validation.csv", index=False)
test_df.to_csv("data/reddit/test.csv", index=False)

print("\nDataset splits saved successfully.")