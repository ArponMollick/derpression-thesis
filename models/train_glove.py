import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

train_df = pd.read_csv("data/reddit/train.csv")
val_df = pd.read_csv("data/reddit/validation.csv")
test_df = pd.read_csv("data/reddit/test.csv")

X_train = train_df["clean_text"].astype(str)
y_train = train_df["is_depression"]

X_val = val_df["clean_text"].astype(str)
y_val = val_df["is_depression"]

X_test = test_df["clean_text"].astype(str)
y_test = test_df["is_depression"]

# --------------------------------------------------
# 2. Load GloVe embeddings
# --------------------------------------------------

print("Loading GloVe embeddings...")

glove_path = "embeddings/glove.6B.100d.txt"

glove = {}

with open(glove_path, "r", encoding="utf-8") as file:
    for line in file:
        values = line.rstrip().split(" ")
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove[word] = vector

print("GloVe vocabulary size:", len(glove))

# --------------------------------------------------
# 3. Convert documents into GloVe vectors
# --------------------------------------------------

def document_vector(text, glove, vector_size=100):
    words = text.split()

    vectors = []

    for word in words:
        if word in glove:
            vectors.append(glove[word])

    if len(vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


print("\nCreating GloVe document vectors...")

X_train_glove = np.array([
    document_vector(text, glove)
    for text in X_train
])

X_val_glove = np.array([
    document_vector(text, glove)
    for text in X_val
])

X_test_glove = np.array([
    document_vector(text, glove)
    for text in X_test
])

print("\nGloVe feature shape:")
print("Training:", X_train_glove.shape)
print("Validation:", X_val_glove.shape)
print("Testing:", X_test_glove.shape)

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Train
    model.fit(X_train_glove, y_train)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    val_predictions = model.predict(X_val_glove)

    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(X_val_glove)[:, 1]
    else:
        val_scores = model.decision_function(X_val_glove)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)
    val_auc = roc_auc_score(y_val, val_scores)

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    test_predictions = model.predict(X_test_glove)

    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(X_test_glove)[:, 1]
    else:
        test_scores = model.decision_function(X_test_glove)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    test_auc = roc_auc_score(y_test, test_scores)

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n--- VALIDATION ---")

    print("Accuracy:", val_accuracy)
    print("Precision:", val_precision)
    print("Recall:", val_recall)
    print("F1 Score:", val_f1)
    print("AUC-ROC:", val_auc)

    print("\n--- TEST ---")

    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("AUC-ROC:", test_auc)

    print("\n--- CLASSIFICATION REPORT ---")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Non-Depression",
                "Depression"
            ]
        )
    )

    print("--- CONFUSION MATRIX ---")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({

        "Model": model_name,

        "Validation Accuracy": val_accuracy,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall,
        "Validation F1": val_f1,
        "Validation AUC": val_auc,

        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1,
        "Test AUC": test_auc
    })

# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/glove_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL GLOVE RESULTS")
print("=" * 60)

print(
    results_df[
        [
            "Model",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1",
            "Test AUC"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:")
print("results/glove_results.csv")