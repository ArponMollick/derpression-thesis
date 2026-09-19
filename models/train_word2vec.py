import pandas as pd
import numpy as np

from gensim.models import Word2Vec

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
# 2. Tokenize text
# --------------------------------------------------

train_tokens = [text.split() for text in X_train]
val_tokens = [text.split() for text in X_val]
test_tokens = [text.split() for text in X_test]

# --------------------------------------------------
# 3. Train Word2Vec
# --------------------------------------------------

print("Training Word2Vec...")

word2vec = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10,
    seed=42
)

print("Word2Vec vocabulary size:", len(word2vec.wv))

# --------------------------------------------------
# 4. Convert each document into a vector
# --------------------------------------------------

def document_vector(tokens, model):
    vectors = []

    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


print("\nCreating document vectors...")

X_train_w2v = np.array([
    document_vector(tokens, word2vec)
    for tokens in train_tokens
])

X_val_w2v = np.array([
    document_vector(tokens, word2vec)
    for tokens in val_tokens
])

X_test_w2v = np.array([
    document_vector(tokens, word2vec)
    for tokens in test_tokens
])

print("\nWord2Vec feature shape:")
print("Training:", X_train_w2v.shape)
print("Validation:", X_val_w2v.shape)
print("Testing:", X_test_w2v.shape)

# --------------------------------------------------
# 5. Define models
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
# 6. Train and evaluate models
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Train
    model.fit(X_train_w2v, y_train)

    # Validation
    val_predictions = model.predict(X_val_w2v)

    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(X_val_w2v)[:, 1]
    else:
        val_scores = model.decision_function(X_val_w2v)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)
    val_auc = roc_auc_score(y_val, val_scores)

    # Test
    test_predictions = model.predict(X_test_w2v)

    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(X_test_w2v)[:, 1]
    else:
        test_scores = model.decision_function(X_test_w2v)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    test_auc = roc_auc_score(y_test, test_scores)

    # Print results
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
            target_names=["Non-Depression", "Depression"]
        )
    )

    print("--- CONFUSION MATRIX ---")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    # Store results
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
# 7. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/word2vec_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL WORD2VEC RESULTS")
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
print("results/word2vec_results.csv")