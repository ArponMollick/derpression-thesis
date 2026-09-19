import pandas as pd
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
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
# 2. TF-IDF
# --------------------------------------------------

print("Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF shape:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Testing:", X_test_tfidf.shape)

# --------------------------------------------------
# 3. VADER sentiment features
# --------------------------------------------------

print("\nCreating VADER sentiment features...")

analyzer = SentimentIntensityAnalyzer()


def get_vader_features(text):
    scores = analyzer.polarity_scores(text)

    return [
        scores["neg"],
        scores["neu"],
        scores["pos"],
        scores["compound"]
    ]


X_train_vader = np.array([
    get_vader_features(text)
    for text in X_train
])

X_val_vader = np.array([
    get_vader_features(text)
    for text in X_val
])

X_test_vader = np.array([
    get_vader_features(text)
    for text in X_test
])

print("VADER feature shape:")
print("Training:", X_train_vader.shape)
print("Validation:", X_val_vader.shape)
print("Testing:", X_test_vader.shape)

# --------------------------------------------------
# 4. Combine TF-IDF + VADER
# --------------------------------------------------

print("\nCombining TF-IDF + VADER...")

X_train_combined = hstack([
    X_train_tfidf,
    csr_matrix(X_train_vader)
]).tocsr()

X_val_combined = hstack([
    X_val_tfidf,
    csr_matrix(X_val_vader)
]).tocsr()

X_test_combined = hstack([
    X_test_tfidf,
    csr_matrix(X_test_vader)
]).tocsr()

print("Combined feature shape:")
print("Training:", X_train_combined.shape)
print("Validation:", X_val_combined.shape)
print("Testing:", X_test_combined.shape)

# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": LinearSVC(
        C=1.0,
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
# 6. Train and evaluate
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # GaussianNB requires dense input
    if model_name == "Naive Bayes":
        train_features = X_train_combined.toarray()
        val_features = X_val_combined.toarray()
        test_features = X_test_combined.toarray()
    else:
        train_features = X_train_combined
        val_features = X_val_combined
        test_features = X_test_combined

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    model.fit(
        train_features,
        y_train
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    val_predictions = model.predict(val_features)

    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(val_features)[:, 1]
    else:
        val_scores = model.decision_function(val_features)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_precision = precision_score(
        y_val,
        val_predictions
    )

    val_recall = recall_score(
        y_val,
        val_predictions
    )

    val_f1 = f1_score(
        y_val,
        val_predictions
    )

    val_auc = roc_auc_score(
        y_val,
        val_scores
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    test_predictions = model.predict(test_features)

    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(test_features)[:, 1]
    else:
        test_scores = model.decision_function(test_features)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions
    )

    test_recall = recall_score(
        y_test,
        test_predictions
    )

    test_f1 = f1_score(
        y_test,
        test_predictions
    )

    test_auc = roc_auc_score(
        y_test,
        test_scores
    )

    # --------------------------------------------------
    # Print validation results
    # --------------------------------------------------

    print("\n--- VALIDATION ---")

    print("Accuracy:", val_accuracy)
    print("Precision:", val_precision)
    print("Recall:", val_recall)
    print("F1 Score:", val_f1)
    print("AUC-ROC:", val_auc)

    # --------------------------------------------------
    # Print test results
    # --------------------------------------------------

    print("\n--- TEST ---")

    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("AUC-ROC:", test_auc)

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

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
# 7. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/tfidf_vader_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL TF-IDF + VADER RESULTS")
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
print("results/tfidf_vader_results.csv")