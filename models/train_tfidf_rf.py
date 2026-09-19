import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

X_train = train_df["clean_text"]
y_train = train_df["is_depression"]

X_val = val_df["clean_text"]
y_val = val_df["is_depression"]

X_test = test_df["clean_text"]
y_test = test_df["is_depression"]

# --------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF feature shape:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Testing:", X_test_tfidf.shape)

# --------------------------------------------------
# 3. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

# --------------------------------------------------
# 4. Validation
# --------------------------------------------------

val_predictions = model.predict(X_val_tfidf)
val_probabilities = model.predict_proba(X_val_tfidf)[:, 1]

print("\n===== VALIDATION RESULTS =====")

print("Accuracy:",
      accuracy_score(y_val, val_predictions))

print("Precision:",
      precision_score(y_val, val_predictions))

print("Recall:",
      recall_score(y_val, val_predictions))

print("F1 Score:",
      f1_score(y_val, val_predictions))

print("AUC-ROC:",
      roc_auc_score(y_val, val_probabilities))

# --------------------------------------------------
# 5. Final test evaluation
# --------------------------------------------------

test_predictions = model.predict(X_test_tfidf)
test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

print("\n===== TEST RESULTS =====")

print("Accuracy:",
      accuracy_score(y_test, test_predictions))

print("Precision:",
      precision_score(y_test, test_predictions))

print("Recall:",
      recall_score(y_test, test_predictions))

print("F1 Score:",
      f1_score(y_test, test_predictions))

print("AUC-ROC:",
      roc_auc_score(y_test, test_probabilities))

# --------------------------------------------------
# 6. Detailed results
# --------------------------------------------------

print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(
    y_test,
    test_predictions,
    target_names=["Non-Depression", "Depression"]
))

print("\n===== CONFUSION MATRIX =====")

print(confusion_matrix(
    y_test,
    test_predictions
))