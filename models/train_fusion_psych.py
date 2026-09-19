import pandas as pd
import numpy as np
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("data/reddit/train.csv")
val_df = pd.read_csv("data/reddit/validation.csv")
test_df = pd.read_csv("data/reddit/test.csv")

X_train = train_df["clean_text"].astype(str)
y_train = train_df["is_depression"]

X_val = val_df["clean_text"].astype(str)
y_val = val_df["is_depression"]

X_test = test_df["clean_text"].astype(str)
y_test = test_df["is_depression"]


# ============================================================
# 2. TF-IDF
# ============================================================

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


# ============================================================
# 3. VADER
# ============================================================

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

print("VADER shape:")
print("Training:", X_train_vader.shape)
print("Validation:", X_val_vader.shape)
print("Testing:", X_test_vader.shape)


# ============================================================
# 4. PSYCHOLINGUISTIC FEATURES
# ============================================================

print("\nCreating psycholinguistic features...")


positive_words = {
    "happy", "joy", "good", "great", "love", "wonderful",
    "excellent", "amazing", "hope", "hopeful", "fun",
    "better", "positive", "excited", "grateful", "thankful",
    "success", "smile", "laugh", "peace"
}

negative_words = {
    "sad", "bad", "hate", "terrible", "awful", "depressed",
    "depression", "hopeless", "worthless", "lonely", "alone",
    "pain", "hurt", "cry", "crying", "failure", "fail",
    "angry", "anxiety", "anxious", "stress", "stressed",
    "suicide", "die", "death", "empty", "tired", "exhausted"
}

emotion_words = {
    "happy", "sad", "angry", "fear", "afraid", "love",
    "hate", "joy", "pain", "anxiety", "anxious", "hope",
    "hopeless", "lonely", "excited", "worried", "stress",
    "stressed", "cry", "crying", "laugh", "laughing"
}

negation_words = {
    "not", "no", "never", "nothing", "nobody",
    "nowhere", "neither", "nor", "cannot", "can't",
    "dont", "don't", "didnt", "didn't", "isnt", "isn't",
    "wasnt", "wasn't", "wont", "won't", "wouldnt", "wouldn't"
}

first_person_words = {
    "i", "me", "my", "mine", "myself"
}

second_person_words = {
    "you", "your", "yours", "yourself"
}

third_person_words = {
    "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "theirs"
}


def get_psych_features(text):

    words = re.findall(
        r"\b[a-zA-Z']+\b",
        text.lower()
    )

    word_count = len(words)

    if word_count == 0:

        return [
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ]

    positive_count = sum(
        word in positive_words
        for word in words
    )

    negative_count = sum(
        word in negative_words
        for word in words
    )

    emotion_count = sum(
        word in emotion_words
        for word in words
    )

    negation_count = sum(
        word in negation_words
        for word in words
    )

    first_person_count = sum(
        word in first_person_words
        for word in words
    )

    second_person_count = sum(
        word in second_person_words
        for word in words
    )

    third_person_count = sum(
        word in third_person_words
        for word in words
    )

    sentence_count = max(
        len(re.findall(r"[.!?]+", text)),
        1
    )

    average_word_length = (
        sum(len(word) for word in words)
        / word_count
    )

    unique_words = len(set(words))

    lexical_diversity = (
        unique_words / word_count
    )

    exclamation_count = text.count("!")

    question_count = text.count("?")

    uppercase_count = sum(
        character.isupper()
        for character in text
    )

    letter_count = sum(
        character.isalpha()
        for character in text
    )

    uppercase_ratio = (
        uppercase_count / letter_count
        if letter_count > 0
        else 0
    )

    return [
        positive_count / word_count,
        negative_count / word_count,
        emotion_count / word_count,
        negation_count / word_count,
        first_person_count / word_count,
        second_person_count / word_count,
        third_person_count / word_count,
        sentence_count,
        word_count,
        average_word_length,
        lexical_diversity,
        exclamation_count,
        question_count,
        uppercase_ratio
    ]


X_train_psych = np.array([
    get_psych_features(text)
    for text in X_train
])

X_val_psych = np.array([
    get_psych_features(text)
    for text in X_val
])

X_test_psych = np.array([
    get_psych_features(text)
    for text in X_test
])


print("Psycholinguistic feature shape:")
print("Training:", X_train_psych.shape)
print("Validation:", X_val_psych.shape)
print("Testing:", X_test_psych.shape)


# ============================================================
# 5. STANDARDIZE PSYCHOLINGUISTIC FEATURES
# ============================================================

print("\nStandardizing psycholinguistic features...")

scaler = StandardScaler()

X_train_psych = scaler.fit_transform(
    X_train_psych
)

X_val_psych = scaler.transform(
    X_val_psych
)

X_test_psych = scaler.transform(
    X_test_psych
)


# ============================================================
# 6. FEATURE FUSION
# ============================================================

print(
    "\nCombining TF-IDF + VADER + "
    "Psycholinguistic features..."
)

X_train_combined = hstack([
    X_train_tfidf,
    csr_matrix(X_train_vader),
    csr_matrix(X_train_psych)
]).tocsr()

X_val_combined = hstack([
    X_val_tfidf,
    csr_matrix(X_val_vader),
    csr_matrix(X_val_psych)
]).tocsr()

X_test_combined = hstack([
    X_test_tfidf,
    csr_matrix(X_test_vader),
    csr_matrix(X_test_psych)
]).tocsr()


print("Final combined feature shape:")
print("Training:", X_train_combined.shape)
print("Validation:", X_val_combined.shape)
print("Testing:", X_test_combined.shape)


# ============================================================
# 7. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": LinearSVC(
        C=1.0,
        max_iter=10000,
        tol=1e-5,
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


# ============================================================
# 8. TRAINING AND EVALUATION
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    if model_name == "Naive Bayes":

        train_features = X_train_combined.toarray()
        val_features = X_val_combined.toarray()
        test_features = X_test_combined.toarray()

    else:

        train_features = X_train_combined
        val_features = X_val_combined
        test_features = X_test_combined


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        train_features,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = model.predict(
        val_features
    )

    if hasattr(model, "predict_proba"):

        val_scores = model.predict_proba(
            val_features
        )[:, 1]

    else:

        val_scores = model.decision_function(
            val_features
        )


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


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = model.predict(
        test_features
    )

    if hasattr(model, "predict_proba"):

        test_scores = model.predict_proba(
            test_features
        )[:, 1]

    else:

        test_scores = model.decision_function(
            test_features
        )


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


    # --------------------------------------------------------
    # PRINT VALIDATION
    # --------------------------------------------------------

    print("\n--- VALIDATION ---")

    print("Accuracy:", val_accuracy)
    print("Precision:", val_precision)
    print("Recall:", val_recall)
    print("F1 Score:", val_f1)
    print("AUC-ROC:", val_auc)


    # --------------------------------------------------------
    # PRINT TEST
    # --------------------------------------------------------

    print("\n--- TEST ---")

    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("AUC-ROC:", test_auc)


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("--- CONFUSION MATRIX ---")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

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


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/tfidf_vader_psych_results_final.csv",
    index=False
)


print("\n" + "=" * 60)
print("FINAL TF-IDF + VADER + PSYCHOLINGUISTIC RESULTS")
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
print(
    "results/tfidf_vader_psych_results_final.csv"
)