"""
Advanced Email/SMS Spam Detection (Beginner-Friendly)
-----------------------------------------------------
This script builds a stronger spam detection project by:
1) Automatically using a large real-world SMS dataset (5k+ messages)
2) Applying better text preprocessing
3) Using TF-IDF features
4) Training and comparing multiple machine learning models
"""

import os
import re
import string
import urllib.request
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.stem import PorterStemmer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def ensure_directories() -> None:
    """Create project folders if they do not already exist."""
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def download_sms_dataset(dataset_path: str) -> None:
    """
    Download the real-world SMS Spam Collection dataset (tab-separated file).
    Source used here contains 5,572 labeled messages.
    """
    dataset_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    print(f"[INFO] Downloading dataset from: {dataset_url}")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print(f"[INFO] Dataset downloaded and saved to: {dataset_path}")


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load and validate dataset.
    If dataset is missing, it tries to download automatically.
    """
    if not os.path.exists(dataset_path):
        try:
            download_sms_dataset(dataset_path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download dataset automatically.\n"
                "Please manually download SMS Spam Collection and place it at:\n"
                f"  {dataset_path}\n"
                "Expected format: tab-separated with columns [label, message]."
            ) from exc

    # The downloaded file is tab-separated and does not include a header row.
    df = pd.read_csv(dataset_path, sep="\t", header=None, names=["label", "message"])

    # Standard cleanup for safe training.
    df = df.dropna().copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    if len(df) < 1000:
        print(
            "[WARNING] Dataset seems small. For better accuracy, use the full SMS Spam Collection dataset."
        )
    return df


def preprocess_text(text: str, stop_words: set, stemmer: PorterStemmer) -> str:
    """
    Preprocess one message using beginner-friendly NLP steps:
      1) Lowercase conversion
      2) URL removal
      3) Punctuation removal
      4) Number removal
      5) Stopword removal
      6) Stemming (Porter Stemmer)
    """
    # 1) Lowercase conversion
    text = text.lower()

    # 2) Remove URLs like http://..., https://..., www....
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3) Remove punctuation symbols
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4) Remove numbers and any remaining non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize spaces to avoid empty tokens
    text = re.sub(r"\s+", " ", text).strip()

    # 5) Remove stopwords and 6) apply stemming
    tokens = text.split()
    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            clean_tokens.append(stemmer.stem(token))

    return " ".join(clean_tokens)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """Save label count chart (ham vs spam)."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="label", palette="Set2")
    plt.title("Label Distribution (Ham vs Spam)")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/label_distribution.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, model_name: str) -> None:
    """Save confusion matrix for the best model."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=300)
    plt.close()


def plot_model_accuracy(model_results: dict) -> None:
    """Save accuracy comparison chart for all models."""
    model_names = list(model_results.keys())
    accuracies = [model_results[name]["accuracy"] for name in model_names]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=accuracies, palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.8, 1.0)
    for i, value in enumerate(accuracies):
        plt.text(i, value + 0.003, f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig("plots/model_accuracy_comparison.png", dpi=300)
    plt.close()


def plot_top_spam_words(df: pd.DataFrame, top_n: int = 15) -> None:
    """Save chart of most frequent words found in spam messages."""
    spam_messages = df[df["label"] == "spam"]["clean_message"]
    all_spam_words = " ".join(spam_messages).split()

    if not all_spam_words:
        return

    word_counts = Counter(all_spam_words)
    top_words = word_counts.most_common(top_n)

    words = [item[0] for item in top_words]
    counts = [item[1] for item in top_words]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette="magma")
    plt.title("Top Spam Words (After Preprocessing)")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig("plots/top_spam_words.png", dpi=300)
    plt.close()


def evaluate_model(model_name: str, model, x_train, x_test, y_train, y_test) -> dict:
    """
    Train model, predict test labels, and collect evaluation metrics.
    Returns all useful outputs in one dictionary.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

    print("\n" + "=" * 25 + f" {model_name} " + "=" * 25)
    print(f"Accuracy Score: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def save_artifacts(vectorizer, model) -> None:
    """
    Save trained TF-IDF vectorizer and best model using joblib.
    These files can be loaded later for instant prediction.
    """
    vectorizer_path = "models/tfidf_vectorizer.joblib"
    model_path = "models/best_model_svm.joblib"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    print("\n[STEP 8] Saved model artifacts in 'models/' folder:")
    print(f" - {vectorizer_path}")
    print(f" - {model_path}")


def load_artifacts():
    """
    Load saved TF-IDF vectorizer and trained model from disk.
    Raises an easy-to-understand error if files are missing.
    """
    vectorizer_path = "models/tfidf_vectorizer.joblib"
    model_path = "models/best_model_svm.joblib"

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Saved model files not found. Please run training first using: python main.py"
        )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def run_cli_predictor(stop_words: set, stemmer: PorterStemmer) -> None:
    """
    Simple command-line predictor:
    - User types a custom message
    - Model instantly predicts spam/ham
    - User can type 'exit' to stop
    """
    print("\n" + "=" * 25 + " LIVE CLI PREDICTOR " + "=" * 25)
    print("Type your own message to predict spam/ham.")
    print("Type 'exit' to close predictor.\n")

    try:
        vectorizer, model = load_artifacts()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    while True:
        user_message = input("Enter message: ").strip()
        if user_message.lower() == "exit":
            print("Exiting predictor. Thank you!")
            break

        if not user_message:
            print("Please type a non-empty message.")
            continue

        clean_message = preprocess_text(user_message, stop_words, stemmer)
        message_vector = vectorizer.transform([clean_message])
        prediction = model.predict(message_vector)[0]
        label = "spam" if prediction == 1 else "ham"

        print(f"Prediction: {label}\n")


def main() -> None:
    print("=" * 80)
    print("EMAIL SPAM DETECTION WITH TF-IDF + MULTIPLE ML MODELS")
    print("=" * 80)

    # Step 0: Ensure folders exist
    ensure_directories()

    # Step 1: Load large real-world dataset
    dataset_path = "dataset/sms_spam_collection.tsv"
    df = load_dataset(dataset_path)
    print(f"\n[STEP 1] Loaded dataset: {dataset_path}")
    print(f"Total messages: {len(df)}")
    print(df.head(5))

    # Step 2: Text preprocessing
    stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)
    stemmer = PorterStemmer()
    df["clean_message"] = df["message"].astype(str).apply(
        lambda text: preprocess_text(text, stop_words, stemmer)
    )
    print(
        "\n[STEP 2] Preprocessing complete (lowercase, URL/punctuation/number removal, stopwords, stemming)."
    )

    # Step 3: Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x = vectorizer.fit_transform(df["clean_message"])
    y = df["label"].map({"ham": 0, "spam": 1})

    print("\n[STEP 3] TF-IDF vectorization complete.")
    print(f"Feature matrix shape: {x.shape}")

    # Step 4: Train-test split with stratify
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\n[STEP 4] Train-test split complete with stratify=y.")
    print(f"Train size: {x_train.shape[0]}")
    print(f"Test size: {x_test.shape[0]}")

    # Step 5: Train and compare multiple models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Support Vector Machine (LinearSVC)": LinearSVC(),
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(
            model_name, model, x_train, x_test, y_train, y_test
        )

    # Step 6: Find best performing model based on accuracy
    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    best_model = results[best_model_name]["model"]
    best_accuracy = results[best_model_name]["accuracy"]
    best_cm = results[best_model_name]["confusion_matrix"]

    print("\n" + "=" * 30 + " BEST MODEL " + "=" * 30)
    print(f"Best Performing Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    # Step 7: Save required visualizations
    plot_label_distribution(df)
    plot_confusion_matrix(best_cm, best_model_name)
    plot_model_accuracy(results)
    plot_top_spam_words(df)

    print("\n[STEP 7] Plots saved in 'plots/' folder:")
    print(" - plots/label_distribution.png")
    print(" - plots/confusion_matrix.png")
    print(" - plots/model_accuracy_comparison.png")
    print(" - plots/top_spam_words.png")

    # Step 8: Save vectorizer + best model as joblib files.
    # Requirement asks to save best model as SVM, so we save trained SVM explicitly.
    svm_model_name = "Support Vector Machine (LinearSVC)"
    if svm_model_name in results:
        save_artifacts(vectorizer, results[svm_model_name]["model"])
    else:
        save_artifacts(vectorizer, best_model)

    # Step 9: Sample predictions on custom messages
    sample_messages = [
        "Congratulations! You won a free ticket. Click now to claim your reward.",
        "Hey, can we meet tomorrow to discuss the internship presentation?",
        "URGENT! Your account is suspended. Verify now at http://fake-link.com",
        "Please send the project files before 6 PM.",
    ]

    clean_samples = [preprocess_text(msg, stop_words, stemmer) for msg in sample_messages]
    sample_vectors = vectorizer.transform(clean_samples)
    sample_predictions = best_model.predict(sample_vectors)

    print("\n" + "=" * 28 + " SAMPLE PREDICTIONS " + "=" * 28)
    for message, pred in zip(sample_messages, sample_predictions):
        label = "spam" if pred == 1 else "ham"
        print(f"Message: {message}")
        print(f"Predicted Label: {label}")
        print("-" * 80)

    # Step 10: Interactive command-line predictor
    run_cli_predictor(stop_words, stemmer)

    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    main()
