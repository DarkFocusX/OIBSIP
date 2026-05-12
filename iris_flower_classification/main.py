"""
Iris Flower Classification - Beginner Friendly Data Science Project

This script demonstrates a complete machine learning workflow:
1. Data loading
2. Data exploration
3. Data visualization
4. Data preprocessing
5. Train-test split
6. Model training
7. Prediction
8. Evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def main():
    # ------------------------------
    # Step 1: Create folder for plots
    # ------------------------------
    # We save all charts in this folder so they are easy to find.
    os.makedirs("plots", exist_ok=True)

    # ------------------------------
    # Step 2: Load Iris dataset
    # ------------------------------
    # load_iris() returns a dictionary-like object with data and metadata.
    iris = load_iris()

    # Create a DataFrame for easier analysis and visualization.
    feature_names = iris.feature_names
    df = pd.DataFrame(iris.data, columns=feature_names)

    # Add the target column (numeric class: 0, 1, 2).
    df["target"] = iris.target

    # Add a readable target name column (setosa, versicolor, virginica).
    df["species"] = df["target"].map(
        {
            0: iris.target_names[0],
            1: iris.target_names[1],
            2: iris.target_names[2],
        }
    )

    # ------------------------------
    # Step 3: Data exploration
    # ------------------------------
    print("\n=== First 5 rows of dataset ===")
    print(df.head())

    print("\n=== Dataset shape (rows, columns) ===")
    print(df.shape)

    print("\n=== Data types ===")
    print(df.dtypes)

    print("\n=== Missing values in each column ===")
    print(df.isnull().sum())

    print("\n=== Statistical summary ===")
    print(df.describe())

    print("\n=== Class distribution ===")
    print(df["species"].value_counts())

    # ------------------------------
    # Step 4: Data visualization
    # ------------------------------
    # Set a clean style for plots.
    sns.set_style("whitegrid")

    # 4A. Count plot to show class distribution.
    plt.figure(figsize=(7, 5))
    sns.countplot(x="species", data=df, palette="Set2")
    plt.title("Iris Class Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/class_distribution.png")
    plt.close()

    # 4B. Pairplot to visualize relationships between features.
    pairplot = sns.pairplot(
        df,
        hue="species",
        diag_kind="hist",
        palette="Set1",
        corner=True,
    )
    pairplot.fig.suptitle("Pairplot of Iris Features", y=1.02)
    pairplot.savefig("plots/pairplot.png")
    plt.close()

    # 4C. Correlation heatmap (numeric columns only).
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    # ------------------------------
    # Step 5: Preprocessing
    # ------------------------------
    # Separate input features (X) and target labels (y).
    X = df[feature_names].values
    y = df["target"].values

    # ------------------------------
    # Step 6: Train-test split
    # ------------------------------
    # We keep 20% data for testing and 80% for training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardize features because KNN uses distance calculations.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------
    # Step 7: Model training
    # ------------------------------
    # We use K-Nearest Neighbors (KNN), which is beginner friendly and intuitive.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)

    # ------------------------------
    # Step 8: Prediction
    # ------------------------------
    y_pred = model.predict(X_test_scaled)

    # ------------------------------
    # Step 9: Accuracy evaluation and reports
    # ------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== Accuracy Score ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualize confusion matrix using heatmap.
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # ------------------------------
    # Step 10: Show sample predictions
    # ------------------------------
    # This helps beginners understand model output clearly.
    sample_count = min(5, len(y_test))
    print("\n=== Sample Predictions ===")
    for i in range(sample_count):
        true_label = iris.target_names[y_test[i]]
        predicted_label = iris.target_names[y_pred[i]]
        print(f"Sample {i + 1}: True = {true_label}, Predicted = {predicted_label}")

    print("\nProject completed successfully.")
    print("All visualizations are saved in the 'plots' folder.")


if __name__ == "__main__":
    main()
