# Task 5: Compare Models
# Merges resulta from all models, produces comparion bar charts, and writes a final summary table to results/final-comparison.csv.
# Also performs a brief over-fitting analysis by re-evaluating the best model on the training set and comparing with test performance.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

PREFIX="[DC: Compare Models]"


def loadResults():
    classicResults = pd.read_csv("results/classic-results.csv")
    neuralNetworkResults = pd.read_csv("results/neural-network-results.csv")

    return pd.concat([classicResults, neuralNetworkResults], ignore_index=True)


def graphComparison(dataFrame: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = sns.color_palette("Set2", len(dataFrame))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Comparison – All Metrics", fontsize=14, y=1.02)

    for ax, metric in zip(axes, metrics):
        bars = ax.barh(dataFrame["model"], dataFrame[metric], color=colors)

        ax.set_title(metric.capitalize())

        ax.set_xlim(0, 1.0)
        ax.set_xlabel(metric.capitalize())

        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    fig.savefig("graphs/model-comparison.png", dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"{PREFIX} Saved graphs/model-comparison.png")


def overfittingAnalysis():
    XTrainTfidf = joblib.load("data/x-train-tfidf.pkl")
    XTestTfidf = joblib.load("data/x-test-tfidf.pkl")
    XTrainSvd = joblib.load("data/x-train-svd.pkl")
    XTestSvd = joblib.load("data/x-test-svd.pkl")

    YTrain = np.load("data/y-train.npy")
    YTest = np.load("data/y-test.npy")

    model_files = {
        "Complement Naive Bayes": ("models/complement-naive-bayes.pkl", XTrainTfidf, XTestTfidf),
        "Logistic Regression": ("models/logistic-regression.pkl", XTrainTfidf, XTestTfidf),
        "SVM (LinearSVC)": ("models/svm-linearsvc.pkl", XTrainTfidf, XTestTfidf),
        "Neural Network (MLP)": ("models/neural-network.pkl", XTrainSvd, XTestSvd)
    }

    rows = []

    for name, (path, XTrain, XTest) in model_files.items():
        if not os.path.exists(path):
            print(f"{PREFIX} WARN: {path} not found, skipping...")
            continue

        classifier = joblib.load(path)

        f1Train = f1_score(YTrain, classifier.predict(XTrain), average="weighted", zero_division=0)
        f1Test = f1_score(YTest, classifier.predict(XTest), average="weighted", zero_division=0)

        rows.append({"model": name, "F1_train": f1Train, "F1_test": f1Test, "gap": f1Train - f1Test})

    gapDataFrame = pd.DataFrame(rows)

    print(f"\n{PREFIX} Over-fitting analysis (F1 train vs test):\n")
    print(gapDataFrame.to_string(index=False))

    gapDataFrame.to_csv("results/overfitting-analysis.csv", index=False)

    x = np.arange(len(gapDataFrame))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_title("Train vs Test F1 – Overfitting Analysis")

    ax.bar(x - w/2, gapDataFrame["F1_train"], w, label="Train F1", color="#4C72B0")
    ax.bar(x + w/2, gapDataFrame["F1_test"], w, label="Test F1",  color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(gapDataFrame["model"], rotation=20, ha="right")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weighted F1-score")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fig.savefig("graphs/overfitting-analysis.png", dpi=150)

    plt.close(fig)

    print(f"\n{PREFIX} Saved graphs/overfitting-analysis.png")

    return gapDataFrame


def main():
    dataFrame = loadResults()
    dataFrame = dataFrame.sort_values("f1", ascending=False).reset_index(drop=True)

    graphComparison(dataFrame)

    dataFrame.to_csv("results/final-comparison.csv", index=False)

    print(f"\n{PREFIX} Final Results\n")
    print(dataFrame[["model", "accuracy", "precision", "recall", "f1"]].to_string(index=False))

    overfittingAnalysis()

if __name__ == "__main__":
    main()
