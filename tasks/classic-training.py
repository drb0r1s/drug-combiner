# Task 3: Classic Training
# Training three classical text-classification models on the TF-IDF features:
# 1. Multinomial Naive Bayes (NB)
# 2. Logistic Regression (LR)
# 3. Support Vector Machine (SVM)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

PREFIX="[DC: Classic Training]"


def loadData():
    XTrain = joblib.load("data/x-train-tfidf.pkl")
    XTest = joblib.load("data/x-test-tfidf.pkl")

    YTrain = np.load("data/y-train.npy")
    YTest  = np.load("data/y-test.npy")

    labelEncoder = joblib.load("models/label-encoder.pkl")

    return XTrain, XTest, YTrain, YTest, labelEncoder


def evaluate(name, model, XTest, YTest, labelEncoder):
    YPred = model.predict(XTest)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(YTest, YPred),
        "precision": precision_score(YTest, YPred, average="weighted", zero_division=0),
        "recall": recall_score(YTest, YPred, average="weighted", zero_division=0),
        "f1": f1_score(YTest, YPred, average="weighted", zero_division=0),
    }

    print(f"{PREFIX} {name}\n")
    print(classification_report(YTest, YPred, target_names=labelEncoder.classes_, zero_division=0))

    confusionMatrix = confusion_matrix(YTest, YPred)
    fig, axes = plt.subplots(figsize=(10, 8))

    sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Blues", xticklabels=labelEncoder.classes_, yticklabels=labelEncoder.classes_, ax=axes)

    axes.set_title(f"Confusion Matrix – {name}", fontsize=13, pad=12)

    axes.set_xlabel("Predicted", fontsize=11)
    axes.set_ylabel("Actual", fontsize=11)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    imageName = name.lower().replace(" ", "_")
    fig.savefig(f"graphs/confusion-matrix-{imageName}.png", dpi=150)
    
    plt.close(fig)

    return metrics


def main():
    XTrain, XTest, YTrain, YTest, labelEncoder = loadData()

    models = [
        (
            "Complement Naive Bayes",
            ComplementNB(alpha=0.1)
        ),

        (
            "Logistic Regression",
            LogisticRegression(C=5.0, max_iter=2000, solver="saga", random_state=42)
        ),

        (
            "SVM (LinearSVC)",
            LinearSVC(C=1.0, max_iter=2000, random_state=42)
        )
    ]

    allMetrics = []

    for name, classifier in models:
        print(f"{PREFIX} Training {name}...\n")

        classifier.fit(XTrain, YTrain)

        joblib.dump(classifier, f"models/{name.lower().replace(' ', '-').replace('(','').replace(')','').strip()}.pkl")
        
        metrics = evaluate(name, classifier, XTest, YTest, labelEncoder)
        allMetrics.append(metrics)

    dataFrameResults = pd.DataFrame(allMetrics)
    dataFrameResults.to_csv("results/classic-results.csv", index=False)

    print(f"\n{PREFIX} Results of classic training are saved to results/classic-results.csv\n")
    print(dataFrameResults.to_string(index=False))


if __name__ == "__main__":
    main()
