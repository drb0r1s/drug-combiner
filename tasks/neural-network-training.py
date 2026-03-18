# Task 4: Neural Network Training
# Input: 200-dimensional SVD-reduced TF-IDF vectors.
# Architecture: Dense(512, relu) -> Dropout(0.3) -> Dense(256, relu) -> Dropout(0.3) -> Dense(nClasses, softmax)
# Results are saved to results/neural-network-results.csv and a confusion matrix graph is produced.

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

PREFIX="[DC: Neural Network Training]"


def main():
    XTrain = joblib.load("data/x-train-svd.pkl")
    XTest = joblib.load("data/x-test-svd.pkl")

    YTrain = np.load("data/y-train.npy")
    YTest = np.load("data/y-test.npy")

    labelEncoder = joblib.load("models/label-encoder.pkl")

    print(f"{PREFIX} Training Feed-Forward Neural Network (MLP)...\n")

    print(f"{PREFIX} Input columns: {XTrain.shape[1]}\n")
    print(f"{PREFIX} Classes: {len(labelEncoder.classes_)}\n")

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )

    mlp.fit(XTrain, YTrain)

    joblib.dump(mlp, "models/neural-network.pkl")

    YPred = mlp.predict(XTest)

    metrics = {
        "model": "Neural Network (MLP)",
        "accuracy": accuracy_score(YTest, YPred),
        "precision": precision_score(YTest, YPred, average="weighted", zero_division=0),
        "recall": recall_score(YTest, YPred, average="weighted", zero_division=0),
        "f1": f1_score(YTest, YPred, average="weighted", zero_division=0),
    }

    print(f"{PREFIX} Neural Network (MLP)\n")
    print(classification_report(YTest, YPred, target_names=labelEncoder.classes_, zero_division=0))

    confusionMatrix = confusion_matrix(YTest, YPred)
    fig, axes = plt.subplots(figsize=(10, 8))

    sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Purples", xticklabels=labelEncoder.classes_, yticklabels=labelEncoder.classes_, ax=axes)
    
    axes.set_title("Confusion Matrix – Neural Network (MLP)", fontsize=13, pad=12)

    axes.set_xlabel("Predicted", fontsize=11)
    axes.set_ylabel("Actual", fontsize=11)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig("graphs/confusion-matrix-neural_network.png", dpi=150)

    plt.close(fig)

    fig2, axes2 = plt.subplots(figsize=(8, 5))

    axes2.plot(mlp.loss_curve_, label="Training loss")

    if mlp.best_loss_ is not None:
        axes2.axhline(y=mlp.best_loss_, color="r", linestyle="--", alpha=0.6, label="Best validation loss")

    axes2.set_title("Neural Network – Training Loss Curve")

    axes2.set_xlabel("Iteration")
    axes2.set_ylabel("Loss")

    axes2.legend()

    axes2.grid(True, alpha=0.3)

    fig2.tight_layout()

    fig2.savefig("graphs/neural-network-loss-curve.png", dpi=150)

    plt.close(fig2)

    pd.DataFrame([metrics]).to_csv("results/neural-network-results.csv", index=False)

    print(f"\n{PREFIX} Results of neural network training are saved to results/neural-network-results.csv\n")

    print(f"{PREFIX} Accuracy: {metrics['accuracy']:.4f}")
    print(f"{PREFIX} F1-score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()