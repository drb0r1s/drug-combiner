# Task 2: Feature Engineering
# Converts descriptions into numerial feature vectors and saves train/test splits for model training.

# Two feature representations:
# 1. TF-IDF unigrams + bigrams (max 30000 features), used by NB, LR, SVM.
# 2. Same TF-IDF but with reduced dimensionality via SVD (200 components), used by the neural network.

# The drug names are masked in the text before vectorisation so that models learn interaction semantics rather than drug identity.

import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from functions.cleanSpecialCharacters import cleanSpecialCharacters

PREFIX = "[DC: Feature Engineering]"

LABELED_DATA = "data/drug-interactions-labeled.csv"
SEED = 999 # Random seed.

# Masking the actual drug name with the word "DRUG".
def maskDrugNames(row: pd.Series) -> str:
    description = row["description"]

    for drug in [row["drug1"], row["drug2"]]:
        description = re.sub(re.escape(str(drug)), "DRUG", description, flags=re.IGNORECASE)
    
    return description


def main():
    labeledData = pd.read_csv(LABELED_DATA)

    # 1. Masking and cleaning.
    labeledData["clean-description"] = labeledData.apply(maskDrugNames, axis=1).apply(cleanSpecialCharacters)

    # 2. Encoding labels.
    labelEncoder = LabelEncoder()

    labeledData["label-id"] = labelEncoder.fit_transform(labeledData["label"])

    print(f"{PREFIX} Encoded labels:", list(labelEncoder.classes_)) # Printing all encoded labels can be useful in case of debugging.
    
    joblib.dump(labelEncoder, "models/label-encoder.pkl") # Saving encoded labels to a file, so that we can easily decode them later.

    # 3. Creating a train/test split.
    XTrain, XTest, YTrain, YTest = train_test_split(
        labeledData["clean-description"], labeledData["label-id"],
        test_size=0.20, random_state=SEED, stratify=labeledData["label-id"]
    )

    print(f"\n{PREFIX} Rows for training: {len(XTrain):,}")
    print(f"{PREFIX} Rows for testing: {len(XTest):,}\n")

    # 4. Vectorizing TF-IDF
    tfIdf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,
        min_df=3,
    )

    XTrainTfidf = tfIdf.fit_transform(XTrain)
    XTestTfidf  = tfIdf.transform(XTest)

    joblib.dump(tfIdf, "models/tfidf-vectorizer.pkl")
    print(f"{PREFIX} Shape of TF-IDF: {XTrainTfidf.shape}\n")

    # 5. SVD reduction for the neural network.
    svd = TruncatedSVD(n_components=200, random_state=SEED)

    XTrainSvd = svd.fit_transform(XTrainTfidf)
    XTestSvd  = svd.transform(XTestTfidf)

    explained = svd.explained_variance_ratio_.sum()

    print(f"{PREFIX} 200 components of SVD explain {explained * 100:.1f}% variance.\n")
    joblib.dump(svd, "models/svd-reducer.pkl")

    # 6. Saving computed matrices.
    joblib.dump(XTrainTfidf, "data/x-train-tfidf.pkl")
    joblib.dump(XTestTfidf,  "data/x-test-tfidf.pkl")

    joblib.dump(XTrainSvd, "data/x-train-svd.pkl")
    joblib.dump(XTestSvd,  "data/x-test-svd.pkl")

    np.save("data/y-train.npy", YTrain.to_numpy())
    np.save("data/y-test.npy",  YTest.to_numpy())

    print(f"{PREFIX} All models and data files are saved.")


if __name__ == "__main__":
    main()