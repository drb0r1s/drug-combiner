# Chat

import re
import joblib
import numpy as np
from functions.cleanSpecialCharacters import cleanSpecialCharacters

PREFIX="[DC: Chat]"
CHAT_PREFIX="ChatDC:"

print(f"{PREFIX} Loading models...")

try:
    tfidf = joblib.load("models/tfidf-vectorizer.pkl")
    svd = joblib.load("models/svd-reducer.pkl")
    logisticRegression = joblib.load("models/logistic-regression.pkl")
    labelEncoder = joblib.load("models/label-encoder.pkl")

    print(f"{PREFIX} All models are ready!\n")

except FileNotFoundError:
    print(f"{PREFIX} ERROR: Some of model files are not ready!\n")
    print(f"{PREFIX} Make sure you prepare the models before running this script (python combiner.py).")

    exit(1)

# Hard-coded info for each label.
LABEL_INFO = {
    "increase_risk": ("⚠️ INCREASED RISK", "Taking these drugs together increases the risk or severity of side effects."),
    "decrease_metabolism": ("🔴 SLOWER METABOLISM", "The first drug slows how your body breaks down the second one. The second drug stays in your system longer, and is stronger."),
    "increase_metabolism": ("🟡 FASTER METABOLISM", "The first drug speeds up how your body breaks down the second one. The second drug becomes less effective."),
    "increase_serum": ("🔴 HIGHER BLOOD LEVELS", "The second drug reaches higher concentrations in your blood, increasing its effects and side effects."),
    "decrease_serum": ("🟡 LOWER BLOOD LEVELS", "The second drug reaches lower concentrations in your blood, reducing its effectiveness."),
    "increase_effect": ("⚠️ AMPLIFIED EFFECT", "The first drug amplifies the pharmacological activity of the second one."),
    "decrease_effect": ("🟡 REDUCED EFFECT", "The first drug reduces the pharmacological activity of the second one."),
    "efficacy_decrease": ("🟡 REDUCED EFFECTIVENESS", "The therapeutic effectiveness of one drug is reduced by the other."),
    "efficacy_increase": ("✅ INCREASED EFFECTIVENESS","The therapeutic effectiveness of one drug is increased by the other."),
    "bioavailability_change": ("🟠 ABSORPTION CHANGE", "How much of the drug is absorbed into your body is affected."),
}

def predict(text: str):
    cleaned = cleanSpecialCharacters(text)

    probability = logisticRegression.predict_proba(tfidf.transform([cleaned]))[0]
    top3Indicies = np.argsort(probability)[::-1][:3]

    top_label = labelEncoder.classes_[top3Indicies[0]]
    info = LABEL_INFO.get(top_label, ("?", "Unknown label."))

    print(f"\n{CHAT_PREFIX} Result: {info[0]}")
    print(f"||      {info[1]}")
    print(f"||      {CHAT_PREFIX} Top predictions:")

    for i in top3Indicies:
        label = labelEncoder.classes_[i]
        short = LABEL_INFO.get(label, (label, ""))[0]

        print(f"||      {probability[i] * 100:5.1f}% {short}")


def buildDescription(drug1: str, drug2: str) -> str:
    templates = [
        f"{drug1} may increase the activities of {drug2}",
        f"The serum concentration of {drug2} can be increased when combined with {drug1}",
        f"The risk or severity of adverse effects can be increased when {drug1} is combined with {drug2}",
        f"The metabolism of {drug2} can be decreased when combined with {drug1}",
    ]

    bestConfidence  = -1
    bestLabel = None

    for template in templates:
        cleaned = cleanSpecialCharacters(template.replace(drug1, "DRUG").replace(drug2, "DRUG"))

        probability = logisticRegression.predict_proba(tfidf.transform([cleaned]))[0]
        confidence = probability.max()

        if confidence > bestConfidence:
            bestConfidence  = confidence
            bestLabel = labelEncoder.classes_[probability.argmax()]

    return bestLabel, bestConfidence


print("CHAT WITH DRUG COMBINER\n")
print(f"{CHAT_PREFIX} You can enter:")
print("||      1) A description, e.g.:")
print('||      "Ibuprofen may increase the bleeding activities of Warfarin"')
print("||      2) Two drug names separated by  +  e.g.:")
print('||      "Ibuprofen + Warfarin"')
print(f"{CHAT_PREFIX} Type quit to exit.\n")

while True:
    try:
        userInput = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{CHAT_PREFIX} Exiting the chat...")
        break

    if not userInput:
        continue

    if userInput.lower() in ("quit", "exit", "q"):
        print(f"\n{CHAT_PREFIX} Exiting the chat...")
        break

    if "+" in userInput:
        parts = userInput.split("+", 1)
        drug1, drug2 = parts[0].strip(), parts[1].strip()

        if not drug1 or not drug2:
            print(f"\n{CHAT_PREFIX} Please enter two drug names separated by +.")
            continue

        label, confidence = buildDescription(drug1, drug2)
        info = LABEL_INFO.get(label, (label, ""))

        print(f"{CHAT_PREFIX} Most likely interaction type for {drug1} + {drug2}:")
        print(f"{info[0]}  ({confidence * 100:.0f}% confidence)")
        print(f"{info[1]}")
    else:
        predict(userInput)
