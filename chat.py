# Chat

import re
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

PREFIX="[DC: Chat]"
CHAT_PREFIX="ChatDC:"
LEFT_PADDING="      "
BOTTOM_PADDING=f"||{90*"_"}"
MEDICAL_KEYWORDS = ["interaction", "increase", "increased", "decrease", "decreased", "metabolism", "serum", "concentration", "risk", "efficacy", "bioavailability", "activities", "adverse", "combine", "combined", "therapeutic"]

TEST_CASES = [
    "Aspirin may increase the anticoagulant activities of Warfarin",
    "The serum concentration of Digoxin can be increased when combined with Amiodarone",
    "The metabolism of Methadone can be decreased when combined with Fluoxetine",
    "The risk or severity of adverse effects can be increased when Oxycodone is combined with Diazepam",
    "The therapeutic efficacy of Metformin can be decreased when used in combination with Prednisone",
    "The bioavailability of Clobetasol can be decreased when combined with Magnesium hydroxide",
    "Fluoxetine may increase the serotonergic activities of Tramadol",
    "The metabolism of Warfarin can be decreased when combined with Amiodarone",
]

print(f"{PREFIX} Loading models...")

try:
    tfidf = joblib.load("models/tfidf-vectorizer.pkl")
    svd = joblib.load("models/svd-reducer.pkl")
    logisticRegression = joblib.load("models/logistic-regression.pkl")
    labelEncoder = joblib.load("models/label-encoder.pkl")
    drugClassEncoder = joblib.load("models/drug-class-encoder.pkl")

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

# Removing every character except for letters and spaces, so that model does not treat "increased" and "increased." as different values.
def cleanSpecialCharacters(text: str) -> str:
    text = text.lower()

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# It is important to block the model from responding if input is not related to pharmacology.
def isInputValid(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in MEDICAL_KEYWORDS)

def predict(text: str, drug1Class: str = "Unknown", drug2Class: str = "Unknown"):
    if not isInputValid(text):
        print(f"\n{CHAT_PREFIX} Please enter a valid drug interaction description.\n")
        return
    
    if len(text.split()) < 5:
        print(f"\n{CHAT_PREFIX} Input too short. Please enter a full description.\n")
        return

    cleaned = cleanSpecialCharacters(text)

    tfidfVector = tfidf.transform([cleaned])
    
    # Encode drug classes and append to TF-IDF vector.
    d1 = drugClassEncoder.transform([drug1Class])[0] if drug1Class in drugClassEncoder.classes_ else 0
    d2 = drugClassEncoder.transform([drug2Class])[0] if drug2Class in drugClassEncoder.classes_ else 0

    classVector = csr_matrix([[d1, d2]])
    
    combined = hstack([tfidfVector, classVector])

    probability = logisticRegression.predict_proba(combined)[0]
    top3Indicies = np.argsort(probability)[::-1][:3]

    top_label = labelEncoder.classes_[top3Indicies[0]]
    info = LABEL_INFO.get(top_label, ("?", "Unknown label."))

    print(f"\n{CHAT_PREFIX} {info[0]}")
    print(f"||{LEFT_PADDING}{info[1]}")
    print("||")
    print(f"||{LEFT_PADDING}Top predictions:")

    for i in top3Indicies:
        label = labelEncoder.classes_[i]
        short = LABEL_INFO.get(label, (label, ""))[0]

        print(f"||{LEFT_PADDING}{probability[i] * 100:5.1f}% {short}")

    print(f"{BOTTOM_PADDING}\n")


def buildDescription(drug1: str, drug2: str) -> str:
    templates = [
        f"{drug1} may increase the activities of {drug2}",
        f"The serum concentration of {drug2} can be increased when combined with {drug1}",
        f"The risk or severity of adverse effects can be increased when {drug1} is combined with {drug2}",
        f"The metabolism of {drug2} can be decreased when combined with {drug1}",
    ]

    bestConfidence  = -1
    bestLabel = None

    d1 = drugClassEncoder.transform([drug1])[0] if drug1 in drugClassEncoder.classes_ else 0
    d2 = drugClassEncoder.transform([drug2])[0] if drug2 in drugClassEncoder.classes_ else 0

    classVector = csr_matrix([[d1, d2]])

    for template in templates:
        cleaned = cleanSpecialCharacters(template.replace(drug1, "DRUG").replace(drug2, "DRUG"))

        tfidfVector = tfidf.transform([cleaned])
        combined = hstack([tfidfVector, classVector])
        
        probability = logisticRegression.predict_proba(combined)[0]
        confidence = probability.max()

        if confidence > bestConfidence:
            bestConfidence  = confidence
            bestLabel = labelEncoder.classes_[probability.argmax()]

    return bestLabel, bestConfidence


print("CHAT WITH DRUG COMBINER - by Boris Marinkovic\n")

print(f"{CHAT_PREFIX} You can enter:")

print(f"||{LEFT_PADDING}1) A description, e.g.:")
print(f'||{LEFT_PADDING}"Ibuprofen may increase the bleeding activities of Warfarin"')
print("||")

print(f"||{LEFT_PADDING}2) Two drug names separated by + e.g.:")
print(f'||{LEFT_PADDING}"Ibuprofen + Warfarin"')
print("||")

print(f'||{LEFT_PADDING}3) Type "test" to run predefined test cases.')
print("||")

print(f"||{LEFT_PADDING}Type quit to exit.")
print(f"{BOTTOM_PADDING}\n")

while True:
    try:
        userInput = input("(Your message...) >>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{CHAT_PREFIX} Exiting the chat...")
        break

    if not userInput:
        continue

    if userInput.lower() in ("quit", "exit", "q"):
        print(f"\n{CHAT_PREFIX} Exiting the chat...")
        break

    if userInput.lower() in ("test", "t"):
        print(f"\n{CHAT_PREFIX} Running predefined test cases...\n")

        for i, testCase in enumerate(TEST_CASES, 1):
            print(f"{CHAT_PREFIX} [TEST {i}] {testCase}")
            predict(testCase)
            
            continue
        
        continue

    if "+" in userInput:
        parts = userInput.split("+", 1)
        drug1, drug2 = parts[0].strip(), parts[1].strip()

        if(len(drug1) < 3 or len(drug2) < 3):
            print(f"\n{CHAT_PREFIX} Please enter valid drug names.\n")
            continue

        label, confidence = buildDescription(drug1, drug2)
        info = LABEL_INFO.get(label, (label, ""))

        print(f"\n{CHAT_PREFIX} Most likely interaction type for {drug1} + {drug2}:")
        print(f"||{LEFT_PADDING}{info[0]}  ({confidence * 100:.0f}% confidence)")
        print(f"||{LEFT_PADDING}{info[1]}")
        print(f"{BOTTOM_PADDING}\n")
    else:
        predict(userInput)