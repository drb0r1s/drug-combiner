import re

# Removing every character except for letters and spaces, so that model does not treat "increased" and "increased." as different values.
def cleanSpecialCharacters(text: str) -> str:
    text = text.lower()

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text