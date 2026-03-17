# Task 1: Data Preparing
# Reads the CSV and assigns an interaction type label to each row based on regular-expression matching of the descriptions.

# ID CLASS DESCRIPTION
# 0 increase_effect ""

# Rows where no pattern matches are dropped.

import re
import pandas as pd

PREFIX="[DC: Data Preparing]"

DATA = "data/drug-interactions.csv"
OUTPUT = "data/drug-interactions-labeled.csv"

LABELS = [
    ("increase_effect", r"may increase the \w[\w\s\-]*activities"),
    ("decrease_effect", r"may decrease the \w[\w\s\-]*activities"),
    ("increase_metabolism", r"metabolism.{0,40}increased"),
    ("decrease_metabolism", r"metabolism.{0,40}decreased"),
    ("increase_serum", r"serum concentration.{0,40}increased"),
    ("decrease_serum", r"serum concentration.{0,40}decreased"),
    ("efficacy_increase", r"therapeutic efficacy.{0,40}increased"),
    ("efficacy_decrease", r"therapeutic efficacy.{0,40}decreased"),
    ("increase_risk", r"risk or severity.{0,40}increased"),
    ("bioavailability_change", r"bioavailability.{0,40}(increased|decreased)")
]

def assignLabel(text: str) -> str | None:
    adjustedText = str(text).lower()
    
    for label, pattern in LABELS:
        if re.search(pattern, adjustedText):
            return label
        
    return None


def main():
    drugs = pd.read_csv(DATA)
    
    drugs.columns = ["drug1", "drug2", "description"] # Overriding default column names to a more appropriate ones.
    drugs = drugs.dropna(subset=["description"]) # Removing empty descriptions.

    drugs["label"] = drugs["description"].apply(assignLabel)

    oldLength = len(drugs)

    drugs = drugs[drugs["label"].notna()].reset_index(drop=True) # Adjusting the table, so that only rows with assigned label are kept.
    
    newLength = len(drugs)

    print(f"{PREFIX} Rows before filtering: {oldLength:,}")
    print(f"{PREFIX} Rows after filtering: {newLength:,}")
    print(f"{PREFIX} {oldLength - newLength:,} rows were removed due to invalid pattern matching.")

    print(f"\n{PREFIX} Class distribution:")
    
    counts = drugs["label"].value_counts()
    
    for label, count in counts.items():
        print(f"{label:25s}: {count:>7,} ({count / newLength * 100:.1f}%)")

    drugs.to_csv(OUTPUT, index=False)
    print(f"\n{PREFIX} Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
