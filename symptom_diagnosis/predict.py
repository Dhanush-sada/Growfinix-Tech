"""
Predict the most likely diseases given a list of symptoms.

Usage as a script (interactive):
    python3 predict.py

Usage as a library:
    from predict import Diagnoser
    d = Diagnoser()
    d.predict(["fever", "dry_cough", "fatigue", "loss_of_taste_smell"])
"""

import json
import joblib

MODEL_DIR = "model"


class Diagnoser:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(f"{model_dir}/model.joblib")
        with open(f"{model_dir}/symptom_columns.json") as f:
            self.symptom_cols = json.load(f)
        with open(f"{model_dir}/label_classes.json") as f:
            self.label_classes = json.load(f)

    def known_symptoms(self):
        return self.symptom_cols

    def predict(self, symptoms, top_k=3):
        """
        symptoms: list of symptom strings (must match known_symptoms(); unknown
                   ones are ignored with a warning).
        Returns: list of (disease, probability) tuples, sorted descending, length top_k.
        """
        symptom_set = set(symptoms)
        unknown = symptom_set - set(self.symptom_cols)
        if unknown:
            print(f"Warning: ignoring unrecognized symptoms: {sorted(unknown)}")

        import pandas as pd
        row = [1 if col in symptom_set else 0 for col in self.symptom_cols]
        row_df = pd.DataFrame([row], columns=self.symptom_cols)
        proba = self.model.predict_proba(row_df)[0]

        ranked = sorted(zip(self.model.classes_, proba), key=lambda x: -x[1])
        return ranked[:top_k]


def _interactive():
    d = Diagnoser()
    print("Known symptoms (pick from this list, comma-separated):\n")
    print(", ".join(d.known_symptoms()))
    print()
    raw = input("Enter your symptoms, comma-separated: ")
    symptoms = [s.strip() for s in raw.split(",") if s.strip()]

    results = d.predict(symptoms, top_k=3)

    print("\n--- Top 3 possible conditions ---")
    for disease, prob in results:
        print(f"  {disease:<30s} {prob*100:5.1f}% confidence")

    print(
        "\nThis is a decision-support demo only, not a medical diagnosis. "
        "Please consult a healthcare professional for any real symptoms."
    )


if __name__ == "__main__":
    _interactive()
