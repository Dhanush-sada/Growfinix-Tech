"""
Trains a symptom -> disease classifier.

Usage:
    python3 train.py

Saves:
    model/model.joblib          - trained classifier
    model/symptom_columns.json  - ordered list of symptom feature names
    model/label_classes.json    - ordered list of disease class names
"""

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, top_k_accuracy_score

DATA_PATH = "data/disease_symptom_dataset.csv"
MODEL_DIR = "model"


def main():
    df = pd.read_csv(DATA_PATH)

    symptom_cols = [c for c in df.columns if c != "disease"]
    X = df[symptom_cols]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    print("=== Top-1 classification report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    y_proba = clf.predict_proba(X_test)
    top3_acc = top_k_accuracy_score(y_test, y_proba, k=3, labels=clf.classes_)
    print(f"Top-3 accuracy: {top3_acc:.3f}")

    # --- Feature importance (which symptoms matter most overall) ---
    importances = sorted(
        zip(symptom_cols, clf.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop 10 most informative symptoms:")
    for sym, imp in importances[:10]:
        print(f"  {sym:<25s} {imp:.4f}")

    # --- Save artifacts ---
    joblib.dump(clf, f"{MODEL_DIR}/model.joblib")
    with open(f"{MODEL_DIR}/symptom_columns.json", "w") as f:
        json.dump(symptom_cols, f, indent=2)
    with open(f"{MODEL_DIR}/label_classes.json", "w") as f:
        json.dump(list(clf.classes_), f, indent=2)

    print(f"\nSaved model + metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
