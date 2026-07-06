import re
import string

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("spam_dataset.csv")
print("Dataset shape:", df.shape)
print(df["label"].value_counts(), "\n")

STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for
with at by from up about into over after before between out against during
this that these those it its i you he she we they them his her our your
their as not no nor so than too very can will just don should now
""".split())


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)


df["clean_message"] = df["message"].apply(clean_text)

print("Example before/after cleaning:")
print("Raw:  ", df["message"].iloc[0])
print("Clean:", df["clean_message"].iloc[0], "\n")

X_text = df["clean_message"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n")

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

for name, model in [("Naive Bayes", nb_model), ("Logistic Regression", lr_model)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"== {name} ==")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("-" * 50)

final_model = nb_model
y_pred_final = final_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["ham", "spam"],
    yticklabels=["ham", "spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes Spam Detector")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

print("Saved confusion matrix to confusion_matrix.png\n")

new_messages = [
    "Congratulations! You've won a free cruise, click now to claim your prize!!!",
    "Hey, can you review the attached document before our meeting tomorrow?",
    "URGENT: verify your bank account now or it will be suspended today.",
    "Let's grab lunch this Friday, I miss catching up with you.",
    "Earn $5000 a week from home, no experience needed, sign up now!",
]

new_clean = [clean_text(m) for m in new_messages]
new_vec = vectorizer.transform(new_clean)

predictions = final_model.predict(new_vec)
probabilities = final_model.predict_proba(new_vec)

print("Predictions on new messages:")
print("=" * 60)

for msg, pred, prob in zip(new_messages, predictions, probabilities):
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred]
    print(f"[{label:4s}] ({confidence:.2%} confidence) {msg}")