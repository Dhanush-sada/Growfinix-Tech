"""Additional evaluation used in Chapter 4 (Section 4.3) of the internship report.
Run from the symptom_diagnosis folder:  python evaluation_extra.py
Uses the same dataset, 80/20 stratified split and random seed (42) as train.py."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, top_k_accuracy_score

df = pd.read_csv("data/disease_symptom_dataset.csv")
cols = [c for c in df.columns if c != "disease"]
X, y = df[cols], df["disease"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)

print("== 4.3.1 Classifier comparison (top-1 / top-3 accuracy) ==")
models = {"Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest (300 trees)": rf,
          "Logistic Regression": LogisticRegression(max_iter=2000), "Bernoulli Naive Bayes": BernoulliNB(),
          "k-Nearest Neighbours (k=5)": KNeighborsClassifier(n_neighbors=5)}
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(f"{name:28s} {accuracy_score(yte, m.predict(Xte)):.4f}  {top_k_accuracy_score(yte, m.predict_proba(Xte), k=3, labels=m.classes_):.4f}")

print("\n== 4.3.2 Five-fold stratified cross-validation ==")
cv = cross_val_score(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
                     X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(np.round(cv, 4), f"mean={cv.mean():.4f} std={cv.std():.4f}")

print("\n== 4.3.3 Effect of number of trees ==")
for n in [1, 5, 10, 25, 50, 100, 300]:
    m = RandomForestClassifier(n_estimators=n, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print(f"{n:4d} trees: {accuracy_score(yte, m.predict(Xte)):.4f}")

print("\n== 4.3.4 Only k randomly chosen symptoms provided (mean of 30 trials) ==")
rng = np.random.RandomState(0); A = Xte.values
for k in range(1, 6):
    t1, t3 = [], []
    for _ in range(30):
        Z = np.zeros_like(A)
        for i, row in enumerate(A):
            idx = np.flatnonzero(row); Z[i, rng.choice(idx, size=min(k, len(idx)), replace=False)] = 1
        P = rf.predict_proba(pd.DataFrame(Z, columns=cols))
        t1.append(accuracy_score(yte, rf.classes_[P.argmax(1)])); t3.append(top_k_accuracy_score(yte, P, k=3, labels=rf.classes_))
    print(f"k={k}: top-1 {np.mean(t1):.4f}  top-3 {np.mean(t3):.4f}")
