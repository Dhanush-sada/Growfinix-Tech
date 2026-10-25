# Symptom-to-Disease Checker (Demo Project)

A simple ML pipeline that predicts the most likely disease(s) given a set of
symptoms. Built for learning/portfolio purposes — **not a real diagnostic
tool.**

## How it works

1. `data/build_dataset.py` — generates a synthetic but realistic
   symptom/disease dataset (20 diseases, ~66 symptoms, 1200 samples). Swap
   this out for a real dataset (e.g. a Kaggle disease-symptom dataset or
   [DDXPlus](https://github.com/mila-iqia/ddxplus)) once you're ready to move
   past prototyping.
2. `train.py` — trains a Random Forest classifier on symptom presence/absence,
   evaluates it (accuracy, per-class precision/recall, top-3 accuracy), and
   saves the model + metadata to `model/`.
3. `predict.py` — loads the trained model and returns ranked disease
   predictions with confidence scores for a given symptom list. Also runnable
   as a CLI.
4. `app.py` — a Streamlit web UI: pick symptoms from a searchable list, get
   ranked predictions with confidence bars.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate the dataset
python3 data/build_dataset.py

# 2. Train the model
python3 train.py

# 3a. Try it from the command line
python3 predict.py

# 3b. Or launch the web app
streamlit run app.py
```

## Using it as a library

```python
from predict import Diagnoser

d = Diagnoser()
results = d.predict(["fever", "dry_cough", "fatigue", "loss_of_taste_smell"], top_k=3)
# [("COVID-19", 0.99), ("Chickenpox", 0.005), ("Food Poisoning", 0.001)]
```

## Next steps to make this more realistic

- **Real data**: replace the synthetic dataset with a real clinical or
  research dataset (DDXPlus is a good middle ground — large and realistic
  without needing hospital access).
- **Better model**: try gradient boosting (`HistGradientBoostingClassifier`
  in scikit-learn, or XGBoost/LightGBM if installed) — usually a few points
  better than Random Forest on tabular data like this.
- **Free-text symptom entry**: add an NLP step (a small classifier or an LLM
  call) that extracts structured symptoms from a sentence like "I've had a
  headache and fever for two days," then feeds those into the existing model.
- **Explainability**: use SHAP to show *why* a prediction was made for a
  specific case, not just global feature importance.
- **Demographics/history**: add age, sex, duration, and severity as features
  — real differential diagnosis depends heavily on these, and a symptom-only
  model will plateau without them.
- **Calibration**: check that predicted probabilities are well-calibrated
  (e.g. with `CalibratedClassifierCV`) before showing confidence percentages
  to real users.

## Important disclaimer

This project is for **learning and prototyping only**. Real diagnosis
requires patient history, physical examination, and lab/imaging results that
a symptom checklist cannot capture. Do not present this as, or use this in
place of, professional medical advice.
