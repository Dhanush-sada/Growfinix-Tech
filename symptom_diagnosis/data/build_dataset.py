"""
Generates a synthetic symptom -> disease dataset for training/demo purposes.

Each disease has:
  - a set of "core" symptoms (usually present)
  - a set of "occasional" symptoms (sometimes present)
Random noise symptoms (unrelated) are occasionally added, and core symptoms
are occasionally dropped, to simulate real-world variability and make the
classification problem non-trivial.

Replace this with a real dataset (e.g. a Kaggle disease-symptom dataset or
DDXPlus) for anything beyond learning/prototyping.
"""

import csv
import random

random.seed(42)

DISEASES = {
    "Common Cold": {
        "core": ["runny_nose", "sneezing", "sore_throat", "cough"],
        "occasional": ["mild_fever", "fatigue", "headache"],
    },
    "Influenza": {
        "core": ["high_fever", "body_ache", "fatigue", "chills"],
        "occasional": ["cough", "headache", "sore_throat"],
    },
    "COVID-19": {
        "core": ["fever", "dry_cough", "fatigue", "loss_of_taste_smell"],
        "occasional": ["shortness_of_breath", "sore_throat", "headache", "body_ache"],
    },
    "Migraine": {
        "core": ["severe_headache", "nausea", "light_sensitivity"],
        "occasional": ["vomiting", "visual_disturbance", "dizziness"],
    },
    "Tension Headache": {
        "core": ["mild_headache", "neck_stiffness"],
        "occasional": ["fatigue", "irritability"],
    },
    "Gastroenteritis": {
        "core": ["diarrhea", "vomiting", "abdominal_pain", "nausea"],
        "occasional": ["mild_fever", "fatigue", "loss_of_appetite"],
    },
    "Food Poisoning": {
        "core": ["nausea", "vomiting", "diarrhea", "abdominal_cramps"],
        "occasional": ["fever", "weakness"],
    },
    "Type 2 Diabetes": {
        "core": ["excessive_thirst", "frequent_urination", "fatigue"],
        "occasional": ["blurred_vision", "weight_loss", "slow_healing_wounds"],
    },
    "Hypothyroidism": {
        "core": ["fatigue", "weight_gain", "cold_intolerance"],
        "occasional": ["dry_skin", "hair_loss", "constipation"],
    },
    "Hyperthyroidism": {
        "core": ["weight_loss", "rapid_heartbeat", "heat_intolerance"],
        "occasional": ["anxiety", "tremor", "sweating"],
    },
    "Asthma": {
        "core": ["shortness_of_breath", "wheezing", "chest_tightness"],
        "occasional": ["cough", "fatigue"],
    },
    "Pneumonia": {
        "core": ["high_fever", "cough_with_phlegm", "chest_pain", "shortness_of_breath"],
        "occasional": ["fatigue", "chills", "rapid_breathing"],
    },
    "Urinary Tract Infection": {
        "core": ["burning_urination", "frequent_urination", "pelvic_pain"],
        "occasional": ["cloudy_urine", "mild_fever", "fatigue"],
    },
    "Allergic Rhinitis": {
        "core": ["sneezing", "itchy_eyes", "runny_nose"],
        "occasional": ["nasal_congestion", "fatigue"],
    },
    "Iron Deficiency Anemia": {
        "core": ["fatigue", "pale_skin", "weakness"],
        "occasional": ["dizziness", "shortness_of_breath", "cold_hands_feet"],
    },
    "Gastroesophageal Reflux (GERD)": {
        "core": ["heartburn", "regurgitation", "chest_pain"],
        "occasional": ["sore_throat", "cough", "nausea"],
    },
    "Chickenpox": {
        "core": ["itchy_rash", "fever", "fatigue"],
        "occasional": ["headache", "loss_of_appetite"],
    },
    "Conjunctivitis": {
        "core": ["red_eyes", "itchy_eyes", "watery_eyes"],
        "occasional": ["eye_discharge", "light_sensitivity"],
    },
    "Sinusitis": {
        "core": ["facial_pain", "nasal_congestion", "headache"],
        "occasional": ["fever", "cough", "loss_of_smell"],
    },
    "Anxiety Disorder": {
        "core": ["excessive_worry", "restlessness", "rapid_heartbeat"],
        "occasional": ["fatigue", "difficulty_concentrating", "sweating"],
    },
}

ALL_SYMPTOMS = sorted({s for d in DISEASES.values() for s in d["core"] + d["occasional"]})

NOISE_SYMPTOMS = ["fatigue", "headache", "mild_fever", "dizziness"]  # symptoms that overlap across many diseases


def generate_case(disease):
    info = DISEASES[disease]
    symptoms = set()

    # Core symptoms: usually present (85% chance each), so cases aren't identical
    for s in info["core"]:
        if random.random() < 0.85:
            symptoms.add(s)

    # Occasional symptoms: sometimes present (35% chance each)
    for s in info["occasional"]:
        if random.random() < 0.35:
            symptoms.add(s)

    # Random unrelated noise symptom (10% chance) - simulates comorbidity / reporting noise
    if random.random() < 0.10:
        symptoms.add(random.choice(NOISE_SYMPTOMS))

    # Safety net: never generate an empty case
    if not symptoms:
        symptoms.add(random.choice(info["core"]))

    return symptoms


def build_dataset(samples_per_disease=60):
    rows = []
    for disease in DISEASES:
        for _ in range(samples_per_disease):
            symptoms = generate_case(disease)
            row = {s: (1 if s in symptoms else 0) for s in ALL_SYMPTOMS}
            row["disease"] = disease
            rows.append(row)
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_dataset(samples_per_disease=60)
    fieldnames = ALL_SYMPTOMS + ["disease"]

    out_path = "/home/claude/symptom_diagnosis/data/disease_symptom_dataset.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows, {len(ALL_SYMPTOMS)} symptom columns, "
          f"{len(DISEASES)} diseases -> {out_path}")
