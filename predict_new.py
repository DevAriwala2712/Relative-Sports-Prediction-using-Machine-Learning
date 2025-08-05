import pickle
import pandas as pd
import numpy as np
import random
import re

# Load model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load fighters
fighters_df = pd.read_csv("data/fighters.csv")
fighters_df['name'] = fighters_df['name'].str.strip().str.lower()

# Helper: Clean numeric fields
def clean_numeric(value):
    try:
        if isinstance(value, str):
            value = value.strip().lower()
            if value == 'unknown' or value == '':
                return 0.0
            match = re.search(r'(\d+(\.\d+)?)', value.replace(',', ''))
            if match:
                return float(match.group(1))
        return float(value) if pd.notnull(value) else 0.0
    except:
        return 0.0

# Helper: Get cleaned stats
def get_cleaned_stats(fighter_name):
    fighter = fighters_df[fighters_df['name'] == fighter_name.strip().lower()]
    if fighter.empty:
        return None
    fighter = fighter.iloc[0]
    return {
        'wins': clean_numeric(fighter['wins']),
        'looses': clean_numeric(fighter['looses']),
        'height': clean_numeric(fighter['height']),
        'reach': clean_numeric(fighter['reach']),
        'age': 25  # default if dob missing
    }

# Get two random fighters who never fought each other
all_fighters = list(fighters_df['name'].unique())
attempts = 0
max_attempts = 50

while attempts < max_attempts:
    fighter_a = random.choice(all_fighters)
    fighter_b = random.choice(all_fighters)

    if fighter_a == fighter_b:
        continue

    stats_a = get_cleaned_stats(fighter_a)
    stats_b = get_cleaned_stats(fighter_b)

    if stats_a and stats_b:
        break

    attempts += 1
else:
    print("❌ Couldn't find two valid fighters with complete stats.")
    exit(1)

# Prepare feature vector
features = [
    stats_a['wins'] - stats_b['wins'],
    stats_a['looses'] - stats_b['looses'],
    stats_a['height'] - stats_b['height'],
    stats_a['reach'] - stats_b['reach'],
    stats_a['age'] - stats_b['age']
]
features = np.array(features).reshape(1, -1)

# Predict
prediction = model.predict(features)[0]
prob = model.predict_proba(features)[0]

# Output result
print(f"\n🥊 Simulated Fight: {fighter_a.title()} vs {fighter_b.title()}")
print(f"📊 Prediction: {'Winner = Fighter A' if prediction == 0 else 'Winner = Fighter B'}")
if len(prob) == 2:
    print(f"🧠 Confidence: {prob[0]*100:.2f}% for A | {prob[1]*100:.2f}% for B")
else:
    print(f"🧠 Model predicted only one class confidently: {prob[0]*100:.2f}%")

