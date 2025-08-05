import pickle
import numpy as np
import pandas as pd
import random
import re

# Load model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load datasets
fights_df = pd.read_csv('data/popular_matches.csv')
fighters_df = pd.read_csv('data/fighters.csv')

# Normalize fighter names for matching
fighters_df['name'] = fighters_df['name'].str.strip().str.lower()

# Function to get stats
def get_fighter_stats(name):
    name = name.strip().lower()
    fighter = fighters_df[fighters_df['name'] == name]
    return fighter.iloc[0].to_dict() if not fighter.empty else None

# Function to extract numeric value from strings like "6.00 ft (1.83 m)"
def extract_numeric(s):
    if pd.isna(s):
        return None
    match = re.search(r'(\d+(\.\d+)?)', str(s))
    return float(match.group(1)) if match else None

# Attempt limit to avoid infinite loops
max_attempts = 100
attempts = 0

while attempts < max_attempts:
    attempts += 1
    row = fights_df.sample(1).iloc[0]
    name_a = row['opponent_1']
    name_b = row['opponent_2']
    winner = row['verdict']

    stats_a = get_fighter_stats(name_a)
    stats_b = get_fighter_stats(name_b)

    if stats_a is None or stats_b is None:
        continue

    try:
        features = [
            int(stats_a['wins']) - int(stats_b['wins']),
            int(stats_a['looses']) - int(stats_b['looses']),
            extract_numeric(stats_a['height']) - extract_numeric(stats_b['height']),
            extract_numeric(stats_a['reach']) - extract_numeric(stats_b['reach']),
            int(stats_a['age']) - int(stats_b['age']),
        ]
        if None in features:
            raise ValueError("Missing numeric data")
    except Exception as e:
        print(f"⚠️ Skipping fight due to data issue: {e}")
        continue

    X_sample = np.array(features).reshape(1, -1)
    predicted_label = model.predict(X_sample)[0]
    actual_label = 0 if winner.strip().lower() == name_a.strip().lower() else 1

    print("\n🧪 RANDOM FIGHT PREDICTION")
    print(f"🥊 Fight: {name_a} vs {name_b}")
    print(f"📊 Fighter A stats: {stats_a}")
    print(f"📊 Fighter B stats: {stats_b}")
    print(f"\n🔮 Prediction: Winner ➤ Fighter {'A' if predicted_label == 0 else 'B'} ({name_a if predicted_label == 0 else name_b})")
    print(f"🎯 Actual Winner ➤ Fighter {'A' if actual_label == 0 else 'B'} ({name_a if actual_label == 0 else name_b})")
    print(f"\n✅ Prediction was ➤ {'CORRECT ✅' if predicted_label == actual_label else 'WRONG ❌'}")
    break

else:
    print("❌ Couldn't find a valid fight after multiple attempts.")
