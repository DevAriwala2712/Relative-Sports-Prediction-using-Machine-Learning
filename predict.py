import pickle
import numpy as np
import random

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load features and labels
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Pick a random index
idx = random.randint(0, len(X) - 1)
sample = X[idx].reshape(1, -1)
actual_label = y[idx]

# Make prediction
predicted_label = model.predict(sample)[0]

# Output
print("🧪 Random Fight Prediction\n")
print(f"📊 Features: {X[idx]}")
print(f"🎯 Actual Winner : Fighter {'A' if actual_label == 0 else 'B'}")
print(f"🔮 Predicted Winner : Fighter {'A' if predicted_label == 0 else 'B'}")

# Accuracy check
print("\n✅ Prediction was", "CORRECT ✅" if predicted_label == actual_label else "WRONG ❌")
