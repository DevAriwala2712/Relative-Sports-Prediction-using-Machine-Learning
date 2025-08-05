import numpy as np

# Load preprocessed data
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("✅ Data loaded successfully!")
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Sample X[0]: {X[0]}")
print(f"Sample y[0]: {y[0]}")
