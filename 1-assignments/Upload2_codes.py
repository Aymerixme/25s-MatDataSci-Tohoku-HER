import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter

# Load the .npz file
data = np.load('/home/a/Downloads/DFT_all.npz',allow_pickle=True)#The path must change depending on the folder the data base has been downloaded on
coordinates = data['coordinates']  # shape: (n_atoms, 3)
atoms = data['atoms']              # shape: (n_atoms,)

# Filter fixed-length molecules (same n_atoms)
n_atoms_arr = np.array([len(a) for a in atoms])
most_common_n = Counter(n_atoms_arr).most_common(1)[0][0]

filtered_data = [
    (coord, atom) for coord, atom in zip(coordinates, atoms)
    if len(atom) == most_common_n
]
filtered_coords = [c for c, _ in filtered_data]
filtered_atoms = [a for _, a in filtered_data]

# Construct X as 3D input: [x, y, z, Z]
X = np.array([
    np.hstack([coord, atom.reshape(-1, 1)])  # shape: (n_atoms, 4)
    for coord, atom in zip(filtered_coords, filtered_atoms)
])

# Select targets
targets = ['U0', 'H', 'gap']
Y_all = np.column_stack([data[prop] for prop in targets])
Y = np.array([
    y for y, atom in zip(Y_all, atoms) if len(atom) == most_common_n
])

# Normalize X per feature
n_atoms = most_common_n
X_reshaped = X  # shape: (samples, n_atoms, 4)
X_flat = X_reshaped.reshape(-1, 4)
x_scaler = StandardScaler().fit(X_flat)
X_scaled = x_scaler.transform(X_flat).reshape(-1, n_atoms, 4)

# Normalize Y
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Build Conv1D model
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_atoms, 4)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),  # Flatten before dense layers
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(targets))
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Predict
Y_pred_scaled = model.predict(X_test)
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

import matplotlib.pyplot as plt

# Extract MAE from training history
train_mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(train_mae) + 1)

# Identify best epoch (lowest validation MAE)
best_epoch = val_mae.index(min(val_mae)) + 1

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_mae, label='Training MAE')
plt.plot(epochs, val_mae, label='Validation MAE')
plt.axvline(x=best_epoch, color='gray', linestyle='--', label=f'Early Stopping (Epoch {best_epoch})')

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training vs Validation MAE Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()