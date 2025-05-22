import matplotlib.pyplot as plt
import numpy as np


# losses and mae to import
val_mae_totals     = np.load("saves/val_mae_totals.npy",allow_pickle = True)
# val_mae_per_target = np.transpose(np.load("saves/val_mae_per_target.npy",allow_pickle = True))
val_mae_per_target = np.load("saves/val_mae_per_target.npy",allow_pickle = True)
val_losses         = np.load("saves/val_losses.npy",allow_pickle = True)
train_losses       = np.load("saves/train_losses.npy",allow_pickle = True)


epoch_max = 50

# Name of the properties estimated
target_names = ['U0', 'gap', 'H', 'dip_x', 'dip_y', 'dip_z']


# Plot of all of them on the two different graphs (except the mae for all targets)
plt.plot(np.linspace(1,len(val_mae_totals),len(val_mae_totals)),val_mae_totals,label = "mae total")
plt.legend()
plt.show()

plt.plot(np.linspace(1,len(val_losses),len(val_losses)),val_losses,label = "validation losses")
plt.plot(np.linspace(1,len(train_losses),len(train_losses)),train_losses,label = "train losses")
plt.legend()
plt.show()

# Plot of the MAE per target
for k in range(len(val_mae_per_target)):
    # fig = plt.figure()
    plt.plot(np.linspace(1,len(val_mae_per_target[k])),val_mae_per_target[k],len(val_mae_per_target[k]), label = target_names[k])
    # plt.xlabel("epochs")
    # plt.ylabel(target_names[k])
    # plt.axis([0,len(val_mae_per_target[k]),0.9*np.min(val_mae_per_target[k]),np.max(val_mae_per_target[k])*1.1])
plt.legend()
plt.show()


# === Plot 1: Training vs Validation Loss ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch_max + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epoch_max + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: Validation MAE Over Epochs ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch_max + 1), val_mae_totals, label='Validation MAE (Total)', color='green', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Validation MAE Over Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Per-Target MAEs Over Epochs ===
val_mae_per_target = np.array(val_mae_per_target)  # shape: (epoch_max, num_targets)

plt.figure(figsize=(12, 6))
for i in range(len(target_names)):
    plt.plot(range(1, epoch_max + 1), val_mae_per_target[:, i], label=f"{target_names[i]}")
plt.xlabel('Epoch')
plt.ylabel('MAE (per target)')
plt.title('Per-Target Validation MAE Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# CNN One poperty

torch.load
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

# Best validation MAE and the epoch where it occurred
best_val_mae = min(history.history['val_mae'])
best_val_mae_epoch = history.history['val_mae'].index(best_val_mae) + 1

# Best validation loss (MSE) and the epoch where it occurred
best_val_loss = min(history.history['val_loss'])
best_val_loss_epoch = history.history['val_loss'].index(best_val_loss) + 1

# Early stopping epoch = total number of epochs trained
early_stopping_epoch = len(history.history['loss'])

# Print results
print(f"Best Validation MAE: {best_val_mae:.4f} at Epoch {best_val_mae_epoch}")
print(f"Best Validation Loss (MSE): {best_val_loss:.4f} at Epoch {best_val_loss_epoch}")

# Predict
Y_pred_scaled = model.predict(X_test)
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)


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



# Extract loss values from training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Find the epoch with the best validation loss
best_val_epoch = val_loss.index(min(val_loss)) + 1

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Training Loss (MSE)')
plt.plot(epochs, val_loss, label='Validation Loss (MSE)')
plt.axvline(x=best_val_epoch, linestyle='--', color='gray', label=f'Best Val Loss (Epoch {best_val_epoch})')

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (Loss)')
plt.title('Training vs Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()