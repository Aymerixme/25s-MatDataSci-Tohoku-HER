#------------------------------------------------------------------------------------------------------------
# This code is an infructuous test for a GNN applied to the QM9 dataset.
#

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


# === CONFIGURATION ===
xyz_dir = "/home/a/Downloads/dsgdb9nsd.xyz/dsgdb9nsd.xyz"
max_files = 1000

# === FIX FOR SCIENTIFIC NOTATION ===
def safe_float(s):
    return float(s.replace('*^-', 'e-').replace('*^', 'e'))


# === QM9 PARSER ===
def parse_xyz_file(filename):
    filepath = os.path.join(xyz_dir, filename)
    try:
        with open(filepath, "r") as file:
            lines = file.readlines()
        n_atoms = int(lines[0].strip())
        properties_line = lines[1].strip()
        atom_lines = lines[2:2 + n_atoms]
        atoms = []
        coords = []
        for line in atom_lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([safe_float(p) for p in parts[1:4]])
        return {
            "atomic_numbers": atoms,
            "coordinates": coords,
            "raw_properties": properties_line
        }
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


# === PARSE ALL QM9 FILES ===
def read_all_xyz_files_parallel():
    file_list = sorted([f for f in os.listdir(xyz_dir) if f.endswith('.xyz')])
    if max_files is not None:
        file_list = file_list[:max_files]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(parse_xyz_file, file_list), total=len(file_list)))
    return pd.DataFrame([r for r in results if r is not None])


# === DFT PARSING ===
dft_data = np.load("/home/a/Downloads/DFT_all.npz", allow_pickle=True)
df_dft = pd.DataFrame({
    "gap": dft_data['gap'],
    "U0": dft_data['U0'],
    "H": dft_data['H'],
    "Cv": dft_data['Cv'],
    "atomic_numbers": list(dft_data['atoms']),
    "coordinates": list(dft_data['coordinates']),
})

# === QM9 PARSING & PROPERTY EXTRACTION ===
df_qm9 = read_all_xyz_files_parallel()

df_qm9["gap"] = df_qm9["raw_properties"].apply(lambda x: safe_float(x.split()[5]) if len(x.split()) > 5 else np.nan)
df_qm9["U0"]  = df_qm9["raw_properties"].apply(lambda x: safe_float(x.split()[11]) if len(x.split()) > 11 else np.nan)
df_qm9["H"]   = df_qm9["raw_properties"].apply(lambda x: safe_float(x.split()[12]) if len(x.split()) > 12 else np.nan)
df_qm9["Cv"]  = df_qm9["raw_properties"].apply(lambda x: safe_float(x.split()[16]) if len(x.split()) > 16 else np.nan)


df_qm9.dropna(subset=["gap", "U0", "H", "Cv"], inplace=True)
df_qm9 = df_qm9[["atomic_numbers", "coordinates", "gap", "U0", "H", "Cv"]]

# === COMBINE & NORMALIZE ===
df_all = pd.concat([df_dft, df_qm9], ignore_index=True)

scaler = StandardScaler()
df_all[["gap", "U0", "H", "Cv"]] = scaler.fit_transform(df_all[["gap", "U0", "H", "Cv"]])


# Atomic symbol to atomic number mapping
atomic_number_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

def build_graph(row):
    # Handle atomic symbol conversion (for QM9 molecules)
    if isinstance(row.atomic_numbers[0], str):
        z = torch.tensor([atomic_number_map[a] for a in row.atomic_numbers], dtype=torch.long)
    else:
        z = torch.tensor(row.atomic_numbers, dtype=torch.long)

    pos = torch.tensor(row.coordinates, dtype=torch.float)
    x = z.view(-1, 1).float()

    # Fully connected graph (excluding self-loops)
    edge_index = torch.combinations(torch.arange(len(z)), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    y = torch.tensor([[row.gap, row.U0, row.H, row.Cv]], dtype=torch.float)  # shape (1, 4)

    return Data(x=x, edge_index=edge_index, pos=pos, y=y)


data_list = []

for row in tqdm(df_all.itertuples(index=False), total=len(df_all)):
    try:
        data = build_graph(row)
        data_list.append(data)
    except Exception as e:
        print("Skipping row due to error:", e)



# A simple 2-layer GCN followed by global mean pooling and MLP
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 64)  # First GCN layer (input: atomic number)
        self.conv2 = GCNConv(64, 128)  # Second GCN layer
        self.lin1 = Linear(128, 64)  # Fully connected layer
        self.lin2 = Linear(64, 4)    # Output layer for 4 molecular properties: [gap, U0, H, Cv]

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))        # Apply first GCN + ReLU
        x = F.relu(self.conv2(x, edge_index))        # Apply second GCN + ReLU
        x = global_mean_pool(x, batch)               # Pool node features to graph-level
        x = F.relu(self.lin1(x))                     # FC layer + ReLU
        return self.lin2(x)                          # Output 4 scalar values


from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# Split the dataset into training and testing (90% train / 10% test)
train_data, test_data = train_test_split(data_list, test_size=0.1, random_state=42)

# DataLoader automatically batches variable-sized graphs for training
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)


device = torch.device('cuda' if torch.cuda.is_available() else 'mpu' if torch.backends.mps.is_available() else 'cpu')

# Instantiate the model and optimizer
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()  # Add this line


best_mae = float('inf')
best_model_path = "saves/best_gnn_model.pt"
per_property_maes = {name: [] for name in ["gap", "U0", "H", "Cv"]}


from sklearn.metrics import mean_absolute_error, mean_squared_error

train_losses = []
val_losses = []
val_maes = []

for epoch in range(1, 50):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === Validation Phase ===
    model.eval()
    total_val_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            total_val_loss += loss_fn(pred, batch.y).item()
            y_true.append(batch.y.cpu())
            y_pred.append(pred.cpu())

    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    epoch_mae = mean_absolute_error(y_true, y_pred)
    val_maes.append(epoch_mae)

    # Track per-property MAE
    for i, name in enumerate(["gap", "U0", "H", "Cv"]):
        prop_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        per_property_maes[name].append(prop_mae)

    # Save best model
    if epoch_mae < best_mae:
        best_mae = epoch_mae
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch {epoch:02d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} | Val MAE: {epoch_mae:.4f} | ✅ {'Saved' if epoch_mae == best_mae else ''}")



# Final test evaluation
model.load_state_dict(torch.load(best_model_path))  # Load best model
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        y_true.append(batch.y.cpu())
        y_pred.append(pred.cpu())

# Stack and convert
y_true = torch.cat(y_true, dim=0).numpy()
y_pred = torch.cat(y_pred, dim=0).numpy()

# Metrics (normalized)
test_loss_mse = mean_squared_error(y_true, y_pred)
test_mae = mean_absolute_error(y_true, y_pred)
np.save("saves/5_test_loss_mse.npy",test_loss_mse)
np.save("saves/5_test_mae.npy",test_mae)

# Print Summary
print("\n📊 Final Model Performance Summary")
print(f"🧪 Test MSE Loss        : {test_loss_mse:.4f}")
print(f"🧪 Test MAE             : {test_mae:.4f}")
print(f"🏆 Best Val MSE Loss    : {min(val_losses):.4f}")
print(f"🏆 Best Val MAE         : {min(val_maes):.4f}")



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train MSE")
plt.plot(val_losses, label="Val MSE")
plt.plot(val_maes, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Validation Loss & MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("saves/Train_Validation_Loss_MAE.pdf", format = 'pdf')
# plt.show()

plt.figure(figsize=(10, 6))
for name, values in per_property_maes.items():
    plt.plot(values, label=f"Val MAE: {name}")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Per-Property Validation MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("saves/Per_Property_Validation_MAE.pdf", format = 'pdf')
# plt.show()


best_model_path = "best_gnn_model.pt"

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train MSE", linewidth=2)
plt.plot(val_losses, label="Val MSE", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs. Validation MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("saves/Training_vs_Validation_MSE_Loss.pdf", format = 'pdf')
# plt.show()









































