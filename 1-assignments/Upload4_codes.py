#------------------------------------------------------------------------------------------------------------
# This code is
#

import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, norm
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def build_graph(Z, R, y):
    x = torch.tensor(Z, dtype=torch.float).reshape(-1, 1)
    pos = torch.tensor(R, dtype=torch.float)

    y = torch.tensor(y, dtype=torch.float).reshape(1, -1)  # ✅ fix: ensures [1, 6] shape
    # y = torch.nn.functional.normalize(y) # This is an attempt to normalize the input
    edge_index = radius_graph(pos, r=5.0, loop=False)
    return Data(x=x, pos=pos, edge_index=edge_index, y=y)

# Load the .npz file
data = np.load('/home/a/Downloads/DFT_all.npz',allow_pickle=True)

epoch_max = 50 #Number of epochs we train the network

# Initialize lists to store metrics for plotting
train_losses, val_losses, val_mae_totals, val_mae_per_target = [], [], [], []

Zs = data['atoms']             # atomic numbers (per molecule)
Rs = data['coordinates']       # 3D positions (per molecule)

# Extract dipole components
dipoles = data['dipole']
dipole_x = np.array([d[0] for d in dipoles])
dipole_y = np.array([d[1] for d in dipoles])
dipole_z = np.array([d[2] for d in dipoles])

# Define targets
targets = ['U0', 'gap', 'H']
Y_all = np.column_stack([
    data['U0'], data['gap'], data['H'],
    dipole_x, dipole_y, dipole_z
])

scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y_all)
Ys = list(Y_scaled)

# 🔽 NEW: Split the data into train/val/test
from sklearn.model_selection import train_test_split

Z_train, Z_temp, R_train, R_temp, Y_train, Y_temp = train_test_split(Zs, Rs, Ys, test_size=0.2, random_state=42)
Z_val, Z_test, R_val, R_test, Y_val, Y_test = train_test_split(Z_temp, R_temp, Y_temp, test_size=0.5, random_state=42)

# 🔽 Build separate datasets
train_dataset = [build_graph(z, r, y) for z, r, y in zip(Z_train, R_train, Y_train)]
val_dataset   = [build_graph(z, r, y) for z, r, y in zip(Z_val, R_val, Y_val)]
test_dataset  = [build_graph(z, r, y) for z, r, y in zip(Z_test, R_test, Y_test)]

# 🔽 Use separate data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)



dataset = [build_graph(z, r, y) for z, r, y in zip(Zs, Rs, Ys)]
loader = DataLoader(dataset, batch_size=32, shuffle=True)


class GNNModel(Module):
    def __init__(self, hidden_dim=64, output_dim=6):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)  # input: atomic number
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)       # Global graph representation
        x = F.relu(self.lin1(x))
        out = self.lin2(x)                   # Multi-target output
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(1, epoch_max+1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * batch.num_graphs

    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # ===== Validation starts here =====
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y
            val_loss += loss_fn(pred, target).item() * batch.num_graphs

            val_preds.append(pred.cpu().numpy())
            val_targets.append(target.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # ==== MAE (real units)
    val_preds_real = scaler.inverse_transform(np.vstack(val_preds))
    val_targets_real = scaler.inverse_transform(np.vstack(val_targets))

    val_mae_real = mean_absolute_error(val_targets_real, val_preds_real)
    val_mae_totals.append(val_mae_real)

    # ==== Per-target MAEs
    per_target_mae = []
    for i in range(val_targets_real.shape[1]):
        mae_i = mean_absolute_error(val_targets_real[:, i], val_preds_real[:, i])
        per_target_mae.append(mae_i)

    val_mae_per_target.append(per_target_mae)

    # Format per-target MAEs for display
    target_names = ['U0', 'gap', 'H', 'dip_x', 'dip_y', 'dip_z']
    mae_str = " | ".join([f"{name}: {mae:.4f}" for name, mae in zip(target_names, per_target_mae)])

    print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE (total): {val_mae_real:.4f}")
    print(f"  MAE per target: {mae_str}")

torch.save(model,"saves/model.nn")
np.save("saves/val_mae_totals.npy",val_mae_totals,allow_pickle = True)
np.save("saves/val_mae_per_target.npy",val_mae_per_target,allow_pickle = True)
np.save("saves/val_losses.npy",val_losses,allow_pickle = True)
np.save("saves/train_losses.npy",train_losses,allow_pickle = True)





