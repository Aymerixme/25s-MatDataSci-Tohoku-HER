import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module

# Load the .npz file
data = np.load('/home/a/Downloads/DFT_all.npz',allow_pickle=True)

epoch_max = 1000 #Number of epochs we train the network

print(list(data.keys()))


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
Ys = list(Y_all)


def build_graph(Z, R, y):
    x = torch.tensor(Z, dtype=torch.float).reshape(-1, 1)
    pos = torch.tensor(R, dtype=torch.float)

    y = torch.tensor(y, dtype=torch.float).reshape(1, -1)  # ✅ fix: ensures [1, 6] shape

    edge_index = radius_graph(pos, r=5.0, loop=False)
    return Data(x=x, pos=pos, edge_index=edge_index, y=y)


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


def get(self, idx):
    Z = self.Zs[idx]
    R = self.Rs[idx]
    y = self.Ys[idx]

    x = torch.tensor(Z, dtype=torch.float).reshape(-1, 1)
    pos = torch.tensor(R, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)  # no reshape

    edge_index = radius_graph(pos, r=self.radius, loop=False)
    return Data(x=x, pos=pos, edge_index=edge_index, y=y)


# Send model to train mode
model.train()

for epoch in range(1, epoch_max+1):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        pred = model(batch)  # shape [batch_size, 6]
        target = batch.y     # shape [batch_size, 6]

        # Confirm shapes match
        if pred.shape != target.shape:
            print(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
            continue  # skip batch or raise an error

        # Loss and backprop
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * batch.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")