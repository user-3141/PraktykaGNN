import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# Load Cora citation dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Define model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Model, optimizer
model = GraphSAGE(dataset.num_node_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs

# Run training
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

'''
Epoch 020, Loss: 0.0000, Train: 1.0000, Val: 0.7760, Test: 0.7770
Epoch 040, Loss: 0.0002, Train: 1.0000, Val: 0.7780, Test: 0.7730
Epoch 060, Loss: 0.0026, Train: 1.0000, Val: 0.7640, Test: 0.7970
Epoch 080, Loss: 0.0032, Train: 1.0000, Val: 0.7560, Test: 0.8030
Epoch 100, Loss: 0.0032, Train: 1.0000, Val: 0.7540, Test: 0.8040
Epoch 120, Loss: 0.0030, Train: 1.0000, Val: 0.7560, Test: 0.8050
Epoch 140, Loss: 0.0029, Train: 1.0000, Val: 0.7600, Test: 0.8040
Epoch 160, Loss: 0.0027, Train: 1.0000, Val: 0.7620, Test: 0.7990
Epoch 180, Loss: 0.0026, Train: 1.0000, Val: 0.7660, Test: 0.7990
Epoch 200, Loss: 0.0025, Train: 1.0000, Val: 0.7660, Test: 0.7990
'''