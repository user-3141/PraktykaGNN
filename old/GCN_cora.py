import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Load the Cora citation dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Define a simple 2-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model, optimizer
model = GCN(dataset.num_node_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Test function
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
Epoch 020, Loss: 0.3215, Train: 0.9929, Val: 0.7520, Test: 0.7850
Epoch 040, Loss: 0.0884, Train: 1.0000, Val: 0.7580, Test: 0.7780
Epoch 060, Loss: 0.0479, Train: 1.0000, Val: 0.7580, Test: 0.7830
Epoch 080, Loss: 0.0386, Train: 1.0000, Val: 0.7620, Test: 0.7870
Epoch 100, Loss: 0.0311, Train: 1.0000, Val: 0.7680, Test: 0.7950
Epoch 120, Loss: 0.0488, Train: 1.0000, Val: 0.7600, Test: 0.7880
Epoch 140, Loss: 0.0299, Train: 1.0000, Val: 0.7600, Test: 0.7940
Epoch 160, Loss: 0.0207, Train: 1.0000, Val: 0.7620, Test: 0.7940
Epoch 180, Loss: 0.0275, Train: 1.0000, Val: 0.7660, Test: 0.7940
Epoch 200, Loss: 0.0366, Train: 1.0000, Val: 0.7580, Test: 0.8020
'''