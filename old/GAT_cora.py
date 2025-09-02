import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# Load Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Define a GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        # Multi-head attention in first layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # Second layer (averaging heads output)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model, optimizer
model = GAT(dataset.num_node_features, 8, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

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
Epoch 020, Loss: 0.8889, Train: 0.9500, Val: 0.7880, Test: 0.8080
Epoch 040, Loss: 0.6748, Train: 0.9857, Val: 0.7880, Test: 0.8210
Epoch 060, Loss: 0.5477, Train: 1.0000, Val: 0.8020, Test: 0.8230
Epoch 080, Loss: 0.4567, Train: 1.0000, Val: 0.7960, Test: 0.8100
Epoch 100, Loss: 0.4748, Train: 1.0000, Val: 0.7920, Test: 0.8100
Epoch 120, Loss: 0.4890, Train: 1.0000, Val: 0.7900, Test: 0.8050
Epoch 140, Loss: 0.3295, Train: 1.0000, Val: 0.7780, Test: 0.7960
Epoch 160, Loss: 0.3981, Train: 1.0000, Val: 0.7980, Test: 0.8090
Epoch 180, Loss: 0.4279, Train: 1.0000, Val: 0.7720, Test: 0.7920
Epoch 200, Loss: 0.4000, Train: 1.0000, Val: 0.7940, Test: 0.7970
'''