import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import pickle
import Cora_Random_Gen
from Cora_info import class_names
import Models
import Eval_util

'''
# Load the Cora citation dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]
num_classes = dataset.num_classes
num_features = dataset.num_node_features
'''
# Load saved random feature graph
with open('data/Cora_Random/cora_random_features_dim64_seed123.pkl', 'rb') as f:
    saved_data = pickle.load(f)
data = saved_data['data']
num_classes = saved_data['num_classes']
num_features = saved_data['num_node_features']

# ------------------------------------------------------------------------------------
# Model, optimizer
'''
model = Models.GCN(num_features, 16, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model = Models.GAT(num_features, 8, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
'''
model = Models.GraphSAGE(num_features, 64, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss
# ------------------------------------------------------------------------------------

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

# Test function with detailed output for analysis
def test_detailed():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs, pred, out

# Run training
print("Starting training "+model.__class__.__name__)
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
print("\nTraining completed!")

# Final evaluation and analysis
print("\n" + "=" * 80)
print("FINAL COMPREHENSIVE ANALYSIS")
print("=" * 80)

# Get final predictions
accs, final_pred, final_out = test_detailed()
train_acc, val_acc, test_acc = accs

# Class distribution analysis
dist_df = Eval_util.analyze_class_distribution(data)

# Detailed analysis for each split
test_metrics = Eval_util.detailed_performance_analysis(data, final_pred, data.y, data.test_mask, "test")
val_metrics = Eval_util.detailed_performance_analysis(data, final_pred, data.y, data.val_mask, "validation")
train_metrics = Eval_util.detailed_performance_analysis(data, final_pred, data.y, data.train_mask, "training")

# Confidence analysis
Eval_util.analyze_predictions_confidence(final_out, data.test_mask, "test")
Eval_util.analyze_predictions_confidence(final_out, data.val_mask, "validation")

# Visualizations
print("\nGenerating visualizations...")
Eval_util.plot_confusion_matrix(final_pred, data.y, data.test_mask, "Test")
Eval_util.plot_class_performance(test_metrics, "Test")

print("\nAnalysis complete!")