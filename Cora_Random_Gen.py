import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import pickle
import os


def generate_random_features_graph(feature_dim=None, seed=42):
    """
    Load Cora dataset, replace node features with random ones, and save the modified graph.

    Args:
        feature_dim (int): Dimension of new random features. If None, uses original dimension.
        seed (int): Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Load the original Cora dataset
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Cora', name='Cora')
    original_data = dataset[0]

    print(f"Original dataset info:")
    print(f"  Number of nodes: {original_data.num_nodes}")
    print(f"  Number of edges: {original_data.num_edges}")
    print(f"  Original feature dimension: {original_data.num_node_features}")
    print(f"  Number of classes: {dataset.num_classes}")

    # Use original feature dimension if not specified
    if feature_dim is None:
        feature_dim = original_data.num_node_features

    print(f"  New feature dimension: {feature_dim}")

    # Generate random node features
    # Using normal distribution with mean=0, std=1
    random_features = torch.randn(original_data.num_nodes, feature_dim)

    # Create new data object with random features
    modified_data = Data(
        x=random_features,
        edge_index=original_data.edge_index,
        y=original_data.y,
        train_mask=original_data.train_mask,
        val_mask=original_data.val_mask,
        test_mask=original_data.test_mask
    )

    # Create directory for saving if it doesn't exist
    save_dir = 'data/Cora_Random'
    os.makedirs(save_dir, exist_ok=True)

    # Save the modified graph data
    save_path = os.path.join(save_dir, f'cora_random_features_dim{feature_dim}_seed{seed}.pkl')

    # Save using pickle
    with open(save_path, 'wb') as f:
        pickle.dump({
            'data': modified_data,
            'num_classes': dataset.num_classes,
            'num_node_features': feature_dim,
            'num_nodes': modified_data.num_nodes,
            'num_edges': modified_data.num_edges,
            'seed': seed,
            'original_feature_dim': original_data.num_node_features
        }, f)

    print(f"\nModified graph saved to: {save_path}")
    print(f"Random features shape: {random_features.shape}")
    print(f"Random features stats - Mean: {random_features.mean():.4f}, Std: {random_features.std():.4f}")

    return modified_data, save_path


def load_random_features_graph(save_path):
    """
    Load the saved graph with random features.

    Args:
        save_path (str): Path to the saved graph file.

    Returns:
        tuple: (data, metadata)
    """
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)

    print(f"Loaded graph from: {save_path}")
    print(f"  Number of nodes: {saved_data['num_nodes']}")
    print(f"  Number of edges: {saved_data['num_edges']}")
    print(f"  Feature dimension: {saved_data['num_node_features']}")
    print(f"  Number of classes: {saved_data['num_classes']}")
    print(f"  Random seed used: {saved_data['seed']}")

    return saved_data['data'], saved_data


def demonstrate_usage():
    """Demonstrate how to create and load the modified graph."""
    print("=" * 50)
    print("CREATING RANDOM FEATURE GRAPH")
    print("=" * 50)

    # Create graph with random features (same dimension as original)
    data, save_path = generate_random_features_graph()

    print("\n" + "=" * 50)
    print("LOADING SAVED GRAPH")
    print("=" * 50)

    # Load the saved graph
    loaded_data, metadata = load_random_features_graph(save_path)

    print("\n" + "=" * 50)
    print("CREATING GRAPH WITH DIFFERENT FEATURE DIMENSION")
    print("=" * 50)

    # Create graph with different feature dimension
    data_64, save_path_64 = generate_random_features_graph(feature_dim=64, seed=123)

    return data, loaded_data, data_64


if __name__ == "__main__":
    # Run the demonstration
    data, loaded_data, data_64 = demonstrate_usage()

    print("\n" + "=" * 50)
    print("EXAMPLE: Using the random feature graph in training")
    print("=" * 50)

    print("""
# Example of how to use the saved graph in your training script:

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle

# Load your saved random feature graph
with open('data/Cora_Random/cora_random_features_dim1433_seed42.pkl', 'rb') as f:
    saved_data = pickle.load(f)

data = saved_data['data']
num_classes = saved_data['num_classes']
num_features = saved_data['num_node_features']

# Use in your GCN model
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

# Initialize model with the random features
model = GCN(num_features, 16, num_classes)

# Train as usual...
    """)