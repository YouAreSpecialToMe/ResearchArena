import sys
sys.path.insert(0, 'exp/shared')

print("Step 1: Import torch")
import torch
print("  -> OK")

print("Step 2: Import models")
from models import create_simclr_model
print("  -> OK")

print("Step 3: Import data_loader")
from data_loader import create_federated_datasets, get_client_dataloader
print("  -> OK")

print("Step 4: Import fcl_utils")
from fcl_utils import InfoNCELoss, fedavg_aggregate, set_seed
print("  -> OK")

print("Step 5: Import nn and optim")
import torch.nn as nn
import torch.optim as optim
print("  -> OK")

print("Step 6: Import argparse")
import argparse
print("  -> OK")

print("Step 7: Import F")
import torch.nn.functional as F
print("  -> OK")

print("\nAll imports successful!")

# Test basic functionality
print("\nTesting basic functionality...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = create_simclr_model().to(device)
print("Model created")

print("Creating datasets...")
client_datasets, test_dataset, client_indices = create_federated_datasets(
    dataset_name='cifar10', num_clients=3, alpha=0.5, data_dir='./data', seed=42
)
print(f"Created {len(client_datasets)} client datasets")

loader = get_client_dataloader(client_datasets[0], batch_size=256, num_workers=0)
print(f"Data loader: {len(loader)} batches")

print("\nAll tests passed!")
