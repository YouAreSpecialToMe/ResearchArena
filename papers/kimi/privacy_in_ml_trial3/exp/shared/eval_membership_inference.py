"""
Membership Inference Attack (LiRA-style) evaluation.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models import create_simclr_model, LinearClassifier
from data_loader import create_federated_datasets, get_linear_eval_dataloaders, create_membership_split
from fcl_utils import set_seed


class ShadowModel:
    """Shadow model for membership inference."""
    def __init__(self, device):
        self.model = create_simclr_model().to(device)
        self.device = device
    
    def train(self, train_loader, epochs=30):
        """Train shadow model."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        from fcl_utils import InfoNCELoss
        contrastive_criterion = InfoNCELoss(temperature=0.5)
        
        for epoch in range(epochs):
            for batch in train_loader:
                (x1, x2), _ = batch
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                optimizer.zero_grad()
                z1 = self.model(x1)
                z2 = self.model(x2)
                loss = contrastive_criterion(z1, z2)
                loss.backward()
                optimizer.step()
    
    def get_predictions(self, data_loader):
        """Get prediction scores for data."""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle both contrastive and non-contrastive data
                if isinstance(batch[0], tuple):
                    (x1, x2), _ = batch
                    x = x1
                else:
                    x, _ = batch
                
                x = x.to(self.device)
                z = self.model(x)
                # Use norm of projection as confidence score
                score = torch.norm(z, dim=1)
                scores.append(score.cpu())
        
        return torch.cat(scores)


def compute_lira_scores(target_model, shadow_models, member_loader, nonmember_loader, device):
    """
    Compute LiRA membership scores.
    Returns TPR at 0.1% FPR and AUC-ROC.
    """
    target_model.eval()
    
    # Get target model predictions
    with torch.no_grad():
        member_scores = []
        for batch in member_loader:
            if isinstance(batch[0], tuple):
                (x1, x2), _ = batch
                x = x1
            else:
                x, _ = batch
            x = x.to(device)
            z = target_model(x)
            score = torch.norm(z, dim=1)
            member_scores.append(score.cpu())
        member_scores = torch.cat(member_scores).numpy()
        
        nonmember_scores = []
        for batch in nonmember_loader:
            if isinstance(batch[0], tuple):
                (x1, x2), _ = batch
                x = x1
            else:
                x, _ = batch
            x = x.to(device)
            z = target_model(x)
            score = torch.norm(z, dim=1)
            nonmember_scores.append(score.cpu())
        nonmember_scores = torch.cat(nonmember_scores).numpy()
    
    # Simple threshold-based attack (LiRA simplified)
    # Use shadow models to estimate score distribution
    shadow_member_scores = []
    shadow_nonmember_scores = []
    
    for shadow in shadow_models:
        shadow.model.eval()
        with torch.no_grad():
            scores = []
            for batch in member_loader:
                if isinstance(batch[0], tuple):
                    (x1, x2), _ = batch
                    x = x1
                else:
                    x, _ = batch
                x = x.to(device)
                z = shadow.model(x)
                score = torch.norm(z, dim=1)
                scores.append(score.cpu())
            shadow_member_scores.append(torch.cat(scores).numpy())
            
            scores = []
            for batch in nonmember_loader:
                if isinstance(batch[0], tuple):
                    (x1, x2), _ = batch
                    x = x1
                else:
                    x, _ = batch
                x = x.to(device)
                z = shadow.model(x)
                score = torch.norm(z, dim=1)
                scores.append(score.cpu())
            shadow_nonmember_scores.append(torch.cat(scores).numpy())
    
    # Compute likelihood ratios
    # For simplicity, use Gaussian approximation
    shadow_member_mean = np.mean([np.mean(s) for s in shadow_member_scores])
    shadow_member_std = np.mean([np.std(s) for s in shadow_member_scores])
    shadow_nonmember_mean = np.mean([np.mean(s) for s in shadow_nonmember_scores])
    shadow_nonmember_std = np.mean([np.std(s) for s in shadow_nonmember_scores])
    
    # Compute membership scores
    member_likelihood = -(member_scores - shadow_member_mean) ** 2 / (2 * shadow_member_std ** 2)
    nonmember_likelihood = -(member_scores - shadow_nonmember_mean) ** 2 / (2 * shadow_nonmember_std ** 2)
    member_lira_scores = member_likelihood - nonmember_likelihood
    
    member_likelihood_nm = -(nonmember_scores - shadow_member_mean) ** 2 / (2 * shadow_member_std ** 2)
    nonmember_likelihood_nm = -(nonmember_scores - shadow_nonmember_mean) ** 2 / (2 * shadow_nonmember_std ** 2)
    nonmember_lira_scores = member_likelihood_nm - nonmember_likelihood_nm
    
    # Compute metrics
    labels = np.concatenate([np.ones(len(member_lira_scores)), np.zeros(len(nonmember_lira_scores))])
    scores_all = np.concatenate([member_lira_scores, nonmember_lira_scores])
    
    # TPR at 0.1% FPR
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(labels, scores_all)
    
    # Find TPR at 0.1% FPR
    idx = np.where(fpr <= 0.001)[0]
    if len(idx) > 0:
        tpr_at_fpr = tpr[idx[-1]]
    else:
        tpr_at_fpr = 0.0
    
    # AUC-ROC
    auc = roc_auc_score(labels, scores_all)
    
    return {
        'tpr_at_0.001_fpr': float(tpr_at_fpr),
        'auc_roc': float(auc),
        'attack_accuracy': float((member_lira_scores > 0).mean() + (nonmember_lira_scores <= 0).mean()) / 2
    }


def evaluate_membership_inference(model_path, dataset, num_clients, alpha, data_dir, 
                                   num_shadows=3, device='cuda'):
    """Evaluate membership inference attack on a trained model."""
    
    set_seed(42)
    
    # Load target model
    checkpoint = torch.load(model_path, map_location=device)
    target_model = create_simclr_model().to(device)
    target_model.load_state_dict({
        'encoder.' + k if not k.startswith('encoder.') else k: v 
        for k, v in checkpoint['encoder'].items()
    } if 'encoder' in checkpoint else checkpoint)
    
    # Get data splits
    client_datasets, test_dataset, client_indices = create_federated_datasets(
        dataset_name=dataset, num_clients=num_clients, alpha=alpha, data_dir=data_dir, seed=42
    )
    
    # Create membership split
    member_indices, nonmember_indices = create_membership_split(client_indices, holdout_ratio=0.3, seed=42)
    
    # Aggregate all member and non-member data
    all_member_indices = []
    all_nonmember_indices = []
    for client_id in range(num_clients):
        all_member_indices.extend([(client_id, idx) for idx in member_indices[client_id]])
        all_nonmember_indices.extend([(client_id, idx) for idx in nonmember_indices[client_id]])
    
    # Create simple datasets for evaluation
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            client_id, sample_idx = self.indices[idx]
            x, y = self.base_dataset[sample_idx]
            return x, y
    
    # For simplicity, use a subset for faster evaluation
    sample_size = min(1000, len(all_member_indices), len(all_nonmember_indices))
    np.random.shuffle(all_member_indices)
    np.random.shuffle(all_nonmember_indices)
    all_member_indices = all_member_indices[:sample_size]
    all_nonmember_indices = all_nonmember_indices[:sample_size]
    
    # Create dataloaders with contrastive transforms
    from data_loader import get_cifar_transforms
    contrastive_transform = get_cifar_transforms(train=True, contrastive=True)
    
    # Need to re-load with contrastive transforms
    if dataset == 'cifar10':
        from torchvision import datasets
        base_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=contrastive_transform)
    else:
        from torchvision import datasets
        base_dataset = datasets.CIFAR100(root=data_dir, train=True, transform=contrastive_transform)
    
    member_dataset = SimpleDataset(base_dataset, all_member_indices)
    nonmember_dataset = SimpleDataset(base_dataset, all_nonmember_indices)
    
    member_loader = DataLoader(member_dataset, batch_size=256, shuffle=False)
    nonmember_loader = DataLoader(nonmember_dataset, batch_size=256, shuffle=False)
    
    # Train shadow models
    print(f"Training {num_shadows} shadow models...")
    shadow_models = []
    for i in range(num_shadows):
        print(f"Training shadow model {i+1}/{num_shadows}")
        shadow = ShadowModel(device)
        
        # Create shadow dataset (random split)
        shadow_indices = np.random.choice(len(member_dataset), size=len(member_dataset)//2, replace=False)
        shadow_dataset = torch.utils.data.Subset(member_dataset, shadow_indices)
        shadow_loader = DataLoader(shadow_dataset, batch_size=256, shuffle=True)
        
        shadow.train(shadow_loader, epochs=20)
        shadow_models.append(shadow)
    
    # Compute LiRA scores
    print("Computing membership inference scores...")
    results = compute_lira_scores(target_model, shadow_models, member_loader, nonmember_loader, device)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_shadows', type=int, default=3)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = evaluate_membership_inference(
        args.model_path, args.dataset, args.num_clients, args.alpha,
        args.data_dir, args.num_shadows, device
    )
    
    results['model'] = args.model_path
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results: {results}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
