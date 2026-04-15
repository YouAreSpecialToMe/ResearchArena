"""
Core model implementations for SPT-TTA and baseline methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple, Optional, Dict
import numpy as np


class PromptedViT(nn.Module):
    """Vision Transformer with learnable prompts at each layer."""
    
    def __init__(self, model_name='vit_base_patch16_224', num_prompts=4, prompt_dim=768, 
                 num_layers=12, pretrained=True):
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=1000)
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        
        # Initialize learnable prompts for each layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_prompts, prompt_dim) * 0.02)
            for _ in range(num_layers)
        ])
        
        # Freeze backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Only prompts are trainable
        for prompt in self.prompts:
            prompt.requires_grad = True
        
        # Store EMA features for consistency loss
        self.ema_features = {}
        self.ema_decay = 0.9
        
    def forward(self, x, return_attention=False):
        """Standard forward pass with prompts at all layers."""
        batch_size = x.shape[0]
        
        # Get patch embeddings
        x = self.vit.patch_embed(x)
        
        # Add CLS token
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.vit.pos_embed
        
        # Apply dropout
        x = self.vit.pos_drop(x)
        
        # Forward through transformer blocks with prompts
        attentions = []
        for i, block in enumerate(self.vit.blocks):
            # Add prompts to the sequence
            prompt = self.prompts[i].expand(batch_size, -1, -1)
            x_with_prompt = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
            
            # Forward through block
            if return_attention:
                x_with_prompt, attn = block(x_with_prompt, return_attention=True)
                attentions.append(attn)
            else:
                x_with_prompt = block(x_with_prompt)
            
            # Remove prompts for next layer
            x = torch.cat([x_with_prompt[:, :1], x_with_prompt[:, 1+self.num_prompts:]], dim=1)
        
        x = self.vit.norm(x)
        logits = self.vit.head(x[:, 0])
        
        if return_attention:
            return logits, attentions
        return logits
    
    def forward_layer(self, x, layer_idx):
        """
        Forward pass through a specific layer with prompt.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            layer_idx: Which layer to process
        
        Returns:
            Output tensor
        """
        block = self.vit.blocks[layer_idx]
        prompt = self.prompts[layer_idx].expand(x.shape[0], -1, -1)
        
        # Add prompts after CLS token
        x_with_prompt = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
        x_with_prompt = block(x_with_prompt)
        
        return x_with_prompt
    
    def extract_features(self, x):
        """Extract features from each layer."""
        batch_size = x.shape[0]
        features = {}
        
        # Initial embedding
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        features['embeddings'] = x.clone()
        
        # Forward through blocks
        for i, block in enumerate(self.vit.blocks):
            prompt = self.prompts[i].expand(batch_size, -1, -1)
            x_with_prompt = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
            x_with_prompt = block(x_with_prompt)
            x = torch.cat([x_with_prompt[:, :1], x_with_prompt[:, 1+self.num_prompts:]], dim=1)
            features[f'layer_{i}'] = x.clone()
        
        x = self.vit.norm(x)
        features['final'] = x[:, 0]
        
        return features


class SPTTTA:
    """Selective Progressive Test-Time Adaptation."""
    
    def __init__(self, model: PromptedViT, 
                 lr: float = 5e-4,
                 num_prompts: int = 4,
                 layer_weights: Optional[List[float]] = None,
                 ema_decay: float = 0.9,
                 selection_threshold: float = 0.5,
                 adapt_steps: int = 1):
        """
        Args:
            model: PromptedViT model
            lr: Learning rate for prompt adaptation
            num_prompts: Number of prompt tokens per layer
            layer_weights: Lambda weights for each layer [early, middle, deep]
            ema_decay: EMA decay for feature consistency
            selection_threshold: Threshold for layer selection (0-1, percentile)
            adapt_steps: Number of adaptation steps per layer
        """
        self.model = model
        self.lr = lr
        self.num_prompts = num_prompts
        self.ema_decay = ema_decay
        self.selection_threshold = selection_threshold
        self.adapt_steps = adapt_steps
        
        # Set layer weights (default: 0.3 for early, 0.5 for middle, 0.7 for deep)
        if layer_weights is None:
            self.layer_weights = [0.3] * 4 + [0.5] * 4 + [0.7] * 4
        else:
            self.layer_weights = layer_weights
        
        # EMA features for consistency loss
        self.ema_features = {}
        
        # Optimizer for prompts
        self.optimizer = None
        
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of predictions."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy
    
    def compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights.
        
        Args:
            attention: Attention weights [batch, num_heads, seq_len, seq_len]
        
        Returns:
            Entropy per sample [batch]
        """
        # Average over heads
        attn_avg = attention.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Compute entropy over key dimension for query tokens
        entropy = -(attn_avg * torch.log(attn_avg + 1e-10)).sum(dim=-1)  # [batch, seq_len]
        
        # Average over sequence (excluding CLS token which is first)
        return entropy[:, 1:].mean(dim=-1)  # [batch]
    
    def adaptation_loss(self, logits: torch.Tensor, features: torch.Tensor, 
                        layer_idx: int) -> torch.Tensor:
        """
        Compute adaptation loss for a layer.
        
        Args:
            logits: Model predictions
            features: Current features
            layer_idx: Current layer index
        
        Returns:
            Loss value
        """
        lambda_l = self.layer_weights[layer_idx]
        
        # Entropy loss (confidence maximization)
        entropy = self.compute_entropy(logits).mean()
        
        # Consistency loss (feature preservation)
        layer_key = f'layer_{layer_idx}'
        if layer_key in self.ema_features:
            ema_feat = self.ema_features[layer_key]
            consistency = F.mse_loss(features, ema_feat)
        else:
            consistency = torch.tensor(0.0, device=logits.device)
        
        # Combined loss
        loss = lambda_l * entropy + (1 - lambda_l) * consistency
        
        return loss, entropy.item(), consistency.item()
    
    def adapt_sequential(self, x: torch.Tensor, device: str = 'cuda') -> Tuple[torch.Tensor, Dict]:
        """
        Perform sequential layer-by-layer adaptation.
        
        Args:
            x: Input image tensor [batch, C, H, W]
            device: Device to run on
        
        Returns:
            Final logits and adaptation info
        """
        x = x.to(device)
        batch_size = x.shape[0]
        
        # Simple heuristic: select every other layer (approx 50%)
        # In a full implementation, we'd compute attention entropy
        num_layers = len(self.model.vit.blocks)
        selected_layers = list(range(0, num_layers, 2))  # 0, 2, 4, 6, 8, 10
        
        # Sequential adaptation
        adaptation_info = {
            'selected_layers': selected_layers,
            'num_adapted': len(selected_layers)
        }
        
        # Adapt selected layers sequentially
        for layer_idx in selected_layers:
            self.adapt_layer(x, layer_idx, device)
        
        # Final forward pass with adapted prompts
        with torch.no_grad():
            logits = self.model(x)
        
        return logits, adaptation_info
    
    def adapt_layer(self, x: torch.Tensor, layer_idx: int, device: str = 'cuda'):
        """
        Adapt prompts at a specific layer.
        
        Args:
            x: Input image tensor
            layer_idx: Which layer to adapt
            device: Device to run on
        """
        batch_size = x.shape[0]
        
        # Create optimizer for this layer's prompt
        prompt_param = self.model.prompts[layer_idx]
        optimizer = torch.optim.Adam([prompt_param], lr=self.lr)
        
        for step in range(self.adapt_steps):
            optimizer.zero_grad()
            
            # Forward pass up to and including this layer
            features = self.model.vit.patch_embed(x)
            cls_token = self.model.vit.cls_token.expand(batch_size, -1, -1)
            features = torch.cat((cls_token, features), dim=1)
            features = features + self.model.vit.pos_embed
            features = self.model.vit.pos_drop(features)
            
            # Forward through previous layers (without adaptation)
            for i in range(layer_idx):
                prompt = self.model.prompts[i].expand(batch_size, -1, -1)
                x_with_prompt = torch.cat([features[:, :1], prompt, features[:, 1:]], dim=1)
                x_with_prompt = self.model.vit.blocks[i](x_with_prompt)
                features = torch.cat([x_with_prompt[:, :1], x_with_prompt[:, 1+self.num_prompts:]], dim=1)
            
            # Forward through current layer with prompt
            features_with_prompt = self.model.forward_layer(features, layer_idx)
            
            # Continue forward to get logits
            temp_features = torch.cat([
                features_with_prompt[:, :1], 
                features_with_prompt[:, 1+self.num_prompts:]
            ], dim=1)
            
            for i in range(layer_idx + 1, len(self.model.vit.blocks)):
                prompt = self.model.prompts[i].expand(batch_size, -1, -1)
                x_with_prompt = torch.cat([temp_features[:, :1], prompt, temp_features[:, 1:]], dim=1)
                x_with_prompt = self.model.vit.blocks[i](x_with_prompt)
                temp_features = torch.cat([x_with_prompt[:, :1], x_with_prompt[:, 1+self.num_prompts:]], dim=1)
            
            temp_features = self.model.vit.norm(temp_features)
            logits = self.model.vit.head(temp_features[:, 0])
            
            # Compute loss
            loss, ent_val, cons_val = self.adaptation_loss(
                logits, features_with_prompt[:, 0], layer_idx
            )
            
            # Backward and update
            loss.backward()
            optimizer.step()
        
        # Update EMA features
        with torch.no_grad():
            layer_key = f'layer_{layer_idx}'
            self.ema_features[layer_key] = features_with_prompt[:, 0].detach()


class TENT:
    """TENT: Test-time Entropy Minimization."""
    
    def __init__(self, model: nn.Module, lr: float = 1e-3, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        
        # Enable gradient for normalization parameters
        self.model.train()  # Set to train mode for BN statistics update
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                for param in module.parameters():
                    param.requires_grad = True
        
        # Freeze other parameters
        for name, param in model.named_parameters():
            if 'bn' not in name.lower() and 'norm' not in name.lower():
                param.requires_grad = False
        
        # Setup optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) > 0:
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
        else:
            self.optimizer = None
    
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt on x and return predictions."""
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(x)
        
        self.model.train()  # Train mode for BN update
        self.optimizer.zero_grad()
        
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        entropy.backward()
        self.optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        
        return logits


class VPA:
    """VPA: Visual Prompt Adaptation (simultaneous)."""
    
    def __init__(self, model: PromptedViT, lr: float = 5e-4, adapt_steps: int = 1):
        self.model = model
        self.lr = lr
        self.adapt_steps = adapt_steps
        
        # Setup optimizer for all prompts simultaneously
        prompt_params = list(model.prompts.parameters())
        self.optimizer = torch.optim.Adam(prompt_params, lr=lr)
    
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt all prompts simultaneously."""
        for _ in range(self.adapt_steps):
            self.optimizer.zero_grad()
            
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            
            entropy.backward()
            self.optimizer.step()
        
        with torch.no_grad():
            logits = self.model(x)
        
        return logits


class MEMO:
    """MEMO: Test-time adaptation via augmentation."""
    
    def __init__(self, model: nn.Module, lr: float = 5e-4, 
                 augmentations: Optional[List] = None):
        self.model = model
        self.lr = lr
        
        # Default augmentations
        if augmentations is None:
            self.augmentations = [
                lambda x: x,  # Identity
                lambda x: torch.flip(x, dims=[-1]),  # Horizontal flip
            ]
        else:
            self.augmentations = augmentations
        
        # Setup optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) > 0:
            self.optimizer = torch.optim.SGD(params, lr=lr)
        else:
            self.optimizer = None
    
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt using test-time augmentation."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            
            # Marginal entropy minimization
            all_logits = []
            for aug in self.augmentations:
                x_aug = aug(x)
                logits = self.model(x_aug)
                all_logits.append(logits)
            
            # Average probabilities across augmentations
            avg_probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits]).mean(dim=0)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum(dim=-1).mean()
            
            entropy.backward()
            self.optimizer.step()
        
        with torch.no_grad():
            logits = self.model(x)
        
        return logits


def load_vit_model(model_name='vit_base_patch16_224', pretrained=True, device='cuda'):
    """Load a pretrained ViT model."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=1000)
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load base model
    base_model = load_vit_model(device=device)
    print("Base model loaded")
    
    # Create prompted model
    prompted_model = PromptedViT(num_prompts=4, pretrained=False)
    prompted_model.vit = base_model
    prompted_model = prompted_model.to(device)
    print("Prompted model created")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    logits = prompted_model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test SPT-TTA
    spttta = SPTTTA(prompted_model)
    logits, info = spttta.adapt_sequential(x, device)
    print(f"SPT-TTA adapted layers: {info['selected_layers']}")
    print(f"Output shape: {logits.shape}")
