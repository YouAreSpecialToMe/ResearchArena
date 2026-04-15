"""Model loading utilities for ViT experiments."""
import timm
import torch


MODEL_CONFIGS = {
    'deit_small': {
        'name': 'deit_small_patch16_224',
        'num_layers': 12,
        'embed_dim': 384,
        'default_stg_layers': [3, 6, 9, 11],
        'batch_size': 256,
        'architecture': 'columnar',
    },
    'deit_base': {
        'name': 'deit_base_patch16_224',
        'num_layers': 12,
        'embed_dim': 768,
        'default_stg_layers': [3, 6, 9, 11],
        'batch_size': 128,
        'architecture': 'columnar',
    },
    'swin_tiny': {
        'name': 'swin_tiny_patch4_window7_224',
        'num_layers': 12,  # 2+2+6+2 blocks across 4 stages
        'embed_dim': 96,  # base embed dim
        'default_stg_layers': [1, 3, 9, 11],  # last block of each stage
        'batch_size': 256,
        'architecture': 'hierarchical',
    },
}


def load_model(model_key, device='cuda'):
    """Load a pretrained model from timm."""
    config = MODEL_CONFIGS[model_key]
    model = timm.create_model(config['name'], pretrained=True)
    model = model.to(device)
    model.eval()
    return model, config


def get_model_blocks(model, model_key):
    """Get the list of transformer blocks for a model."""
    config = MODEL_CONFIGS[model_key]
    if config['architecture'] == 'columnar':
        # DeiT models have model.blocks[i]
        return list(model.blocks)
    elif config['architecture'] == 'hierarchical':
        # Swin has model.layers[stage].blocks[block]
        blocks = []
        for stage in model.layers:
            for block in stage.blocks:
                blocks.append(block)
        return blocks
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")
