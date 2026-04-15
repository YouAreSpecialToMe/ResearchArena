import torch
import tome.patch.timm as tome_timm_patch


def apply_tome(model, r):
    """Apply standard Token Merging with fixed ratio r."""
    tome_timm_patch(model)
    model.r = r
    return model


def unpatch_tome(model):
    """Remove ToMe patching - need to reload model."""
    pass  # ToMe doesn't have a clean unpatch for timm; reload model instead


class RandomTokenDrop:
    """Random token dropping at each layer. Drop r tokens per layer (never drop CLS)."""

    def __init__(self, model, r):
        self.model = model
        self.r = r
        self.hooks = []
        self._patch()

    def _patch(self):
        for block in self.model.blocks:
            hook = block.register_forward_pre_hook(self._drop_hook)
            self.hooks.append(hook)

    def _drop_hook(self, module, input):
        x = input[0]
        B, N, C = x.shape
        if N <= self.r + 1:
            return input
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        num_patches = patch_tokens.shape[1]
        keep = max(1, num_patches - self.r)
        idx = torch.randperm(num_patches, device=x.device)[:keep]
        idx = idx.sort().values
        patch_tokens = patch_tokens[:, idx, :]
        return (torch.cat([cls_token, patch_tokens], dim=1),)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class EViTStyle:
    """Training-free EViT-style token keeping.
    At specified layers, keep top-k tokens by CLS attention score.
    Fuse remaining tokens via mean pooling.
    NOTE: This is a training-free variant, NOT the original EViT which requires fine-tuning.
    """
    def __init__(self, model, keep_rate=0.7, apply_layers=None):
        self.model = model
        self.keep_rate = keep_rate
        self.apply_layers = apply_layers or [4, 7, 10]
        self.hooks = []
        self._last_cls_attn = {}
        self._patch()

    def _patch(self):
        for i, block in enumerate(self.model.blocks):
            if i in self.apply_layers:
                # Hook on attention to capture CLS attention weights
                attn_hook = block.attn.register_forward_hook(self._make_attn_hook(i))
                self.hooks.append(attn_hook)

        # Use a forward hook on each apply_layer block to do the token selection AFTER the block
        for i, block in enumerate(self.model.blocks):
            if i in self.apply_layers:
                fwd_hook = block.register_forward_hook(self._make_select_hook(i))
                self.hooks.append(fwd_hook)

    def _make_attn_hook(self, layer_idx):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            # CLS attention to patch tokens (exclude CLS itself)
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
            self._last_cls_attn[layer_idx] = cls_attn.detach()
        return hook_fn

    def _make_select_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if layer_idx not in self._last_cls_attn:
                return output
            x = output
            B, N, C = x.shape
            if N <= 2:
                return output

            cls_attn = self._last_cls_attn[layer_idx]
            num_patches = N - 1
            if cls_attn.shape[1] != num_patches:
                return output

            keep = max(1, int(num_patches * self.keep_rate))
            _, top_idx = cls_attn.topk(keep, dim=-1)
            top_idx = top_idx.sort(dim=-1).values

            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]

            # Gather kept tokens
            kept = torch.gather(patch_tokens, 1, top_idx.unsqueeze(-1).expand(-1, -1, C))

            # Create fused token from dropped tokens
            all_idx = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(B, -1)
            mask = torch.ones(B, num_patches, device=x.device, dtype=torch.bool)
            mask.scatter_(1, top_idx, False)
            num_dropped = mask.sum(dim=1, keepdim=True).clamp(min=1)
            dropped = patch_tokens * mask.unsqueeze(-1).float()
            fused = dropped.sum(dim=1, keepdim=True) / num_dropped.unsqueeze(-1).float()

            x_new = torch.cat([cls_token, kept, fused], dim=1)
            del self._last_cls_attn[layer_idx]
            return x_new
        return hook_fn

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._last_cls_attn = {}
