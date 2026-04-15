"""Fast GPU-based data loading for CIFAR datasets.

Loads entire dataset into GPU memory and applies augmentation on GPU,
bypassing the CPU data loading bottleneck.
"""

import torch
import torch.nn.functional as F
import torchvision
import kornia.augmentation as K


CIFAR100_SUPERCLASS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 17, 5, 0,
    2, 4, 6, 17, 10, 16, 12, 1, 9, 19,
    8, 8, 15, 13, 16, 15, 7, 12, 2, 4,
    6, 0, 10, 2, 14, 1, 10, 16, 19, 3,
    8, 4, 12, 5, 18, 19, 2, 17, 8, 2,
    1, 9, 19, 13, 17, 15, 5, 5, 1, 13,
    12, 9, 0, 19, 4, 13, 16, 18, 6, 9,
]


class GPUDataset:
    """Dataset stored entirely in GPU memory with GPU augmentation."""

    def __init__(self, dataset_name='cifar100', train=True, device='cuda',
                 data_root='./data'):
        if dataset_name == 'cifar100':
            ds = torchvision.datasets.CIFAR100(
                root=data_root, train=train, download=True)
            self.mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)
            self.num_classes = 100
        elif dataset_name == 'cifar10':
            ds = torchvision.datasets.CIFAR10(
                root=data_root, train=train, download=True)
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
            self.num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Load entire dataset to GPU as float32 [0, 1]
        images = torch.tensor(ds.data, dtype=torch.float32, device=device)
        self.images = images.permute(0, 3, 1, 2) / 255.0  # [N, C, H, W]
        self.labels = torch.tensor(ds.targets, dtype=torch.long, device=device)
        self.n = len(self.labels)
        self.device = device
        self.train = train

        # GPU augmentation pipeline
        if train:
            self.augment = K.AugmentationSequential(
                K.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                K.RandomGrayscale(p=0.2),
                data_keys=["input"],
            )
        else:
            self.augment = None

    def get_batches(self, batch_size, two_crop=True, shuffle=True):
        """Yield (images, labels) batches with GPU augmentation."""
        if shuffle:
            perm = torch.randperm(self.n, device=self.device)
            images = self.images[perm]
            labels = self.labels[perm]
        else:
            images = self.images
            labels = self.labels

        n_batches = (self.n + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, self.n)
            if end - start < batch_size // 2:
                continue  # drop small last batch
            batch_img = images[start:end]
            batch_lab = labels[start:end]

            if self.train and self.augment is not None:
                if two_crop:
                    view1 = self.augment(batch_img)
                    view2 = self.augment(batch_img)
                    view1 = (view1 - self.mean) / self.std
                    view2 = (view2 - self.mean) / self.std
                    yield [view1, view2], batch_lab
                else:
                    aug = self.augment(batch_img)
                    aug = (aug - self.mean) / self.std
                    yield aug, batch_lab
            else:
                normed = (batch_img - self.mean) / self.std
                yield normed, batch_lab

    @property
    def n_batches(self):
        return self.n // 512  # approximate
