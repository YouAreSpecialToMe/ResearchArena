import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]
REF_SEEDS = [100, 101, 102, 103]
REF_SEEDS_EXTRA = [104, 105, 106, 107]  # For K=8 ablation
BATCH_SIZE = 512
NUM_WORKERS = 0
DEFAULT_ALPHA = 1.0

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Training hyperparams
CIFAR_EPOCHS = 80
PURCHASE_EPOCHS = 80
UNLEARN_EPOCHS = 5
SCRUB_PASSES = 3

# Forget set
FORGET_SIZE = 1000
REF_POOL_SIZE = 10000

DATASETS = ['cifar10', 'cifar100', 'purchase100']
UNLEARN_METHODS = ['ft', 'ga', 'rl', 'scrub', 'neggrad']
