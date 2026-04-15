"""
Download and prepare CIFAR-10-C and CIFAR-100-C datasets.
"""

import os
import sys
import urllib.request
import tarfile
import numpy as np
from tqdm import tqdm

# Zenodo URLs for CIFAR-C datasets
CIFAR10_C_URL = "https://zenodo.org/record/3555552/files/CIFAR-10-C.tar"
CIFAR100_C_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"


def download_file(url: str, dest_path: str):
    """Download a file with progress bar."""
    print(f"Downloading {url}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def extract_tar(tar_path: str, extract_path: str):
    """Extract tar file."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_path)
    print(f"Extracted to {extract_path}")


def download_cifar_c(data_root: str = 'data'):
    """Download CIFAR-10-C and CIFAR-100-C."""
    os.makedirs(data_root, exist_ok=True)
    
    # Download CIFAR-10-C
    cifar10_tar = os.path.join(data_root, 'CIFAR-10-C.tar')
    if not os.path.exists(cifar10_tar) and not os.path.exists(os.path.join(data_root, 'CIFAR-10-C')):
        try:
            download_file(CIFAR10_C_URL, cifar10_tar)
            extract_tar(cifar10_tar, data_root)
        except Exception as e:
            print(f"Error downloading CIFAR-10-C: {e}")
            print("Please download manually from https://zenodo.org/record/3555552")
    else:
        print("CIFAR-10-C already exists")
    
    # Download CIFAR-100-C
    cifar100_tar = os.path.join(data_root, 'CIFAR-100-C.tar')
    if not os.path.exists(cifar100_tar) and not os.path.exists(os.path.join(data_root, 'CIFAR-100-C')):
        try:
            download_file(CIFAR100_C_URL, cifar100_tar)
            extract_tar(cifar100_tar, data_root)
        except Exception as e:
            print(f"Error downloading CIFAR-100-C: {e}")
            print("Please download manually from https://zenodo.org/record/3555552")
    else:
        print("CIFAR-100-C already exists")
    
    print("\nDataset preparation complete!")
    
    # Verify data
    cifar10_path = os.path.join(data_root, 'CIFAR-10-C')
    if os.path.exists(cifar10_path):
        files = os.listdir(cifar10_path)
        print(f"\nCIFAR-10-C contents: {len(files)} files")
        print(f"  Corruptions: {[f for f in files if f.endswith('.npy')][:5]}...")
    
    cifar100_path = os.path.join(data_root, 'CIFAR-100-C')
    if os.path.exists(cifar100_path):
        files = os.listdir(cifar100_path)
        print(f"\nCIFAR-100-C contents: {len(files)} files")
        print(f"  Corruptions: {[f for f in files if f.endswith('.npy')][:5]}...")


if __name__ == '__main__':
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data'
    download_cifar_c(data_root)
