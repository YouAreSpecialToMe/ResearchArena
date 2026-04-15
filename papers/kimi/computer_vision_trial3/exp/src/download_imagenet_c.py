"""
Script to download ImageNet-C from various sources.
"""
import os
import sys
import urllib.request
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a URL to a local file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_imagenet_c_zenodo(data_dir):
    """Download ImageNet-C from Zenodo."""
    url = "https://zenodo.org/records/2235448/files/imagenet-c.tar"
    output_path = os.path.join(data_dir, "imagenet-c.tar")
    
    print(f"Downloading ImageNet-C from Zenodo...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    # Use wget with resume capability
    os.system(f"wget --continue --tries=0 --timeout=60 -O {output_path} '{url}'")
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000000:  # > 1GB
        print(f"Download successful: {os.path.getsize(output_path) / 1e9:.2f} GB")
        return True
    else:
        print("Download failed or incomplete")
        return False


def extract_imagenet_c(data_dir):
    """Extract ImageNet-C tar file."""
    tar_path = os.path.join(data_dir, "imagenet-c.tar")
    
    if not os.path.exists(tar_path):
        print(f"Tar file not found: {tar_path}")
        return False
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(data_dir)
    
    print("Extraction complete")
    return True


if __name__ == "__main__":
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Try to download
    if download_imagenet_c_zenodo(data_dir):
        extract_imagenet_c(data_dir)
    else:
        print("Failed to download ImageNet-C")
        print("Please download manually from: https://zenodo.org/record/2235448")
        sys.exit(1)
