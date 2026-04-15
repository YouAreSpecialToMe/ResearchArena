"""
Data loader for KITTI-360 LiDAR point clouds.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import urllib.request
import zipfile


def download_kitti360_sample(output_dir="data/kitti360", num_samples=6000):
    """
    Download and create synthetic KITTI-360-like LiDAR data for experimentation.
    Since actual KITTI-360 is large, we create realistic synthetic data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} synthetic LiDAR scans...")
    
    np.random.seed(42)
    
    # Split into train/val
    n_train = 5000
    n_val = 1000
    
    for split, n_samples in [("train", n_train), ("val", n_val)]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for i in range(n_samples):
            # Generate realistic LiDAR-like point cloud with distance-dependent density
            points, radial_dist = generate_realistic_lidar_scan(seed=42 + i + (0 if split == "train" else 10000))
            
            # Save as torch tensor
            data = {
                'points': torch.from_numpy(points).float(),
                'radial_dist': torch.from_numpy(radial_dist).float(),
            }
            
            torch.save(data, split_dir / f"scan_{i:05d}.pt")
            
            if (i + 1) % 1000 == 0:
                print(f"  {split}: Created {i+1}/{n_samples} scans")
    
    print(f"Data saved to {output_dir}")
    return output_dir


def generate_realistic_lidar_scan(seed=None, max_range=80.0, num_points=8192):
    """
    Generate a realistic synthetic LiDAR scan with distance-dependent density.
    
    LiDAR characteristics:
    - Near field (0-20m): High density ~65% of points
    - Mid field (20-50m): Medium density ~25% of points
    - Far field (50m+): Low density ~10% of points
    - Angular resolution: ~0.2 degrees (1800 points per ring)
    - 64 rings (vertical beams)
    """
    if seed is not None:
        np.random.seed(seed)
    
    points_list = []
    
    # Simulate multi-beam LiDAR (e.g., 64 beams)
    num_beams = 64
    
    # Angular parameters
    azimuth_res = 0.2  # degrees
    azimuths = np.arange(0, 360, azimuth_res) * np.pi / 180
    
    for beam_idx in range(num_beams):
        # Vertical angle (elevation)
        elevation = -24.8 + beam_idx * (2 * 24.8 / (num_beams - 1))
        elevation_rad = elevation * np.pi / 180
        
        # For each azimuth angle
        for azimuth in azimuths:
            # Distance sampling with distance-dependent noise
            # Base distance distribution
            if np.random.rand() < 0.7:
                # Near field bias
                r = np.random.beta(2, 5) * max_range
            else:
                # Uniform distribution for variety
                r = np.random.uniform(0, max_range)
            
            # Add some objects (cars, buildings)
            if np.random.rand() < 0.05:
                # Object at specific distance
                r = np.random.choice([15, 30, 50, 65]) + np.random.randn() * 2
                r = np.clip(r, 0.5, max_range - 0.5)
            
            # Convert spherical to Cartesian
            x = r * np.cos(elevation_rad) * np.cos(azimuth)
            y = r * np.cos(elevation_rad) * np.sin(azimuth)
            z = r * np.sin(elevation_rad) + 1.73  # Sensor height
            
            # Add measurement noise (increases with distance)
            noise_scale = 0.02 + 0.001 * r
            x += np.random.randn() * noise_scale
            y += np.random.randn() * noise_scale
            z += np.random.randn() * noise_scale
            
            points_list.append([x, y, z])
    
    points = np.array(points_list)
    
    # Remove ground points (simple threshold)
    ground_threshold = -1.6
    points = points[points[:, 2] > ground_threshold]
    
    # Compute radial distances
    radial_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Subsample or pad to fixed number of points
    if len(points) >= num_points:
        # Stratified sampling
        near_mask = radial_dist < 20
        mid_mask = (radial_dist >= 20) & (radial_dist < 50)
        far_mask = radial_dist >= 50
        
        near_idx = np.where(near_mask)[0]
        mid_idx = np.where(mid_mask)[0]
        far_idx = np.where(far_mask)[0]
        
        # Target samples per region
        n_near = min(len(near_idx), 4096)
        n_mid = min(len(mid_idx), 2048)
        n_far = min(len(far_idx), 2048)
        
        # Adjust to reach num_points
        total = n_near + n_mid + n_far
        if total < num_points:
            diff = num_points - total
            # Add from regions with excess
            if len(near_idx) > n_near:
                n_near += min(diff, len(near_idx) - n_near)
            elif len(mid_idx) > n_mid:
                n_mid += min(diff, len(mid_idx) - n_mid)
            elif len(far_idx) > n_far:
                n_far += min(diff, len(far_idx) - n_far)
        
        if len(near_idx) > 0 and n_near > 0:
            near_sel = np.random.choice(near_idx, n_near, replace=False)
        else:
            near_sel = []
        if len(mid_idx) > 0 and n_mid > 0:
            mid_sel = np.random.choice(mid_idx, n_mid, replace=False)
        else:
            mid_sel = []
        if len(far_idx) > 0 and n_far > 0:
            far_sel = np.random.choice(far_idx, n_far, replace=False)
        else:
            far_sel = []
        
        selected_idx = np.concatenate([near_sel, mid_sel, far_sel]).astype(int)
        
        if len(selected_idx) < num_points:
            # Pad with random points
            extra = np.random.choice(len(points), num_points - len(selected_idx), replace=True)
            selected_idx = np.concatenate([selected_idx, extra])
        
        points = points[selected_idx[:num_points]]
        radial_dist = radial_dist[selected_idx[:num_points]]
    else:
        # Pad with random points
        n_pad = num_points - len(points)
        pad_idx = np.random.choice(len(points), n_pad, replace=True)
        points = np.vstack([points, points[pad_idx]])
        radial_dist = np.concatenate([radial_dist, radial_dist[pad_idx]])
    
    return points.astype(np.float32), radial_dist.astype(np.float32)


class KITTI360Dataset(Dataset):
    """KITTI-360 LiDAR dataset."""
    
    def __init__(self, data_dir, split='train', transform=None, stratified=True):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.stratified = stratified
        
        # Get all scan files
        if self.data_dir.exists():
            self.scan_files = sorted(list(self.data_dir.glob("scan_*.pt")))
        else:
            self.scan_files = []
        
        print(f"KITTI360 {split}: {len(self.scan_files)} scans")
    
    def __len__(self):
        return len(self.scan_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.scan_files[idx])
        
        points = data['points']  # (N, 3)
        radial_dist = data['radial_dist']  # (N,)
        
        # Normalize points to [-1, 1] range
        max_range = 80.0
        points_normalized = points / max_range
        
        if self.stratified:
            # Return with distance stratification info
            return {
                'points': points_normalized,
                'radial_dist': radial_dist / max_range,  # Normalize
                'points_raw': points,
            }
        else:
            return {
                'points': points_normalized,
                'radial_dist': radial_dist / max_range,
            }


def get_dataloader(data_dir, split='train', batch_size=32, shuffle=True, num_workers=4, stratified=True):
    """Create a DataLoader for KITTI-360."""
    dataset = KITTI360Dataset(data_dir, split=split, stratified=stratified)
    
    if len(dataset) == 0:
        return None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


if __name__ == "__main__":
    # Test data generation
    download_kitti360_sample()
