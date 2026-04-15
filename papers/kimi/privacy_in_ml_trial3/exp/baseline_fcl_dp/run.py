#!/usr/bin/env python3
"""Baseline: FCL + Differential Privacy"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))

from trainer import train_federated
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    args.experiment_name = 'baseline_fcl_dp_cifar10'
    args.experiment_type = 'fcl_dp'
    args.dataset = 'cifar10'
    args.num_clients = 5
    args.alpha = 0.5
    args.global_rounds = 20
    args.local_epochs = 3
    args.batch_size = 256
    args.lr = 0.001
    args.use_adversarial = False
    args.use_privacy_reg = False
    args.use_grad_noise = False
    args.max_grad_norm = 1.0
    args.noise_multiplier = 1.0  # Epsilon ~ 8 for reasonable privacy
    args.data_dir = './data'
    args.output_dir = './results'
    
    train_federated(args)

if __name__ == '__main__':
    main()
