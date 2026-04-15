#!/usr/bin/env python3
"""Baseline: Standard FCL (FedSimCLR)"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))

from trainer import train_federated
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    args = parser.parse_args()
    
    # Set all arguments
    args.experiment_name = f'baseline_fcl_{args.dataset}'
    args.experiment_type = 'standard'
    args.num_clients = 5
    args.alpha = 0.5
    args.global_rounds = 20 if args.dataset == 'cifar10' else 15
    args.local_epochs = 3
    args.batch_size = 256
    args.lr = 0.001
    args.use_adversarial = False
    args.use_privacy_reg = False
    args.use_grad_noise = False
    args.data_dir = './data'
    args.output_dir = './results'
    
    train_federated(args)

if __name__ == '__main__':
    main()
