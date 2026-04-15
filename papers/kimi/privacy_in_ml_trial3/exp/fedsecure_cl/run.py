#!/usr/bin/env python3
"""FedSecure-CL: Full Framework"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))

from trainer import train_federated
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--ablation', type=str, default='none', 
                       choices=['none', 'no_privacy', 'no_grad_noise', 'no_adv'])
    args = parser.parse_args()
    
    dataset = args.dataset
    seed = args.seed
    ablation = args.ablation
    
    if ablation == 'none':
        args.experiment_name = f'fedsecure_cl_{dataset}'
        args.use_adversarial = True
        args.use_privacy_reg = True
        args.use_grad_noise = True
        args.alpha_at = 1.0
        args.beta_privacy = 0.5
    elif ablation == 'no_privacy':
        args.experiment_name = f'ablation_no_privacy_{dataset}'
        args.use_adversarial = True
        args.use_privacy_reg = False
        args.use_grad_noise = True
        args.alpha_at = 1.0
        args.beta_privacy = 0.0
    elif ablation == 'no_grad_noise':
        args.experiment_name = f'ablation_no_grad_noise_{dataset}'
        args.use_adversarial = True
        args.use_privacy_reg = True
        args.use_grad_noise = False
        args.alpha_at = 1.0
        args.beta_privacy = 0.5
    elif ablation == 'no_adv':
        args.experiment_name = f'ablation_no_adv_{dataset}'
        args.use_adversarial = False
        args.use_privacy_reg = True
        args.use_grad_noise = True
        args.alpha_at = 0.0
        args.beta_privacy = 0.5
    
    args.experiment_type = 'fedsecure'
    args.dataset = dataset
    args.num_clients = 5
    args.alpha = 0.5
    args.global_rounds = 20 if dataset == 'cifar10' else 15
    args.local_epochs = 3
    args.batch_size = 256
    args.lr = 0.001
    args.attack_eps = 8/255
    args.attack_steps = 7
    args.data_dir = './data'
    args.output_dir = './results'
    
    train_federated(args)

if __name__ == '__main__':
    main()
