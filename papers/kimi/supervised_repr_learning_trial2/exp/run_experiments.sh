#!/bin/bash
cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method supcon --epochs 15 --seed 42 --num_workers 2 --save_dir results
python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method supcon_lr --epochs 15 --seed 42 --num_workers 2 --save_dir results
python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method laser_scl --epochs 15 --seed 42 --num_workers 2 --save_dir results

python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method supcon --epochs 15 --seed 123 --num_workers 2 --save_dir results
python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method supcon_lr --epochs 15 --seed 123 --num_workers 2 --save_dir results
python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method laser_scl --epochs 15 --seed 123 --num_workers 2 --save_dir results

python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method supcon --epochs 15 --seed 456 --num_workers 2 --save_dir results
python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method supcon_lr --epochs 15 --seed 456 --num_workers 2 --save_dir results
python exp/shared/train.py --dataset cifar100 --noise_rate 0.4 --method laser_scl --epochs 15 --seed 456 --num_workers 2 --save_dir results
