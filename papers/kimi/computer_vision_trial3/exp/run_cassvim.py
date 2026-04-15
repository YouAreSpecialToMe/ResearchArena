"""Quick CASS-ViM experiments."""
import os, sys, json, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as transforms
sys.path.insert(0, '.')
from src.minimal_models import MinimalCASSViM

def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def train(model, trainloader, testloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    best_acc = 0
    test_accs = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100. * correct / total
        test_accs.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f'  Epoch {epoch}: Test={test_acc:.1f}%, Best={best_acc:.1f}%')
    return best_acc, test_accs

def run(model_name, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'\nModel: {model_name}, Seed: {seed}')
    
    if model_name == 'cassvim_4d':
        model = MinimalCASSViM(num_directions=4, selector_type='gradient', topks=[1,1,1,1])
    elif model_name == 'cassvim_8d':
        model = MinimalCASSViM(num_directions=8, selector_type='gradient', topks=[1,1,1,1])
    elif model_name == 'random_selection':
        model = MinimalCASSViM(num_directions=4, selector_type='random', topks=[1,1,1,1])
    elif model_name == 'fixed_perlayer':
        model = MinimalCASSViM(num_directions=4, selector_type='fixed', topks=[1,1,1,1])
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.2f}M')
    
    trainloader, testloader = get_data()
    start = time.time()
    best_acc, test_accs = train(model, trainloader, testloader, epochs, device)
    elapsed = time.time() - start
    
    result = {'model': model_name, 'seed': seed, 'best_acc': best_acc, 
              'final_acc': test_accs[-1], 'train_time': elapsed/60, 
              'n_params': n_params, 'test_accs': test_accs}
    with open(f'./results/{model_name}_seed{seed}.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Result: {best_acc:.2f}%, Time: {elapsed/60:.1f}min')
    return result

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    epochs = 30
    seeds = [42, 123, 456]
    models = ['cassvim_4d', 'cassvim_8d', 'random_selection', 'fixed_perlayer']
    for model_name in models:
        for seed in seeds:
            run(model_name, seed, epochs, device)
    print('\nAll CASS-ViM experiments completed!')

if __name__ == '__main__':
    main()
