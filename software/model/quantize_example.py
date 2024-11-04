import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

from quantize import (
    gatherStats, testQuant, save_quantized_model, load_quantized_model,
    quantize_tensor, dequantize_tensor
)

# Define the MODEL constant
MODEL = 'digit_model'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=2, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(10 * 3 * 3, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 10 * 3 * 3)
        x = self.classifier(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def prepare_data():
    # Load the digits dataset
    digits = load_digits()
    X, y = digits.images, digits.target

    # Reshape the data for CNN input (samples, channels, height, width)
    X = X.reshape(X.shape[0], 1, 8, 8)
    X = X.astype('float32') / 16.0  # Normalize pixel values to [0, 1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).long()

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Net().to(device)
    train_loader, test_loader = prepare_data()
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU...")
    
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, 11):  # 10 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save the original model
    torch.save(model.state_dict(), f"{MODEL}.pt")
    print(f"Original model saved to {MODEL}.pt")

    # Gather statistics for quantization
    stats = gatherStats(model, train_loader)
    print("Activation stats:", stats)

    # Test quantized model
    criterion = nn.NLLLoss()
    quant_loss, quant_accuracy = testQuant(model, test_loader, criterion, quant=True, stats=stats)
    print(f"Quantized model - Test loss: {quant_loss:.4f}, Accuracy: {quant_accuracy:.2f}%")

    # Save quantized model
    save_quantized_model(model, stats, f"q{MODEL}.pt")

    # Load and test quantized model
    loaded_model, loaded_stats = load_quantized_model(Net().to(device), f"q{MODEL}.pt")
    loaded_loss, loaded_accuracy = testQuant(loaded_model, test_loader, criterion, quant=True, stats=loaded_stats)
    print(f"Loaded quantized model - Test loss: {loaded_loss:.4f}, Accuracy: {loaded_accuracy:.2f}%")

if __name__ == "__main__":
    main()

