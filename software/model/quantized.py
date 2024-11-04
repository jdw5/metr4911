from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import back
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import namedtuple
import copy
import os
import numpy as np

# Define the MODEL constant
MODEL = 'model1'

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

def main():
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Net().to(device)
    
    if os.path.exists(f"{MODEL}.pt"):
        print(f"Loading existing model from {MODEL}.pt")
        model.load_state_dict(torch.load(f"{MODEL}.pt"))
    else:
        print("Training new model")
        optimizer = optim.Adam(model.parameters())
        for epoch in range(1, 21):  # 10 epochs
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)

        torch.save(model.state_dict(), f"{MODEL}.pt")
    
    return model, train_loader, test_loader

model, train_loader, test_loader = main()

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point
    zero_point = int(zero_point)
    return scale, zero_point

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def quantizeLayer(x, layer, stat, scale_x, zp_x):
    W = layer.weight.data
    w = quantize_tensor(layer.weight.data) 
    layer.weight.data = w.tensor.float()
    scale_w = w.scale
    zp_w = w.zero_point
    
    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])
    X = x.float() - zp_x
    layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
    
    # Handle bias if it exists
    if layer.bias is not None:
        B = layer.bias.data
        b = quantize_tensor(layer.bias.data)
        layer.bias.data = b.tensor.float()
        scale_b = b.scale
        zp_b = b.zero_point
        layer.bias.data = scale_b*(layer.bias.data + zp_b)
    
    x = (layer(X)/ scale_next) + zero_point_next 
    x = F.relu(x)
    
    # Reset weights
    layer.weight.data = W
    if layer.bias is not None:
        layer.bias.data = B
    
    return x, scale_next, zero_point_next

def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)
    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1
    return stats

def gatherActivationStats(model, x, stats):
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = model.features(x)
    x = x.view(-1, 10 * 3 * 3)
    stats = updateStats(x, stats, 'fc1')
    x = model.classifier(x)
    return stats

def gatherStats(model, test_loader):    
    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            stats = gatherActivationStats(model, data, stats)
    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats

def quantForward(model, x, stats):
    x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
    
    # Quantize the convolutional layer
    conv_layer = model.features[0]
    x, scale_next, zero_point_next = quantizeLayer(x.tensor, conv_layer, stats['fc1'], x.scale, x.zero_point)
    
    # Apply ReLU and MaxPool
    x = model.features[1:](x)
    
    x = x.view(-1, 10 * 3 * 3)
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    
    # Apply the classifier
    x = model.classifier(x)
    return x

def testQuant(model, test_loader, quant=False, stats=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
                output = quantForward(model, data, stats)
            else:
                output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

q_model = copy.deepcopy(model)

stats = gatherStats(q_model, test_loader)
print(stats)

testQuant(q_model, test_loader, quant=True, stats=stats)

def save_quantized_model(model, stats, filename):
    quantized_state_dict = {}
    for name, param in model.state_dict().items():
        if name.endswith('.weight') or name.endswith('.bias'):
            quantized_param = quantize_tensor(param)
            quantized_state_dict[name] = {
                'tensor': quantized_param.tensor,
                'scale': quantized_param.scale,
                'zero_point': quantized_param.zero_point
            }
        else:
            quantized_state_dict[name] = param
    
    torch.save({
        'model_state_dict': quantized_state_dict,
        'stats': stats
    }, filename)
    print(f"Quantized model saved to {filename}")

save_quantized_model(q_model, stats, f"q{MODEL}.pt")

# Save the original model
torch.save(model.state_dict(), f"{MODEL}.pt")