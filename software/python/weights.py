import torch
import csv

# Load the model state dict
state_dict = torch.load("models/mnist.pth")

# Function to save tensor to CSV, one value per line
def save_to_csv(tensor, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in tensor.view(-1):
            writer.writerow([value.item()])

# Extract and save conv layer weights
conv_weights = state_dict['conv_block.0.weight']
save_to_csv(conv_weights, 'conv_weights.csv')

# Extract and save fully connected layer weights
fc_weights = state_dict['linear_block.0.weight']
save_to_csv(fc_weights, 'fc_weights.csv')

# Extract and save fully connected layer biases
fc_biases = state_dict['linear_block.0.bias']
save_to_csv(fc_biases, 'fc_biases.csv')

print("Weights and biases have been saved to CSV files.")
