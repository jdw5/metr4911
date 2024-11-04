import os
import torch
import torch.nn as nn
import numpy as np

def get_layer_name(name, module):
    if isinstance(module, nn.Conv2d):
        return 'conv'
    elif isinstance(module, nn.Linear):
        return 'fc'
    else:
        return name.split('.')[-1]  # fallback to the last part of the name

def get_and_save_weights(model, quantized_state_dict, output_dir):
    """
    Retrieve quantized weights and biases from a PyTorch model and save them as text files.
    
    Args:
    model (torch.nn.Module): The PyTorch model structure.
    quantized_state_dict (dict): The quantized state dictionary of the model.
    output_dir (str): The directory to save the weight files in.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    layer_counters = {'conv': 1, 'fc': 1}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_type = get_layer_name(name, module)
            layer_num = layer_counters[layer_type]
            layer_counters[layer_type] += 1
            
            # Get quantized weights
            if name + '.weight' in quantized_state_dict:
                weights = quantized_state_dict[name + '.weight']['tensor'].numpy()
                filename = f"{layer_type}{layer_num}_weights.txt"
                filepath = os.path.join(output_dir, filename)
                np.savetxt(filepath, weights.flatten(), fmt='%d', delimiter='\n')
                print(f"Saved {filename}")
            
            # Get quantized biases if they exist
            if name + '.bias' in quantized_state_dict:
                biases = quantized_state_dict[name + '.bias']['tensor'].numpy()
                filename = f"{layer_type}{layer_num}_bias.txt"
                filepath = os.path.join(output_dir, filename)
                np.savetxt(filepath, biases.flatten(), fmt='%d', delimiter='\n')
                print(f"Saved {filename}")

def load_quantized_model(model_class, model_path):
    """
    Load a quantized model and return the model structure and quantized state dict.
    
    Args:
    model_class (type): The class of the model to be loaded.
    model_path (str): Path to the saved quantized model.
    
    Returns:
    model (torch.nn.Module): The model structure.
    quantized_state_dict (dict): The quantized state dictionary.
    """
    quantized_model_data = torch.load(model_path)
    model = model_class()
    quantized_state_dict = quantized_model_data['model_state_dict']
    
    return model, quantized_state_dict

# Example usage:
if __name__ == "__main__":
    from quantized import Net, MODEL  # Import your model class and MODEL constant
    
    # Load the quantized model
    model, quantized_state_dict = load_quantized_model(Net, f'q{MODEL}.pt')
    
    # Get and save weights for the quantized model
    get_and_save_weights(model, quantized_state_dict, 'dir')
