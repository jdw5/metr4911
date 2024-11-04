import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import platform
import os
import numpy as np
from sklearn.datasets import load_digits

# Constant to specify which image from the digits dataset to use
INDEX = 0  # You can change this to any valid index (0 to len(digits.images) - 1)

def load_model(model_path, device):
    """Load the model and move it to the specified device (CPU or CUDA)"""
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode (important for inference)
    return model

def preprocess_image(image, device):
    """Preprocess the 8x8 grayscale image and prepare it for model input"""
    transform = transforms.Compose([
        transforms.ToTensor(),           # Convert image to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
    ])
    
    image = Image.fromarray(image)  # Convert numpy array to PIL Image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

def get_device_info():
    """Retrieve and return CPU or GPU (CUDA) device information"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_info = f"Device: GPU ({device_name})\nTotal GPU Memory: {total_memory:.2f} GB"
    else:
        cpu_name = platform.processor()
        num_cores = os.cpu_count()
        device_info = f"Device: CPU\nProcessor: {cpu_name}\nNumber of Cores: {num_cores}"
    
    return device_info

def classify_image(model, image_tensor, labels):
    """Perform inference on the image tensor and return the predicted label"""
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)  # Get the index of the highest score
        predicted_label = labels[predicted_class.item()]
    return predicted_label

if __name__ == "__main__":
    model_path = 'model.pth'  # Path to your saved model

    # Load the digits dataset
    digits = load_digits()
    images = digits.images
    labels = digits.target_names

    # Use the specified image based on the INDEX constant
    image = images[INDEX]
    true_label = digits.target[INDEX]  # True label for reference

    # Determine device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(model_path, device)

    # Preprocess the image
    image_tensor = preprocess_image(image, device)

    # Get device info
    device_info = get_device_info()

    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None  # Sync GPU
    start_time = time.time()

    # Perform classification
    prediction = classify_image(model, image_tensor, labels)

    torch.cuda.synchronize() if torch.cuda.is_available() else None  # Sync GPU
    end_time = time.time()
    
    inference_time = end_time - start_time

    # Print results
    print(f"True label: {true_label}")
    print(f"Classification result (prediction): {prediction}")
    print(f"Inference time: {inference_time:.6f} seconds")
    print(device_info)
