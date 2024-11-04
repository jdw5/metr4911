import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

INDEX = 1

def process_digit_image(index=INDEX, output_file='digit_image.txt'):
    # Load the digits dataset
    digits = load_digits()
    
    # Get the specified image
    image = digits.images[index]
    
    # Scale the image to 0-255 range and convert to integers
    image_scaled = np.round(image * 255 / image.max()).astype(int)
    
    # Display the image
    plt.imshow(image_scaled, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Digit: {digits.target[index]}")
    plt.axis('off')
    plt.show()
    
    # Flatten the 8x8 image into a 1D array
    flattened_image = image_scaled.flatten()
    
    # Save the pixel values to a text file
    with open(output_file, 'w') as f:
        for pixel_value in flattened_image:
            f.write(f"{pixel_value}\n")
    
    print(f"Image saved to {output_file}")

# Example usage
process_digit_image(index=INDEX, output_file=f"digit_{INDEX}.txt")
