import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

# Load the digits dataset
digits = load_digits()

# Get the first instance of digit 1
digit_1_indices = digits.target == 1
digit_1_image = digits.images[digit_1_indices][0]

# Scale the image to 0-255 range (8-bit)
digit_1_image = (digit_1_image * 255 / digit_1_image.max()).astype(np.uint8)

# Create a single figure
plt.figure(figsize=(6, 6))

# Plot the digit with axis scales
plt.imshow(digit_1_image, cmap='gray', vmin=0, vmax=255)
plt.colorbar(label='Pixel intensity (8-bit)')

# Add grid and labels
plt.grid(False)
plt.xlabel('Pixel Position')
plt.ylabel('Pixel Position')
plt.title('Digit: 1')

# Show the plot
plt.tight_layout()
plt.show()
