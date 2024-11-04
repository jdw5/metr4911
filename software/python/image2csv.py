import sys
from PIL import Image
import csv

def image_to_csv(image_path, output_path):
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Get image dimensions
    width, height = img_gray.size
    
    # Create a list to store pixel values
    pixel_values = []
    
    # Iterate through each pixel and append its value to the list
    for y in range(height):
        for x in range(width):
            pixel_values.append(img_gray.getpixel((x, y)))
    
    # Write pixel values to CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for value in pixel_values:
            writer.writerow([value])

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python image2csv.py <input_image_path>")
    #     sys.exit(1)
    
    # input_image_path = sys.argv[1]
    # output_csv_path = sys.argv[2]
    
    image_to_csv("./dataset/train/0/1.png", "1.csv")
    print(f"Conversion complete. CSV file saved to: 1.csv")
