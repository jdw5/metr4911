import os

def generate_coe_file(data_file, output_file):
    with open(data_file, 'r') as file:
        data = [line.strip() for line in file]

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as coe:
        coe.write('memory_initialization_radix=10;\n')
        coe.write('memory_initialization_vector=\n')
        coe.write(',\n'.join(data))
        coe.write(';\n')

# Example usage
generate_coe_file('digit_0.txt', 'coe/image.coe')
