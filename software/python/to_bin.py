import sys
import os

def int_to_binary(num):
    return format(int(num), '08b')

def convert_file_to_binary(input_path, output_path):
    try:
        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            for line in input_file:
                num = line.strip()
                if num:
                    binary = int_to_binary(num)
                    output_file.write(binary + '\n')
        print(f"Conversion complete. Binary output saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except ValueError:
        print(f"Error: Invalid integer found in the input file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python to_bin.py <input_file_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = os.path.splitext(input_path)[0] + "_binary.txt"

    convert_file_to_binary(input_path, output_path)

