# verify_convolution_sim.py

import torch
from torch import conv2d
import math
import time
import statistics

def verify_convolution_sim(input_file_path, kernel_file_path, output_file_path, device='cpu'):
    device = torch.device(device)

    input  = open(input_file_path, 'r')
    kernel = open(kernel_file_path, 'r')
    output = open(output_file_path, 'r')

    input_data = torch.tensor([int(val) for val in input.readlines()], device=device)
    kernel_data = torch.tensor([int(val) for val in kernel.readlines()], device=device)
    output_data = torch.tensor([int(val) for val in output.readlines()], device=device)

    I = int(input_data[0])
    C = int(input_data[2])
    K = int(input_data[1])
    Q = int(input_data[3])
    O = int(input_data[4])
    S = int(input_data[5])
    Z = int(input_data[6])

    feature_size = int((I + 2 * Z - K) / S + 1)
    conv_batches = int(input_data.size()[0] / (C * I**2))

    # Initialize multi-dimensional arrays
    input_array = torch.zeros(conv_batches, 1,            C,  I,  I, device=device)
    kernel_array = torch.zeros(conv_batches, O, C,  K, K, device=device)
    output_array = torch.zeros(conv_batches, O, feature_size, feature_size, device=device)

    idx_i = 8
    idx_k = 0
    idx_o = 0

    for batch in range(conv_batches):
        for row in range(I):
            for col in range(I):
                for chn in range(C):
                    input_array[batch, 0, chn, row, col] = input_data[idx_i]
                    idx_i += 1
        for row in range(K):
            for col in range(K):
                for chn_o in range(O):
                    for chn_i in range(C):
                        kernel_array[batch, chn_o, chn_i, row, col] = kernel_data[idx_k]
                        idx_k += 1
        for row in range(feature_size):
            for col in range(feature_size):
                for chn in range(O):
                    output_array[batch, chn, row, col] = output_data[idx_o]
                    idx_o += 1

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    conv2d(input_array[0], kernel_array[0], padding=Z, stride=S)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize() if device.type == 'cuda' else None

    conv_time = start.elapsed_time(end)

    print('----------------------------------------')
    print(f'Convolution operation time: {conv_time} nanoseconds')
    print(f'Convolution operation time: {conv_time / 1e6:.6f} milliseconds')
    print(f'Convolution operation time: {conv_time / 1e9:.9f} seconds')
    print('----------------------------------------')

    input.close()
    kernel.close()
    output.close()

    return conv_time

verify_convolution_sim('input_data.txt', 'kernel_data.txt', 'output_data.txt', device='cpu')