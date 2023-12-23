import os

import torch
import torch.nn as nn

def get_checkpoint(network_id, gen, checkpoint):
    network_path = os.path.join("save", str(network_id))
    if gen == -1:
        gen = max([int(gen) for gen in os.listdir(network_path)])
    path = os.path.join(network_path, str(gen))
    save_files = [int(file.split(".")[0]) for file in os.listdir(path)]
    if len(save_files) == 0:
        start_checkpoint = None
    else:
        if checkpoint == -1:
            start_checkpoint = max(save_files)
        else:
            if checkpoint in save_files:
                start_checkpoint = checkpoint
            else:
                message = f"Checkpoint {checkpoint} not found."
                raise AssertionError(message)
    return start_checkpoint

def check(network_id, gen):
    path_network = os.path.join("save", str(network_id))
    if "save" not in os.listdir():
        os.mkdir("save")
    if str(network_id) not in os.listdir("save"):
        os.mkdir(path_network)
        
    if gen == -1:
        if len(os.listdir(path_network)) == 0:
            gen = 0
        else:
            gen = max([int(gen.split(".")[0]) for gen in os.listdir(path_network)])
    if str(gen) not in os.listdir(path_network):
        os.mkdir(os.path.join(path_network, str(gen)))
    
    return gen

class conv1D_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, num_conv, activation):
        super().__init__()
        
        conv = []
        for _ in range(num_conv):
            conv.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding="same"))
            conv.append(nn.BatchNorm1d(out_c))
            conv.append(activation())
            in_c = out_c
        self.conv = nn.Sequential(*conv)
        
    def forward(self, x):
        y = self.conv(x)
        return y

class encoder1D_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, num_conv=2, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.Mish
        self.conv = conv1D_block(in_c, out_c, kernel_size, num_conv, activation)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        y = self.conv(x)
        p = self.pool(y)
        return y, p

class decoder1D_block(nn.Module):
    def __init__(self, in_c, out_c, in_skip, kernel_size, num_conv=2, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.Mish
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv1D_block(out_c+in_skip, out_c, kernel_size, num_conv, activation)
    def forward(self, inputs, skip=None, length=None):
        x = self.up(inputs)
        if skip is not None:
            min_size = min(x.shape[2], skip.shape[2])
            max_size = max(x.shape[2], skip.shape[2])
            delta = max_size - min_size
            if delta > 0:
                if x.shape[2] == min_size:
                    x = nn.functional.pad(x, (0, delta))
                else:
                    skip = nn.functional.pad(skip, (0, delta))
            x = torch.cat([x, skip], axis=1)
        else:
            if length is not None:
                if x.shape[2] > length:
                    x = x[:,:,:length]
                elif x.shape[2] < length:
                    x = nn.functional.pad(x, (0, length-x.shape[2]))
        x = self.conv(x)
        return x