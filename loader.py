import os
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

def get_relevent(audio, length):
    odd_length = length + 1 - length % 2
    audio_length = audio.shape[0]
    with torch.no_grad():
        stride = audio_length // odd_length * 3
        window = torch.ones(odd_length)
        indices = torch.conv1d(torch.abs(audio.unsqueeze(0).unsqueeze(0))**2, window.unsqueeze(0).unsqueeze(0), stride=stride).squeeze()
        index = stride*torch.argmax(indices)
    return audio[index:index+length]

class SoundDataset(Dataset):
    def __init__(self, dir, length, train=True, ratio=0.9, reduce=0, partition=1):
        self.dir = dir
        self.length = length
        self.partition = partition
        self.data = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".flac"):
                    self.data.append(os.path.join(root, file))
        
        n = len(self.data)
        n_target = int(n*ratio)
        if train:
            self.data = self.data[:n_target]
            if reduce > 0:
                n = len(self.data)
                n_target = int(n*(1-reduce))
                self.data = random.sample(self.data, n_target)
        else:
            self.data = self.data[n_target:]
        
        step = len(self.data)//partition
        self.data = self.data[:step*partition]
        self.number_audio = len(self.data)
        
        self.data = [self.data[k*step:(k+1)*step] for k in range(partition)]
        self.partition_length = (step-1)*step

    def __getitem__(self, id):
        partition = id // self.partition_length
        n = len(self.data[partition])
        id = id % self.partition_length
        id1 = id // (n-1)
        id2 = id % (n-1)
        if id2 >= id1:
            id2 += 1
        path_audio1 = self.data[partition][id1]
        path_audio2 = self.data[partition][id2]
        audio1, sample_rate_1 = sf.read(path_audio1)
        audio2, sample_rate_2 = sf.read(path_audio2)
        
        audio1 = torch.from_numpy(audio1)
        audio2 = torch.from_numpy(audio2)
        
        audio1 = audio1.type(torch.float32)
        audio2 = audio2.type(torch.float32)
        
        if audio1.shape[0] < self.length:
            delta = self.length - audio1.shape[0]
            audio1 = torch.nn.functional.pad(audio1, (0, delta))
        elif audio1.shape[0] > self.length:
            audio1 = get_relevent(audio1, self.length)
        
        if audio2.shape[0] < self.length:
            delta = self.length - audio2.shape[0]
            audio2 = torch.nn.functional.pad(audio2, (0, delta))
        elif audio2.shape[0] > self.length:
            audio2 = get_relevent(audio2, self.length)
        
        audio = audio1 + audio2
        target = torch.stack([audio1, audio2], dim=0)
        return audio, target

    def __len__(self):
        return self.partition_length*len(self.data)
        