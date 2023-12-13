import os
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

class SoundDataset(Dataset):
    def __init__(self, dir, lenght, train=True, ratio=0.8, reduce=0, partition=1):
        self.dir = dir
        self.lenght = lenght
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
            
        random.shuffle(self.data)
        step = len(self.data)//partition
        self.data = self.data[:step*partition]
        self.number_audio = len(self.data)
        
        self.data = [self.data[k*step:(k+1)*step] for k in range(partition)]
        self.partition_lenght = (step-1)*step
        pass

    def __getitem__(self, id):
        partition = id // self.partition_lenght
        n = len(self.data[partition])
        id = id % self.partition_lenght
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
        
        if audio1.shape[0] < self.lenght:
            audio1 = torch.cat([audio1, torch.zeros(self.lenght - audio1.shape[0])])
        else:
            audio1 = audio1[:self.lenght]
        
        if audio2.shape[0] < self.lenght:
            audio2 = torch.cat([audio2, torch.zeros(self.lenght - audio2.shape[0])])
        else:
            audio2 = audio2[:self.lenght]
        
        audio = audio1 + audio2
        target = torch.stack([audio1, audio2], dim=0)
        return audio, target

    def __len__(self):
        return self.partition_lenght*len(self.data)
        