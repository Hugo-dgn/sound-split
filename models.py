import os

import torch
import torchaudio
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
            gen = max([int(gen) for gen in os.listdir(path_network)])
    if str(gen) not in os.listdir(path_network):
        os.mkdir(os.path.join(path_network, str(gen)))
    
    return gen

def get_network(n):
    if f"Network{n}" in globals():
        return globals()[f"Network{n}"]
    else:
        message = f"Topology Network{n} is not defined but was requested."
        raise AssertionError(message)

####################################################################################################

class Network(nn.Module):
    
    def __init__(self, network_id, gen, checkpoint):
        nn.Module.__init__(self)
        self.network_id = network_id
        self.gen = gen
        self.checkpoint = checkpoint
    
    def save(self):
        checkpoint = get_checkpoint(self.network_id, self.gen, -1)
        if checkpoint is None:
            checkpoint = 0
        else:
            checkpoint += 1
        path_checkpoint = os.path.join("save", str(self.network_id), str(self.gen), str(checkpoint) + ".pth")
        torch.save(
            self.state_dict(), 
            path_checkpoint)
    
    def load(self):
        self.gen = check(self.network_id, self.gen)
        start_checkpoint = get_checkpoint(self.network_id, self.gen, self.checkpoint)
        self.checkpoint = start_checkpoint
        if start_checkpoint is not None:
            path_start_checkpoint = os.path.join("save", str(self.network_id), str(self.gen), str(start_checkpoint) + ".pth")
            checkpoint = torch.load(path_start_checkpoint, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint)

####################################################################################################

class SoundLoss(nn.Module):
    def __init__(self, device, sample_rate):
        nn.Module.__init__(self)
        self.sample_rate = sample_rate
        self.to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=512, hop_length=256, n_mels=32).to(device)
        self.relu = nn.ReLU()
        self.db_threshold = 20
    
    def forward(self, x1, x2, target):
        
        arrange11 = x1 - target[:,0,:]
        arrange12 = x1 - target[:,1,:]
        
        arrange21 = x2 - target[:,0,:]
        arrange22 = x2 - target[:,1,:]
        
        loss1 = torch.mean(arrange11**2, dim=1) + torch.mean(arrange22**2, dim=1)
        loss1 += 2/self.db_threshold*(torch.mean(self.relu(self.to_db(arrange11) + self.db_threshold)**2, dim=1) + torch.mean(self.relu(self.to_db(arrange22) + self.db_threshold)**2, dim=1))
        loss2 = torch.mean(arrange12**2, dim=1) + torch.mean(arrange21**2, dim=1)
        loss2 += 2/self.db_threshold*(torch.mean(self.relu(self.to_db(arrange12) + self.db_threshold)**2, dim=1) + torch.mean(self.relu(self.to_db(arrange21) + self.db_threshold)**2, dim=1))
        
        loss = torch.stack([loss1, loss2], dim=1)
        loss, indices = torch.min(loss, dim=1)        
        timedomaineloss = torch.mean(loss)
        
        loss1 = torch.mean(self.MelSpectrogram(arrange11).reshape(arrange11.shape[0], -1)**2, dim=1) + torch.mean(self.MelSpectrogram(arrange22).reshape(arrange22.shape[0], -1)**2, dim=1)
        loss2 = torch.mean(self.MelSpectrogram(arrange12).reshape(arrange12.shape[0], -1)**2, dim=1) + torch.mean(self.MelSpectrogram(arrange21).reshape(arrange21.shape[0], -1)**2, dim=1)
        
        frequencydomainloss = 0.0001*torch.mean(torch.stack([loss1, loss2], dim=1).gather(1, indices.unsqueeze(1)).squeeze(1))
        
        loss = timedomaineloss + frequencydomainloss
        
        return loss

####################################################################################################

class Network1(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 1, gen, checkpoint)
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.PReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=31, stride=1, padding="same"),
        )
        
        self.load()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.encoder(x)
        x1 = self.decoder(x1)
        x2 = x - x1
        
        return x1.squeeze(1), x2.squeeze(1)
        
####################################################################################################

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_c),
            nn.Mish(),
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_c),
            nn.Mish()
        )
    def forward(self, x):
        y = self.network(x)
        return y

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.conv = conv_block(in_c, out_c, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        y = self.conv(x)
        p = self.pool(y)
        return y, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, in_skip, kernel_size):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+in_skip, out_c, kernel_size)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        min_size = min(x.shape[2], skip.shape[2])
        max_size = max(x.shape[2], skip.shape[2])
        delta = max_size - min_size
        if delta > 0:
            if x.shape[2] == min_size:
                x = nn.functional.pad(x, (0, delta))
            else:
                skip = nn.functional.pad(skip, (0, delta))
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

####################################################################################################

class Network2(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 2, gen, checkpoint)
        
        self.e1 = encoder_block(1, 16, kernel_size=25)
        self.e2 = encoder_block(16, 32, kernel_size=13)
        self.e3 = encoder_block(32, 64, kernel_size=7)
        self.e4 = encoder_block(64, 64, kernel_size=5)
        self.b = conv_block(64, 128, kernel_size=3)
        self.d1 = decoder_block(128, 64, 64, kernel_size=3)
        self.d2 = decoder_block(64, 32, 64, kernel_size=3)
        self.d3 = decoder_block(32, 16, 32, kernel_size=3)
        self.d4 = decoder_block(16, 8, 16, kernel_size=3)
        self.outputs = nn.Conv1d(8, 1, kernel_size=1, padding=0)
        
        self.load()
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        
        y1 = outputs
        y2 = inputs - outputs
        return y1.squeeze(1), y2.squeeze(1)