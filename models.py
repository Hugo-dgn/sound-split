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

class TmeDomainLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, x1, x2, target):
        
        arrange11 = x1 - target[:,0,:]
        arrange12 = x1 - target[:,1,:]
        
        arrange21 = x2 - target[:,0,:]
        arrange22 = x2 - target[:,1,:]
        
        loss1 = torch.mean(arrange11**2, dim=1) + torch.mean(arrange22**2, dim=1)
        loss2 = torch.mean(arrange12**2, dim=1) + torch.mean(arrange21**2, dim=1)
        loss = torch.min(torch.stack([loss1, loss2], dim=1), dim=1).values
        loss = torch.mean(loss)
        
        return loss

class FreqDomainLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, x1, x2, target):
        transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=None).to(x1.device)
        
        spectro1 = transform(x1)
        spectro2 = transform(x2)
        spectro_target = transform(target)
        
        power1 = torch.abs(spectro1)
        power2 = torch.abs(spectro2)
        power_target = torch.abs(spectro_target)
        
        arrange11 = power1 - power_target[:,0,:,:]
        arrange12 = power1 - power_target[:,1,:,:]
        
        arrange21 = power2 - power_target[:,0,:,:]
        arrange22 = power2 - power_target[:,1,:,:]
        
        loss1 = torch.mean(arrange11**2, dim=(1,2)) + torch.mean(arrange22**2, dim=(1,2))
        loss2 = torch.mean(arrange12**2, dim=(1,2)) + torch.mean(arrange21**2, dim=(1,2))
        loss = torch.min(torch.stack([loss1, loss2], dim=1), dim=1).values
        loss = torch.mean(loss)
        
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

class conv1D_block(nn.Module):
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

class encoder1D_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.conv = conv1D_block(in_c, out_c, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        y = self.conv(x)
        p = self.pool(y)
        return y, p

class decoder1D_block(nn.Module):
    def __init__(self, in_c, out_c, in_skip, kernel_size):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv1D_block(out_c+in_skip, out_c, kernel_size)
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
        
        self.e1 = encoder1D_block(1, 16, kernel_size=25)
        self.e2 = encoder1D_block(16, 32, kernel_size=13)
        self.e3 = encoder1D_block(32, 64, kernel_size=7)
        self.e4 = encoder1D_block(64, 64, kernel_size=5)
        self.b = conv1D_block(64, 128, kernel_size=3)
        self.d1 = decoder1D_block(128, 64, 64, kernel_size=3)
        self.d2 = decoder1D_block(64, 32, 64, kernel_size=3)
        self.d3 = decoder1D_block(32, 16, 32, kernel_size=3)
        self.d4 = decoder1D_block(16, 8, 16, kernel_size=3)
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

####################################################################################################

class Network3(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 3, gen, checkpoint)
        
        self.e1 = encoder1D_block(1, 16, kernel_size=25)
        self.e2 = encoder1D_block(16, 32, kernel_size=13)
        self.e3 = encoder1D_block(32, 64, kernel_size=7)
        self.e4 = encoder1D_block(64, 64, kernel_size=5)
        self.b = conv1D_block(64, 128, kernel_size=3)
        self.d1 = decoder1D_block(128, 64, 64, kernel_size=3)
        self.d2 = decoder1D_block(64, 32, 64, kernel_size=17)
        self.d3 = decoder1D_block(32, 16, 32, kernel_size=25)
        self.d4 = decoder1D_block(16, 8, 16, kernel_size=33)
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

####################################################################################################

class Network4(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 4, gen, checkpoint)
        
        self.sparator = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(11, 11), padding="same"),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(11, 11), padding="same"),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(11, 11), padding="same"),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(11, 11), padding="same"),
            nn.Sigmoid()
        )
        
        self.load()
        
    def forward(self, inputs):
        transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=None).to(inputs.device)
        inverse_transform = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256).to(inputs.device)
        
        inputs = inputs.unsqueeze(1)
        spectro = transform(inputs)
        
        power = torch.abs(spectro)
        phase = torch.angle(spectro)
        
        mask = self.sparator(power)
        power = power * mask
        power1 = power[:,0,:,:].unsqueeze(1)
        power2 = power[:,1,:,:].unsqueeze(1)
        
        spectro1 = power1 * torch.exp(1j * phase)
        spectro2 = power2 * torch.exp(1j * phase)
        
        signal1 = inverse_transform(spectro1).squeeze(1)
        signal2 = inverse_transform(spectro2).squeeze(1)
        
        return signal1, signal2

####################################################################################################

class Network5(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 5, gen, checkpoint)
        
        self.separator = nn.RNN(input_size=257, hidden_size=257, num_layers=2, batch_first=True)
        self.activation = nn.Sigmoid()
        
        self.load()
    
    def forward(self, inputs):
        transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=None).to(inputs.device)
        inverse_transform = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256).to(inputs.device)
        
        spectro = transform(inputs)
        
        power = torch.abs(spectro)
        phase = torch.angle(spectro)
        
        mask = self.separator(torch.transpose(power, 1, 2))[0]
        mask = torch.transpose(mask, 1, 2)
        mask = self.activation(mask)
        
        power1 = power * mask
        power2 = power - power1
        
        spectro1 = power1 * torch.exp(1j * phase)
        spectro2 = power2 * torch.exp(1j * phase)
        
        signal1 = inverse_transform(spectro1)
        signal2 = inverse_transform(spectro2)
        
        return signal1, signal2