import os

import torch
import torch.nn as nn
import torchaudio

from models.utils import get_checkpoint, check, conv1D_block, encoder1D_block, decoder1D_block


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
        
        self.freq_loss = 1
        self.time_loss = 1
        self.upit_loss = 1
    
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
        
        self.freq_loss = 0
        self.time_loss = 0.1
        self.uipt_loss = 1
        
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
        length = inputs.shape[1]
        transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=None).to(inputs.device)
        inverse_transform = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256).to(inputs.device)
        
        spectro = transform(inputs)
        
        power = torch.abs(spectro)
        phase = torch.angle(spectro)
        
        mask = self.separator(torch.transpose(power, 1, 2))[0]
        mask = torch.transpose(mask, 1, 2)
        mask = self.activation(mask)
        
        power1 = power * mask
        
        spectro1 = power1 * torch.exp(1j * phase)
        
        signal1 = inverse_transform(spectro1, length)
        signal2 = inputs.squeeze(1) - signal1
        
        return signal1, signal2

####################################################################################################

class Network6(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 6, gen, checkpoint)
        
        self.e1 = encoder1D_block(1, 16, kernel_size=25)
        self.e2 = encoder1D_block(16, 32, kernel_size=13)
        self.e3 = encoder1D_block(32, 16, kernel_size=7)
        self.e4 = encoder1D_block(16, 1, kernel_size=5)
        
        self.separator = nn.RNN(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.activation = nn.Sigmoid()
        
        self.d1 = decoder1D_block(1, 16, 0, kernel_size=3)
        self.d2 = decoder1D_block(16, 32, 0, kernel_size=17)
        self.d3 = decoder1D_block(32, 16, 0, kernel_size=25)
        self.d4 = decoder1D_block(16, 8, 0, kernel_size=33)
        self.outputs = nn.Conv1d(8, 1, kernel_size=1, padding=0)
        
        self.load()
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        m = self.activation(m)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        d4 = self.d4(d3, length=s1.shape[2])
        outputs = self.outputs(d4)
        
        y1 = outputs
        y2 = inputs - outputs
        return y1.squeeze(1), y2.squeeze(1)

####################################################################################################

class Network7(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 7, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 1
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 16, kernel_size=25)
        self.e2 = encoder1D_block(16, 32, kernel_size=13)
        self.e3 = encoder1D_block(32, 16, kernel_size=7)
        self.e4 = encoder1D_block(16, 2, kernel_size=5)
        
        self.separator = nn.RNN(input_size=2, hidden_size=2, num_layers=2, batch_first=True)
        
        self.d1 = decoder1D_block(2, 16, 0, kernel_size=3)
        self.d2 = decoder1D_block(16, 32, 0, kernel_size=17)
        self.d3 = decoder1D_block(32, 16, 0, kernel_size=25)
        self.d4 = decoder1D_block(16, 8, 0, kernel_size=33)
        self.outputs = nn.Conv1d(8, 2, kernel_size=1, padding=0)
        
        self.load()
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        d4 = self.d4(d3, length=s1.shape[2])
        outputs = self.outputs(d4)
        
        y1 = outputs[:,0,:]
        y2 = outputs[:,1,:]
        return y1, y2

####################################################################################################
    
class Network8(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 8, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 1
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 16, kernel_size=53)
        self.e2 = encoder1D_block(16, 32, kernel_size=37)
        self.e3 = encoder1D_block(32, 16, kernel_size=25)
        self.e4 = encoder1D_block(16, 2, kernel_size=17)
        
        self.separator = nn.RNN(input_size=2, hidden_size=2, num_layers=2, batch_first=True)
        
        self.d1 = decoder1D_block(2, 16, 0, kernel_size=11)
        self.d2 = decoder1D_block(16, 32, 0, kernel_size=17)
        self.d3 = decoder1D_block(32, 16, 0, kernel_size=25)
        self.d4 = decoder1D_block(16, 8, 0, kernel_size=33)
        self.outputs = nn.Conv1d(8, 2, kernel_size=5, padding="same")
        
        self.load()
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        d4 = self.d4(d3, length=s1.shape[2])
        outputs = self.outputs(d4)
        
        y1 = outputs[:,0,:]
        y2 = outputs[:,1,:]
        return y1, y2

####################################################################################################
    
class Network9(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 9, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0.1
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 2, kernel_size=53)
        self.e2 = encoder1D_block(2, 4, kernel_size=37)
        self.e3 = encoder1D_block(4, 2, kernel_size=25)
        self.e4 = encoder1D_block(2, 2, kernel_size=17)
        
        self.separator = nn.RNN(input_size=2, hidden_size=2, num_layers=100, batch_first=True)
        
        self.d1 = decoder1D_block(2, 4, 0, kernel_size=11)
        self.d2 = decoder1D_block(4, 4, 0, kernel_size=17)
        self.d3 = decoder1D_block(4, 2, 0, kernel_size=25)
        self.d4 = decoder1D_block(2, 2, 0, kernel_size=33)
        
        self.load()
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        d4 = self.d4(d3, length=s1.shape[2])
        outputs = d4
        
        y1 = outputs[:,0,:]
        y2 = outputs[:,1,:]
        return y1, y2

####################################################################################################

class Network10(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 10, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0.1
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51)
        self.e2 = encoder1D_block(4, 8, kernel_size=21)
        self.e3 = encoder1D_block(8, 16, kernel_size=13)
        self.e4 = encoder1D_block(16, 32, kernel_size=5)
        self.b = conv1D_block(32, 64, kernel_size=3)
        self.d1 = decoder1D_block(64, 64, 32, kernel_size=3)
        self.d2 = decoder1D_block(64, 32, 16, kernel_size=17)
        self.d3 = decoder1D_block(32, 16, 8, kernel_size=25)
        self.d4 = decoder1D_block(16, 8, 4, kernel_size=33)
        self.outputs = nn.Conv1d(8, 2, kernel_size=1, padding=0)
        
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
        
        y1 = outputs[:,0,:]
        y2 = outputs[:,1,:]
        return y1, y2

####################################################################################################

class Network11(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 11, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 0, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(16, 8, 0, kernel_size=17, num_conv=1)
        self.d3 = decoder1D_block(8, 4, 0, kernel_size=25, num_conv=1)
        self.d4 = decoder1D_block(4, 1, 0, kernel_size=33, num_conv=1)
        
        self.load()
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        y1 = self.d4(d3, length=s1.shape[2])
        
        y2 = inputs - y1
        
        return y1.squeeze(1), y2.squeeze(1)
    
####################################################################################################

class Network12(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 12, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1, activation=nn.PReLU)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1, activation=nn.PReLU)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1, activation=nn.PReLU)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1, activation=nn.PReLU)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 0, kernel_size=3, num_conv=1, activation=nn.PReLU)
        self.d2 = decoder1D_block(16, 8, 0, kernel_size=17, num_conv=1, activation=nn.PReLU)
        self.d3 = decoder1D_block(8, 4, 0, kernel_size=25, num_conv=1, activation=nn.PReLU)
        self.d4 = decoder1D_block(4, 1, 0, kernel_size=33, num_conv=1, activation=nn.PReLU)
        
        self.load()
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        y1 = self.d4(d3, length=s1.shape[2])
        
        y2 = inputs - y1
        
        return y1.squeeze(1), y2.squeeze(1)


####################################################################################################

class Network13(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 13, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 0, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(16, 8, 0, kernel_size=17, num_conv=1)
        self.d3 = decoder1D_block(8, 4, 0, kernel_size=25, num_conv=1)
        self.d4 = decoder1D_block(4, 2, 0, kernel_size=33, num_conv=1, activation=nn.Identity)
        
        working_freq = 8000
        self.transform = torchaudio.transforms.Resample(16000, working_freq)
        self.inverse_transform = torchaudio.transforms.Resample(working_freq, 16000)
        
        self.load()
    
    def forward(self, inputs):
        
        inputs = self.transform(inputs)
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, length=s4.shape[2])
        d2 = self.d2(d1, length=s3.shape[2])
        d3 = self.d3(d2, length=s2.shape[2])
        y = self.d4(d3, length=s1.shape[2])
        
        y = self.inverse_transform(y)
        
        y1 = y[:,0,:]
        y2 = y[:,1,:]
        
        return y1, y2

####################################################################################################

class Network14(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 14, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 32, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(16, 8, 16, kernel_size=17, num_conv=1)
        self.d3 = decoder1D_block(8, 4, 8, kernel_size=25, num_conv=1)
        self.d4 = decoder1D_block(4, 2, 4, kernel_size=33, num_conv=1, activation=nn.Identity, normalize=False)
        
        working_freq = 8000
        self.transform = torchaudio.transforms.Resample(16000, working_freq)
        self.inverse_transform = torchaudio.transforms.Resample(working_freq, 16000)
        
        self.load()
    
    def forward(self, inputs):
        
        inputs = self.transform(inputs)
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        y = self.d4(d3, s1)
        
        y = self.inverse_transform(y)
        
        y1 = y[:,0,:]
        y2 = y[:,1,:]
        
        return y1, y2

####################################################################################################

class Network15(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 15, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 32, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(16, 8, 16, kernel_size=17, num_conv=1)
        self.d3 = decoder1D_block(8, 4, 8, kernel_size=25, num_conv=1)
        self.d4 = decoder1D_block(4, 1, 4, kernel_size=33, num_conv=1, activation=nn.Identity, normalize=False)
        
        working_freq = 8000
        self.transform = torchaudio.transforms.Resample(16000, working_freq)
        self.inverse_transform = torchaudio.transforms.Resample(working_freq, 16000)
        
        self.load()
    
    def forward(self, inputs):
        
        inputs = self.transform(inputs)
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        y1 = self.d4(d3, s1)
        
        y2 = inputs - y1
        
        y1 = y1.squeeze(1)
        y2 = y2.squeeze(1)
        
        y1 = self.inverse_transform(y1)
        y2 = self.inverse_transform(y2)
        
        return y1, y2

####################################################################################################

class Network16(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 16, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 32, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(16, 8, 16, kernel_size=17, num_conv=1)
        self.d3 = decoder1D_block(8, 4, 8, kernel_size=25, num_conv=1)
        self.d4 = decoder1D_block(4, 2, 4, kernel_size=33, num_conv=1, activation=nn.Identity, normalize=False)
        self.activation = nn.Tanh()
        
        working_freq = 8000
        self.transform = torchaudio.transforms.Resample(16000, working_freq)
        self.inverse_transform = torchaudio.transforms.Resample(working_freq, 16000)
        
        self.load()
    
    def forward(self, inputs):
        
        inputs = self.transform(inputs)
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        y = self.d4(d3, s1)
        
        upper_bond = torch.max(torch.abs(inputs))
        y = upper_bond * self.activation(y)
        
        y = self.inverse_transform(y)
        
        y1 = y[:,0,:]
        y2 = y[:,1,:]
        
        return y1, y2

####################################################################################################

class Network17(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 17, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        
        
        self.d1 = decoder1D_block(32, 16, 32, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(16, 8, 16, kernel_size=17, num_conv=1)
        self.d3 = decoder1D_block(8, 4, 8, kernel_size=25, num_conv=1)
        self.d4 = decoder1D_block(4, 2, 4, kernel_size=33, num_conv=1, activation=nn.Identity, normalize=False)
        
        working_freq = 8000
        self.transform = torchaudio.transforms.Resample(16000, working_freq)
        self.inverse_transform = torchaudio.transforms.Resample(working_freq, 16000)
        
        self.load()
    
    def forward(self, inputs):
        
        inputs = self.transform(inputs)
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        m = torch.transpose(self.separator(torch.transpose(p4, 1, 2))[0], 1, 2)
        b = m*p4
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        y = self.d4(d3, s1)
        
        y = (y - torch.mean(y, dim=2, keepdim=True)) / torch.std(y, dim=2, keepdim=True)
        y = torch.std(inputs, dim=2, keepdim=True) * y + torch.mean(inputs, dim=2, keepdim=True)
        
        
        y = self.inverse_transform(y)
        
        y1 = y[:,0,:]
        y2 = y[:,1,:]
        
        return y1, y2

####################################################################################################

class Network18(Network):
    def __init__(self, gen, checkpoint):
        Network.__init__(self, 18, gen, checkpoint)
        
        self.freq_loss = 0
        self.time_loss = 0
        self.uipt_loss = 1
        
        self.e1 = encoder1D_block(1, 4, kernel_size=51, num_conv=1)
        self.e2 = encoder1D_block(4, 8, kernel_size=21, num_conv=1)
        self.e3 = encoder1D_block(8, 16, kernel_size=13, num_conv=1)
        self.e4 = encoder1D_block(16, 32, kernel_size=5, num_conv=1)
        self.e5 = encoder1D_block(32, 64, kernel_size=5, num_conv=1)
        
        self.separator = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        
        self.d1 = decoder1D_block(64, 32, 64, kernel_size=3, num_conv=1)
        self.d2 = decoder1D_block(32, 16, 32, kernel_size=5, num_conv=1)
        self.d3 = decoder1D_block(16, 8, 16, kernel_size=7, num_conv=1)
        self.d4 = decoder1D_block(8, 4, 8, kernel_size=11, num_conv=1)
        self.d5 = decoder1D_block(4, 2, 4, kernel_size=15, num_conv=1, activation=nn.Identity, normalize=False)
        
        working_freq = 8000
        self.transform = torchaudio.transforms.Resample(16000, working_freq)
        self.inverse_transform = torchaudio.transforms.Resample(working_freq, 16000)
        
        self.load()
    
    def forward(self, inputs):
        
        inputs = self.transform(inputs)
        inputs = inputs.unsqueeze(1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        
        m = torch.transpose(self.separator(torch.transpose(p5, 1, 2))[0], 1, 2)
        b = m*p5
        
        d1 = self.d1(b, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        y = self.d5(d4, s1)
        
        y = (y - torch.mean(y, dim=2, keepdim=True)) / torch.std(y, dim=2, keepdim=True)
        y = torch.std(inputs, dim=2, keepdim=True) * y + torch.mean(inputs, dim=2, keepdim=True)
        
        
        y = self.inverse_transform(y)
        
        y1 = y[:,0,:]
        y2 = y[:,1,:]
        
        return y1, y2