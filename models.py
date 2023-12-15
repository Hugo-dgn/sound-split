import os

import torch
import torch.nn as nn

def get_last_checkpoint(checkpoints_id):
    save_files = [int(file.split(".")[0]) for file in os.listdir(os.path.join("save", str(checkpoints_id)))]
    if len(save_files) == 0:
        last_checkpoint = 0
    else:
        last_checkpoint = max(save_files)
    return last_checkpoint

def check(checkpoints_id):
    checkpoint = None
    last_checkpoint = None
    if "save" not in os.listdir():
        os.mkdir("save")
    if str(checkpoints_id) not in os.listdir("save"):
        os.mkdir(os.path.join("save", str(checkpoints_id)))
    elif len(os.listdir(os.path.join("save", str(checkpoints_id)))) > 0:
        last_checkpoint = get_last_checkpoint(checkpoints_id)
        checkpoint_path = os.path.join("save", str(checkpoints_id), str(last_checkpoint) + ".pth")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    return checkpoint, last_checkpoint

def get_network(n):
    if f"Network{n}" in globals():
        return globals()[f"Network{n}"]
    else:
        message = f"Topology Network{n} is not defined but was requested."
        raise AssertionError(message)

####################################################################################################

class Network(nn.Module):
    
    def __init__(self, checkpoints_id):
        nn.Module.__init__(self)
        self.checkpoints_id = checkpoints_id
    
    def save(self):
        checkpoint = get_last_checkpoint(self.checkpoints_id) + 1
        checkpoint_path = os.path.join("save", str(self.checkpoints_id))
        torch.save({
            'model_state_dict': self.state_dict(),
        }, os.path.join(checkpoint_path, str(checkpoint) + ".pth"))
    
    def load(self):
        checkpoint, last_checkpoint = check(self.checkpoints_id)
        if checkpoint is not None:
            print(f"Loading checkpoint {last_checkpoint}")
            self.load_state_dict(checkpoint['model_state_dict'])

####################################################################################################

class SoundLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, x1, x2, target):
        
        arrange1 = x1 - target[:,0,:]
        arrange2 = x1 - target[:,1,:]
        
        arrange1 = arrange1 / torch.max(torch.abs(arrange1), dim=1)[0].unsqueeze(1)
        arrange2 = arrange2 / torch.max(torch.abs(arrange2), dim=1)[0].unsqueeze(1)
        
        arrangement = torch.stack([arrange1, arrange2], dim=1)**2
        arrangement = torch.mean(arrangement, dim=2)
        value, index = torch.min(arrangement, dim=1)
        loss1 = value.mean()
        
        arrange1 = x2 - target[:,0,:]
        arrange2 = x2 - target[:,1,:]
        
        arrange1 = arrange1 / torch.max(torch.abs(arrange1), dim=1)[0].unsqueeze(1)
        arrange2 = arrange2 / torch.max(torch.abs(arrange2), dim=1)[0].unsqueeze(1)
        
        arrangement = torch.stack([arrange1, arrange2], dim=1)**2
        arrangement = torch.mean(arrangement, dim=2)
        value = arrangement.gather(1, (1-index).unsqueeze(1)).squeeze(1)
        loss2 = value.mean()
        return loss1 + loss2
        

class Network1(Network):
    def __init__(self):
        Network.__init__(self, 1)
        
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
            nn.Dropout(0.2),
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
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
        return x, p

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

class Network2(Network):
    def __init__(self):
        Network.__init__(self, 2)
        
        self.e1 = encoder_block(1, 16, kernel_size=25)
        self.e2 = encoder_block(16, 32, kernel_size=13)
        self.e3 = encoder_block(32, 64, kernel_size=7)
        self.e4 = encoder_block(64, 64, kernel_size=5)
        self.b = conv_block(64, 128, kernel_size=3)
        self.d1 = decoder_block(128, 64, 64, kernel_size=5)
        self.d2 = decoder_block(64, 32, 32, kernel_size=7)
        self.d3 = decoder_block(32, 16, 16, kernel_size=13)
        self.d4 = decoder_block(16, 8, 1, kernel_size=15)
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