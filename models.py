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

class SoundLoss(nn.Module):
    def __init__(self):
        super(SoundLoss, self).__init__()
    
    def forward(self, x1, x2, target):
        
        arrange1 = x1 - target[:,0,:]
        arrange2 = x1 - target[:,1,:]
        
        arrangement = torch.stack([arrange1, arrange2], dim=1)**2
        arrangement = torch.mean(arrangement, dim=2)
        value, index = torch.min(arrangement, dim=1)
        loss1 = value.mean()
        
        arrange1 = x2 - target[:,0,:]
        arrange2 = x2 - target[:,1,:]
        
        arrangement = torch.stack([arrange1, arrange2], dim=1)**2
        arrangement = torch.mean(arrangement, dim=2)
        value = arrangement.gather(1, (1-index).unsqueeze(1)).squeeze(1)
        loss2 = value.mean()
        return loss1 + loss2
        

class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()
        checkpoints_id = 1
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
        
        checkpoint, last_checkpoint = check(checkpoints_id)
        if checkpoint is not None:
            print(f"Loading checkpoint {last_checkpoint}")
            self.load_state_dict(checkpoint['model_state_dict'])
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.encoder(x)
        x1 = self.decoder(x1)
        x2 = x - x1
        
        return x1.squeeze(1), x2.squeeze(1)

    def save(self):
        checkpoints_id = 1
        checkpoint = get_last_checkpoint(checkpoints_id) + 1
        checkpoint_path = os.path.join("save", str(checkpoints_id))
        torch.save({
            'model_state_dict': self.state_dict(),
        }, os.path.join(checkpoint_path, str(checkpoint) + ".pth"))
        