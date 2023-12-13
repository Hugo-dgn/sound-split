import os

import torch
import torch.nn as nn

def get_last_checkpoint(checkpoints_id):
    last_checkpoint = max([int(file.split(".")[0]) for file in os.listdir(os.path.join("save", str(checkpoints_id)))])
    return last_checkpoint

def check(checkpoints_id):
    checkpoint = None
    if "save" not in os.listdir():
        os.mkdir("save")
    if str(checkpoints_id) not in os.listdir("save"):
        os.mkdir(os.path.join("save", str(checkpoints_id)))
    elif len(os.listdir(os.path.join("save", str(checkpoints_id)))) > 0:
        last_checkpoint = get_last_checkpoint(checkpoints_id)
        checkpoint_path = os.path.join("save", str(checkpoints_id), str(last_checkpoint) + ".pth")
        checkpoint = torch.load(checkpoint_path)
    
    return checkpoint

def get_network(n):
    if f"Network{n}" in globals():
        return globals()[f"Network{n}"]
    else:
        message = f"Topology Network{n} is not defined but was requested."
        raise AssertionError(message)
        

class Network1(nn.Module):
    def __init__(self, checkpoints_id):
        super(Network1, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=31, stride=1, padding="same"),
            nn.PReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=31, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=31, stride=1, padding="same"),
        )
        
        checkpoint = check(checkpoints_id)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.encoder(x)
        x1 = self.decoder(x1)
        x2 = x - x1
        
        return x1.squeeze(1), x2.squeeze(1)

    def save(self, checkpoints_id):
        checkpoint = get_last_checkpoint(checkpoints_id) + 1
        checkpoint_path = os.path.join("save", str(checkpoints_id))
        torch.save({
            'model_state_dict': self.state_dict(),
        }, os.path.join(checkpoint_path, str(checkpoint) + ".pth"))
        