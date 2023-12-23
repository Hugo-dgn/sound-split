import torch
import torch.nn as nn
import torchaudio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

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

class uPITLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.eps = 1e-10
    
    def forward(self, x1, x2, target):
        
        p = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        sisnr1 = torch.sum(scale_invariant_signal_noise_ratio(p, target), dim=1)
        
        p = torch.cat([x2.unsqueeze(1), x1.unsqueeze(1)], dim=1)
        sisnr2 = torch.sum(scale_invariant_signal_noise_ratio(p, target), dim=1)
        
        sisnr = torch.max(torch.stack([sisnr1, sisnr2], dim=1), dim=1).values
        
        return torch.mean(-sisnr)