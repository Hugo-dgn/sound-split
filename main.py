import argparse
from tqdm.auto import tqdm

import torch
import torchinfo
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import wandb

try:
    import sounddevice as sd
    SOUND = True
except OSError as e:
    print("Warning: sounddevice module not found. You won't be able to listen to the audio.")
    print(e)
    SOUND = False

import loader
from models import get_network, SoundLoss

def listen(args):
    if not SOUND:
        message = "Sounddevice module not found. You won't be able to listen to the audio."
        raise AssertionError(message)
    dataset = loader.SoundDataset(DATASET_PATH, length=args.length)
    audio, target = dataset.__getitem__(args.id)
        
    if args.audio == 0:
        audio_np = audio.numpy()
    elif args.audio == 1:
        audio_np = target[0].numpy()
    elif args.audio == 2:
        audio_np = target[1].numpy()
        
    sd.play(audio_np, SAMPLE_RATE, blocking=True)

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"train on {device}")
    Network = get_network(args.network)
    model = Network()
    model = model.to(device)
    
    dataset = loader.SoundDataset(DATASET_PATH, length=args.length, reduce=args.reduce, partition=args.partition)
    traindataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    
    testdataset = loader.SoundDataset(DATASET_PATH, length=args.length, reduce=args.reduce, partition=args.partition, train=False)
    testdatasetloader = DataLoader(testdataset, batch_size=args.batch, shuffle=True)
    
    criterion = SoundLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="SoundSplit",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": f"Network{args.network}",
    }
    )
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for audio, target in tqdm(traindataloader):
            audio = audio.to(device)
            target = target.to(device)
            x1, x2 = model(audio)
            
            optimizer.zero_grad()
            loss = criterion(x1, x2, target)
            loss.backward()
            optimizer.step()
            
            wandb.log({"loss": loss.detach().cpu().item()})
        
        print("Testing")
        loss = 0
        for audio, target in tqdm(testdatasetloader):
            audio = audio.to(device)
            target = target.to(device)
            x1, x2 = model(audio)
            loss += criterion(x1, x2, target).cpu().item()
        print(f"Accuracy: {1/(1+loss/len(testdataset))}")
        
        model.save()
    
    wandb.finish()

def info(args):
    Network = get_network(args.network)
    model = Network()
    
    torchinfo.summary(model, (1, args.length))
    
def compute(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute on {device}")
    Network = get_network(args.network)
    model = Network()
    model = model.to(device)
    model.eval()
    
    dataset = loader.SoundDataset(DATASET_PATH, length=args.length)
    audio, target = dataset.__getitem__(args.id)
    
    audio = audio.to(device).reshape(1, -1)
    x1, x2 = model(audio)
    
    criterion = SoundLoss()
    
    loss = criterion(x1, x2, target.unsqueeze(0))
    
    print(f"Loss: {loss}")
    print(f"Acuuracy: {1/(1+loss)}")
    
    x1 = x1.detach().numpy().squeeze()
    x2 = x2.detach().numpy().squeeze()
    
    y1 = target[0].detach().numpy()
    y2 = target[1].detach().numpy()
        
    if not SOUND:
        
        message = "Sounddevice module not found. You won't be able to listen to the audio."
        print(message)
    else:
        
        if args.audio == 1:
            print("Playing audio 1")
            sd.play(x1, SAMPLE_RATE, blocking=True)
        
        if args.audio == 2:
            print("Playing audio 2")
            sd.play(x2, SAMPLE_RATE, blocking=True)
            
    if args.plot:
    
        d1 = np.linalg.norm(x1-y1)
        d2 = np.linalg.norm(x1-y2)
        
        if d1 < d2:
            plt.figure("sound 1")
            plt.plot(x1, label="x1")
            plt.plot(y1, label="y1", alpha=0.5)
            
            plt.figure("sound 2")
            plt.plot(x2, label="x2")
            plt.plot(y2, label="y2", alpha=0.5)
        else:
            plt.figure("sound 1")
            plt.plot(x1, label="x1")
            plt.plot(y2, label="y1", alpha=0.5)
            
            plt.figure("sound 2")
            plt.plot(x2, label="x2")
            plt.plot(y1, label="y1", alpha=0.5)
        
        plt.legend()
        
        plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Sound split")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    listen_parser = subparsers.add_parser(
        "listen", help="Play an exemple of the dataset")
    listen_parser.add_argument(
        "--length", help="length of the audio signal", type=int, default=32000)
    listen_parser.add_argument(
        "--audio", help="Which audio to play : 0 for combined audio, 1 and 2 for individual audio", type=int, choices=[0, 1, 2], default=0)
    listen_parser.add_argument("--id", help="id of the audio", type=int, default=0)
    listen_parser.set_defaults(func=listen)
    
    train_parser = subparsers.add_parser(
        "train", help="Train the model")
    train_parser.add_argument(
        "--length", help="length of the audio signal", type=int, default=32000)
    train_parser.add_argument(
        "--epochs", help="number of epochs", type=int, default=10)
    train_parser.add_argument(
        "--batch", help="batch size", type=int, default=32)
    train_parser.add_argument(
        "--reduce", help="reduce the dataset size by this percent", type=float, default=0)
    train_parser.add_argument(
        "--partition", help="number of partitions", type=int, default=1)
    train_parser.add_argument(
        "--network", help="network topology", type=int, default=1)
    train_parser.add_argument(
        "--lr", help="learning rate", type=float, default=0.001)
    train_parser.set_defaults(func=train)
    
    info_parser = subparsers.add_parser(
        "info", help="Display model information")
    info_parser.add_argument(
        "--length", help="length of the audio signal", type=int, default=32000)
    info_parser.add_argument(
        "--network", help="network topology", type=int, default=1)
    info_parser.set_defaults(func=info)
    
    compute_parser = subparsers.add_parser(
        "compute", help="Compute the output of the model")
    compute_parser.add_argument(
        "--length", help="length of the audio signal", type=int, default=32000)
    compute_parser.add_argument(
        "--network", help="network topology", type=int, default=1)
    compute_parser.add_argument(
        "--id", help="id of the audio", type=int, default=0)
    compute_parser.add_argument(
        "--audio", help="Which audio to play : 0 for combined audio, 1 and 2 for individual audio", type=int, choices=[0, 1, 2], default=0)
    compute_parser.add_argument(
        "--plot", help="plot the audio", action="store_true")
    compute_parser.set_defaults(func=compute)
    
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    DATASET_PATH = 'dataset'
    SAMPLE_RATE = 16000
    main()
