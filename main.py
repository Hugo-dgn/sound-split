import argparse
from tqdm.auto import tqdm

import torch
import torchinfo
from torch.utils.data import DataLoader

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
        
        print("Testing")
        loss = 0
        for audio, target in tqdm(testdatasetloader):
            audio = audio.to(device)
            target = target.to(device)
            x1, x2 = model(audio)
            loss += criterion(x1, x2, target).cpu().item()
        print(f"Accuracy: {1/(1+loss/len(testdataset))}")
        
        model.save()

def info(args):
    Network = get_network(1)
    model = Network(1)
    
    torchinfo.summary(model, (1, args.length))
    
def compute(args):
    if not SOUND:
        message = "Sounddevice module not found. You won't be able to listen to the audio."
        raise AssertionError(message)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute on {device}")
    Network = get_network(args.network)
    model = Network()
    model = model.to(device)
    model.eval()
    
    dataset = loader.SoundDataset(DATASET_PATH, length=args.length)
    audio, target = dataset.__getitem__(args.id)
    
    audio = audio.to(device)
    x1, x2 = model(audio)
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()
        
    sd.play(x1, SAMPLE_RATE, blocking=True)
    sd.play(x2, SAMPLE_RATE, blocking=True)
    

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
    info_parser.set_defaults(func=info)
    
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    DATASET_PATH = 'dataset'
    SAMPLE_RATE = 16000
    main()
