import argparse
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

import sounddevice as sd

import loader
from models import get_network

def listen(args):
    dataset = loader.SoundDataset(DATASET_PATH, lenght=args.lenght)
    audio, target = dataset.__getitem__(args.id)
        
    if args.audio == 0:
        audio_np = audio.numpy()
    elif args.audio == 1:
        audio_np = target[0].numpy()
    elif args.audio == 2:
        audio_np = target[1].numpy()
        
    sd.play(audio_np, SAMPLE_RATE, blocking=True)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Network = get_network(1)
    model = Network(1)
    model.to(device)
    dataset = loader.SoundDataset(DATASET_PATH, lenght=args.lenght, reduce=args.reduce, partition=args.partition)
    traindataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    testdataset = loader.SoundDataset(DATASET_PATH, lenght=args.lenght, train=False, reduce=args.reduce)
    
    for epoch in range(args.epochs):
        for audio, target in tqdm(traindataloader):
            audio = audio.to(device)
            target = target.to(device)
            x1, x2 = model(audio)

def main():
    parser = argparse.ArgumentParser(description="Sound split")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    listen_parser = subparsers.add_parser(
        "listen", help="Play an exemple of the dataset")
    listen_parser.add_argument(
        "--lenght", help="lenght of the audio signal", type=int, default=32000)
    listen_parser.add_argument(
        "--audio", help="Which audio to play : 0 for combined audio, 1 and 2 for individual audio", type=int, choices=[0, 1, 2], default=0)
    listen_parser.add_argument("--id", help="id of the audio", type=int, default=0)
    listen_parser.set_defaults(func=listen)
    
    train_parser = subparsers.add_parser(
        "train", help="Train the model")
    train_parser.add_argument(
        "--lenght", help="lenght of the audio signal", type=int, default=32000)
    train_parser.add_argument(
        "--epochs", help="number of epochs", type=int, default=10)
    train_parser.add_argument(
        "--batch_size", help="batch size", type=int, default=32)
    train_parser.add_argument(
        "--reduce", help="reduce the dataset size by this percent", type=float, default=0)
    train_parser.add_argument(
        "--partition", help="number of partitions", type=int, default=1)
    train_parser.set_defaults(func=train)
    
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    DATASET_PATH = 'dataset'
    SAMPLE_RATE = 16000
    main()