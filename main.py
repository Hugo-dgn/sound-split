import os
from tqdm.auto import tqdm

import argparse
from tqdm.auto import tqdm

import torch
import torchaudio
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
from models import get_network, FreqDomainLoss, TmeDomainLoss, uPITLoss

def listen(args):
    if not SOUND:
        message = "Sounddevice module not found. You won't be able to listen to the audio."
        raise AssertionError(message)
    dataset = loader.SoundDataset(TRAIN_DATASET_PATH, letrain_ngth=args.length)
    audio, target = dataset.__getitem__(args.id)
        
    if args.audio == 0:
        audio_np = audio.numpy()
    elif args.audio == 1:
        audio_np = target[0].numpy()
    elif args.audio == 2:
        audio_np = target[1].numpy()
        
    sd.play(audio_np, SAMPLE_RATE, blocking=True)

def spectrogram(args):
    dataset = loader.SoundDataset(TRAIN_DATASET_PATH, letrain_ngth=args.length)
    audio, target = dataset.__getitem__(args.id)
    
    meltransform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128)
    
    spectro = meltransform(audio)
    spectro1 = meltransform(target[0])
    spectro2 = meltransform(target[1])
    
    audio = audio.numpy()
    audio1 = audio1.numpy()
    audio2 = audio2.numpy()
    
    plt.figure("audio")
    plt.subplot(311)
    plt.plot(audio)
    plt.ylabel("combined")
    plt.subplot(312)
    plt.plot(audio1)
    plt.ylabel("audio 1")
    plt.subplot(313)
    plt.plot(audio2)
    plt.ylabel("audio 2")
    
    plt.figure("spectrogram")
    plt.subplot(311)
    plt.imshow(spectro.log2().detach().numpy(), aspect='auto', origin='lower')
    plt.ylabel("combined")
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(spectro1.log2().detach().numpy(), aspect='auto', origin='lower')
    plt.ylabel("audio 1")
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(spectro2.log2().detach().numpy(), aspect='auto', origin='lower')
    plt.ylabel("audio 2")
    plt.colorbar()
    plt.show()

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"train on {device}")
    Network = get_network(args.network)
    model = Network(args.gen, args.checkpoint)
    model = model.to(device)
    
    dataset = loader.SoundDataset(TRAIN_DATASET_PATH, length=args.length, reduce=args.reduce, partition=args.partition)
    traindataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    
    testdataset = loader.SoundDataset(TEST_DATASET_PATH, length=args.length, reduce=args.reduce, partition=args.partition)
    testdatasetloader = DataLoader(testdataset, batch_size=args.batch, shuffle=False)
    
    timecriterion = TmeDomainLoss()
    freqcriterion = FreqDomainLoss()
    uiptcreiterion = uPITLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    model.train()
    
    if args.log:
        wandb.init(
        project="SoundSplit",
        name=f"Network{model.network_id}-Gen{model.gen}",
        id=f"Loss-Network{model.network_id}-Gen{model.gen}",
        resume="allow",
        config={
        "network": args.network,
        "time_loss": model.time_loss,
        "freq_loss": model.freq_loss,
        "uipt_loss": model.uipt_loss,
        }
        )
    
    print(f"training Network{model.network_id}-Gen{model.gen}")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for i, (audio, target) in tqdm(enumerate(traindataloader), total=len(traindataloader)):
            audio = audio.to(device)
            target = target.to(device)
            x1, x2 = model(audio)
            
            optimizer.zero_grad()
            
            if model.freq_loss > 0:
                freqloss = freqcriterion(x1, x2, target)
            else:
                freqloss = 0
            if model.time_loss > 0:
                timeloss = timecriterion(x1, x2, target)
            else:
                timeloss = 0
            if model.uipt_loss > 0:
                uiptloss = uiptcreiterion(x1, x2, target)
            else:
                uiptloss = 0
            
            loss = freqloss + timeloss + uiptloss
            loss.backward()
            optimizer.step()
            
            if args.log:
                report = {}
                if model.freq_loss > 0:
                    report["freqloss"] = freqloss.detach().cpu().item()
                if model.time_loss > 0:
                    report["timeloss"] = timeloss.detach().cpu().item()
                if model.uipt_loss > 0:
                    report["uiptloss"] = uiptloss.detach().cpu().item()
                report["loss"] = loss.detach().cpu().item()
                wandb.log(report)
            
            if (i+1) % args.save == 0:
                model.save()
                
            del audio, target, x1, x2, loss
            torch.cuda.empty_cache()
        
        scheduler.step()
        
        print("Testing")
        test_freq_loss = 0
        test_time_loss = 0
        test_uipt_loss = 0
        test_loss = 0
        with torch.no_grad():
            for audio, target in tqdm(testdatasetloader):
                audio = audio.to(device)
                target = target.to(device)
                x1, x2 = model(audio)
                freqloss = freqcriterion(x1, x2, target)
                timeloss = timecriterion(x1, x2, target)
                uiptloss = uiptcreiterion(x1, x2, target)
                
                test_freq_loss += freqloss.detach().cpu().item()
                test_time_loss += timeloss.detach().cpu().item()
                test_uipt_loss += uiptloss.detach().cpu().item()
                
                test_loss += model.freq_loss * freqloss + model.time_loss * timeloss + model.uipt_loss * uiptloss
                
                del audio, target, x1, x2
                torch.cuda.empty_cache()
        model.save()
        
        if args.log:
            wandb.log({"test_freq_loss": test_freq_loss/len(testdatasetloader),
                      "test_time_loss": test_time_loss/len(testdatasetloader),
                      "test_uipt_loss": test_uipt_loss/len(testdatasetloader),
                      "test_loss": test_loss/len(testdatasetloader)})
    
    if args.log:
        wandb.finish()

def info(args):
    Network = get_network(args.network)
    model = Network(args.gen, args.checkpoint)
    
    torchinfo.summary(model, (1, args.length))
    
def compute(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute on {device}")
    Network = get_network(args.network)
    model = Network(args.gen, args.checkpoint)
    model.eval()
    
    print(f"Using Network{model.network_id}-Gen{model.gen} on checkpoint {model.checkpoint} and audio {args.id}")
    
    dataset = loader.SoundDataset(TEST_DATASET_PATH, length=args.length)
    audio, target = dataset.__getitem__(args.id)
    
    audio = audio.to(device).reshape(1, -1)
    x1, x2 = model(audio)
    
    timecriterion = TmeDomainLoss()
    freqcriterion = FreqDomainLoss()
    uiptcreiterion = uPITLoss()
    
    if model.freq_loss > 0:
        freqloss = freqcriterion(x1, x2, target.unsqueeze(0))
        print(f"Freq loss: {freqloss}")
    else:
        freqloss = 0
    if model.time_loss > 0:
        timeloss = timecriterion(x1, x2, target.unsqueeze(0))
        print(f"Time loss: {timeloss}")
    else:
        timeloss = 0
    if model.uipt_loss > 0:
        uiptloss = uiptcreiterion(x1, x2, target.unsqueeze(0))
        print(f"uPIT loss: {uiptloss}")
    else:
        uiptloss = 0
    
    loss = freqloss + timeloss + uiptloss
    
    print(f"Loss: {loss}")
    
    x1 = x1.detach().numpy().squeeze()
    x2 = x2.detach().numpy().squeeze()
    
    y1 = target[0].detach().numpy()
    y2 = target[1].detach().numpy()
    
    l1 = np.mean((x1-y1)**2) + np.mean((x2-y2)**2)
    l2 = np.mean((x1-y2)**2) + np.mean((x2-y1)**2)
        
    if not SOUND:
        
        message = "Sounddevice module not found. You won't be able to listen to the audio."
        print(message)
    else:
        
        if args.audio == 1:
            print("Playing predicted audio 1")
            sd.play(x1, SAMPLE_RATE, blocking=True)
            print("Playing audio 1")
            if l1 < l2:
                sd.play(y1, SAMPLE_RATE, blocking=True)
            else:
                sd.play(y2, SAMPLE_RATE, blocking=True)
        
        if args.audio == 2:
            print("Playing audio 2")
            sd.play(x2, SAMPLE_RATE, blocking=True)
            print("Playing audio 2")
            if l1 > l2:
                sd.play(y1, SAMPLE_RATE, blocking=True)
            else:
                sd.play(y2, SAMPLE_RATE, blocking=True)
            
    if args.plot:
        
        if l1 < l2:
            plt.figure("sound 1")
            plt.subplot(211)
            plt.plot(y1, label="y1")
            plt.plot(x1, label="x1", alpha=0.5)
            plt.legend(loc="upper right")
            plt.subplot(212)
            plt.plot(y2, label="y2")
            plt.plot(x2, label="x2", alpha=0.5)
            plt.legend(loc="upper right")
        else:
            plt.figure("sound 1")
            plt.subplot(211)
            plt.plot(y2, label="y1")
            plt.plot(x1, label="x1", alpha=0.5)
            plt.legend(loc="upper right")
            plt.subplot(212)
            plt.plot(y1, label="y1")
            plt.plot(x2, label="x2", alpha=0.5)
        
        plt.show()
    
    
def clean(args):
    if args.network is not None:
        networks = [os.path.join("save", str(args.network))]
    else:
        networks = [os.path.join("save", network) for network in os.listdir("save")]
    if args.gen is not None:
        if args.network is None:
            message = "You must specify a network id to specify a generation id"
            raise AssertionError(message)
        if str(args.gen) not in os.listdir(networks[0]):
            message = f"Generation {args.gen} not found in network {args.network}"
            raise AssertionError(message)
        gens = [os.path.join(network, str(args.gen)) for network in networks]
    else:
        gens = [os.path.join(network, gen) for network in networks for gen in os.listdir(network)]
    
    for gen in tqdm(gens):
        checkpoints = os.listdir(gen)
        if len(checkpoints) == 0:
            pass
        else:
            last_checkpoint = max([int(checkpoint.split(".")[0]) for checkpoint in checkpoints])
        if last_checkpoint == 0:
            pass
        else:
            for checkpoint in range(last_checkpoint):
                if os.path.exists(os.path.join(gen, str(checkpoint) + ".pth")):
                    os.remove(os.path.join(gen, str(checkpoint) + ".pth"))
    

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
        "--gen", help="network generation", type=int, default=-1)
    train_parser.add_argument(
        "--checkpoint", help="checkpoint number", type=int, default=-1)
    train_parser.add_argument(
        "--lr", help="learning rate", type=float, default=0.001)
    train_parser.add_argument(
        "--gamma", help="learning rate decay", type=float, default=0.9)
    train_parser.add_argument(
        "--log", help="log the training to wandb", action="store_true")
    train_parser.add_argument(
        "--save", help="number of step between each save", type=int, default=100)
    train_parser.add_argument(
        "--workers", help="number of workers", type=int, default=4)
    train_parser.set_defaults(func=train)
    
    info_parser = subparsers.add_parser(
        "info", help="Display model information")
    info_parser.add_argument(
        "--length", help="length of the audio signal", type=int, default=32000)
    info_parser.add_argument(
        "--network", help="network topology", type=int, default=1)
    info_parser.add_argument(
        "--gen", help="network generation", type=int, default=-1)
    info_parser.add_argument(
        "--checkpoint", help="checkpoint number", type=int, default=-1)
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
    compute_parser.add_argument(
        "--gen", help="network generation", type=int, default=-1)
    compute_parser.add_argument(
        "--checkpoint", help="checkpoint number", type=int, default=-1)
    compute_parser.set_defaults(func=compute)
    
    spectrogram_parser = subparsers.add_parser(
        "spectrogram", help="Display the spectrogram of the audio")
    spectrogram_parser.add_argument(
        "--length", help="length of the audio signal", type=int, default=32000)
    spectrogram_parser.add_argument(
        "--id", help="id of the audio", type=int, default=0)
    spectrogram_parser.set_defaults(func=spectrogram)
    
    clean_parser = subparsers.add_parser(
        "clean", help="Clean the save folder")
    clean_parser.add_argument(
        "--network", help="network topology", type=int)
    clean_parser.add_argument(
        "--gen", help="network generation", type=int)
    clean_parser.set_defaults(func=clean)
    
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    TRAIN_DATASET_PATH = 'train_dataset'
    TEST_DATASET_PATH = 'test_dataset'
    SAMPLE_RATE = 16000
    main()
