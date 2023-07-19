import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm
import yaml
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from data import *
from model import SRGAN

DATASET_REGISTRY = {
    "set14": "Set14Dataset", 
    "set5": "Set5Dataset",
    "bsd100": "BSD100Dataset"
}

parser = ArgumentParser()
parser.add_argument("config_file", type=str, help="The training configuration file.")
args = parser.parse_args()

config_file = Path(args.config_file)
config = yaml.safe_load(config_file.open())

## Dataset
if config["dataset"] not in DATASET_REGISTRY:
    raise Exception(f"Dataset {config['dataset']} doesn't exsit.")

lr_transform = T.Resize((96, 96))
hr_transform = T.Resize((384, 384))
dataset = globals()[DATASET_REGISTRY[config["dataset"]]](
    root="./data",
    download=True,
    lr_transform=lr_transform,
    hr_transform=hr_transform
)
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

## Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRGAN().to(device)

## Training
optimizer = optim.Adam(params=model.parameters(), lr=config['lr'])
epochs = config['epochs']
patience = config['patience']
if 'save_path' in config:
    save_path = Path(config['save_path'])
else:
    save_path = Path("./checkpoints")
    if not Path("./checkpoints").exists():
        save_path.mkdir()
    
save_period = config['save_period']
count = patience
previous_loss = float("inf")
n_batch = len(dataloader)

model.train()
with tqdm(range(epochs)) as t:
    for epoch in t:
        loss = .0
        for x_lr, x_hr in dataloader:
            batch_loss = model.loss(x_lr.to(device), x_hr.to(device))
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        loss /= n_batch

        if epoch % save_period == 0:
            t.set_postfix(Epoch=epoch, Loss=loss)
            print(f"Save a new checkpoint to{save_path} ...")
            filename = save_path / datetime.now().strftime("%d_%m_%Y-%H_%M_%S.pt")
            torch.save(model.state_dict(), filename)
        
        if previous_loss < loss:
            count -= 1
            if count == 0:
                print(f"Early stopping at epoch {epoch} ...")
        else:
            count = patience

print(f"Saving the last checkpoint to {save_path} ...")
filename = save_path / datetime.now().strftime("%d_%m_%Y-%H_%M_%S.pt")
torch.save(model.state_dict(), filename)