import torch
from torchvision.transforms.functional import to_tensor, resize
import matplotlib.pyplot as plt
from PIL import Image
from model import SRGAN
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help="The path to the trained weights to use.")
parser.add_argument("image", type=str, help="Path to the image to super resolve.")
args = parser.parse_args()

checkpoint = Path(args.checkpoint)
model = SRGAN()
model.load_state_dict(torch.load(checkpoint))

image_path = Path(args.image)
image = resize(to_tensor(Image.open(image_path)), (96, 96))
with torch.no_grad():
    image_sr = model.generate(image.unsqueeze(0)).squeeze(0)

plt.imshow(image_sr.permute(1, 2, 0))