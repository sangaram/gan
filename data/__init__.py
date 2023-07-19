from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from PIL import Image
import requests
from pathlib import Path
from zipfile import ZipFile


class BaseDataset(Dataset):
    def __init__(self, root, download=False, upscale_factor=4, lr_transform=None, hr_transform=None):
        super().__init__()
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.upscale_factor = upscale_factor
        self.root = Path(root)
        self.download = download
        
    def build_data(self):
        upscale_factor = self.upscale_factor
        lr_transform = self.lr_transform
        hr_transform = self.hr_transform
        image_dir = self.root / f'image_SRF_{upscale_factor}'
        self.data = []
        hr_images = []
        for file in image_dir.iterdir():
            if file.name[-6:] == 'HR.png' and file.name not in hr_images:
                hr_images.append(file)
        
        for hr_image_file in hr_images:
            hr_image = to_tensor(Image.open(hr_image_file))
            lr_image_file = hr_image_file.parent / hr_image_file.name.replace('HR', 'LR')
            lr_image = to_tensor(Image.open(lr_image_file))
            if lr_transform is not None:
                lr_image = lr_transform(lr_image)
            
            if hr_transform is not None:
                hr_image = hr_transform(hr_image)
            
            self.data.append((lr_image, hr_image))
            
    def download_data(self, link):
        res = requests.get(link, stream=True)
        file = Path(self.root.name)
        file.touch()
        with file.open("wb") as fd:
            for chunk in res.iter_content(chunk_size=128):
                fd.write(chunk)

        zipfile = ZipFile(file)
        zipfile.extractall(self.root.parent)
        file.unlink()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class Set14Dataset(BaseDataset):
    def __init__(self, root, download=False, upscale_factor=4, lr_transform=None, hr_transform=None):
        super().__init__(f'{root}/Set14', download, upscale_factor, lr_transform, hr_transform)
        if self.download:
            if not self.root.exists():
                link = "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip"
                self.download_data(link)
        
        self.build_data()
        
class Set5Dataset(BaseDataset):
    def __init__(self, root, download=False, upscale_factor=4, lr_transform=None, hr_transform=None):
        super().__init__(f'{root}/Set5', download, upscale_factor, lr_transform, hr_transform)
        if self.download:
            if not self.root.exists():
                link = "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip"
                self.download_data(link)
        
        self.build_data()
        
class BSD100Dataset(BaseDataset):
    def __init__(self, root, download=False, upscale_factor=4, lr_transform=None, hr_transform=None):
        super().__init__(f'{root}/BSD100', download, upscale_factor, lr_transform, hr_transform)
        if self.download:
            if not self.root.exists():
                link = "https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip"
                self.download_data(link)
        
        self.build_data()