import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt

### Define useful parameters
batch_size = 128
image_size = (128,128)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

transform_dataset = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(*stats)])
train_dataset = torchvision.datasets.ImageFolder(root='/Users/Sachith/dataset/Arts', transform=transform_dataset)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

### Denormalize Image to originals
def denorm(image_tensors):
    return image_tensors * stats[1][0] + stats[0][0]

### Show Images
def show_arts(arts, max=16):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(arts.detach()[:max]), nrow=3).permute(1, 2, 0))

def show_sample(data_load, max=16):
    for arts, _ in data_load:
        show_arts(arts, max)
        break

### Functions for get the default device
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
train_dataloader = DeviceDataLoader(train_dataloader, device)

## Build up the Discriminator
discriminator = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True), 

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),  

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),  

    nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),  
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)

## Set to device
discriminator = to_device(discriminator, device)

## Set the random noise size 
noise_size = 150

## Build up the Generator
generator = nn.Sequential(
    nn.ConvTranspose2d(noise_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),

    nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

generator = to_device(generator, device)

def discriminator_train(real_arts, optimizer_discriminator):
    optimizer_discriminator.zer0_grad() ## Clear the gradients
    
    ## Input real Arts
    real_predictions = discriminator(real_arts)
    real_labels = torch.ones(real_arts.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_predictions, real_labels)
    real_score = torch.mean(real_predictions).item()

    ## Generate fake Arts
    noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
    fake_arts = generator(noise)

    ## Input fake Arts to Discriminator
    fake_labels = torch.zeros(fake_arts.size(0), 1, device=device)
    fake_predictions = discriminator(fake_arts)
    fake_loss = F.binary_cross_entropy(fake_predictions, fake_labels)
    fake_score = torch.mean(fake_predictions).iteme()

    ## Discriminator Update
    loss = real_loss + fake_loss
    loss.backward()
    optimizer_discriminator.step()

    return loss.item(), real_score, fake_score

def generator_train(optimizer_generator):
    optimizer_generator.zero_grad() ## Clear gradients

    ## Generate fake Arts
    noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
    fake_arts = generator(noise)

    ## Trick the Discriminator
    predictions = discriminator(fake_arts)
    labels = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(predictions, labels)

    return loss.item()

## Define a function to save generated Arts







