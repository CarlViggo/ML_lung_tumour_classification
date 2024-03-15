import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class LightingNoise(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

def calculate_mean_std(data_dir, size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)

    return mean, std

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           size, 
                           valid_size=0.1,
                           shuffle=True, 
                           imagenet = False):
    if imagenet:
        # ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    normalize,

])

    else:
        train_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = ImageFolder(
    root=data_dir,
    transform=train_transform,
    )

    valid_dataset = ImageFolder(
    root=data_dir,
    transform=valid_transform,
    )


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    size,
                    shuffle=True,
                    imagenet = False):
    if imagenet:
        # ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    # define transform
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = ImageFolder(
    root=data_dir,
    transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


def get_loader_for_fold(data_dir, indices, batch_size, augment, size, imagenet=False):
    if imagenet:
        # ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean, std=std)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform.transforms.insert(0, transforms.RandomHorizontalFlip())
        train_transform.transforms.insert(0, transforms.RandomRotation(10))
        # Add other augmentations here as needed

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # Creating data loaders for the specified fold
    train_sampler = SubsetRandomSampler(indices['train'])
    valid_sampler = SubsetRandomSampler(indices['valid'])

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader
