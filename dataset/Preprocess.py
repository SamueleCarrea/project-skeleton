from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import os
import shutil

def Preprocess():
    # Download and extract Tiny ImageNet dataset if not already present
    if not os.path.exists('tiny-imagenet'):
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system('unzip tiny-imagenet-200.zip -d tiny-imagenet')

    # Rearrange validation images into class-specific folders
    with open('tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

        shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')

    # Define the transformations for the training and validation datasets

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=train_transform)
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=64, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, val_loader