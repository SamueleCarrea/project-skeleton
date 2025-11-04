import os
import shutil

# Download and extract Tiny ImageNet dataset if not already present
os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
os.system('unzip tiny-imagenet-200.zip -d data/tiny-imagenet')

# Rearrange validation images into class-specific folders
with open('data/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'data/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'data/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'data/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('data/tiny-imagenet/tiny-imagenet-200/val/images')