from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
from PIL import Image

import os
import os.path
import sys
import pandas as pd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split

        file_path = f"{root}/{split}.txt"
        data = pd.read_csv(file_path, sep="/", header=None, usecols=[0, 1], names=["label", "img"])
        images = []
        labels = []
        for i, row in data.iterrows():
            label = row["label"]
            print(row["label"])
            if label != "BACKGROUND_Google":
                labels.append(label)
                images.append(pil_loader(f"{root}/101_ObjectCategories/{row['label']}/{row['img']}"))
        self.images = images
        self.labels = pd.factorize(labels)[0]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.images)

