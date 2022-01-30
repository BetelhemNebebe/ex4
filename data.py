import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.label = data
        self.mode = mode

        self._transform = {'train': tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), 
                                                  tv.transforms.Normalize(mean = train_mean, std=train_std),])
                            'val': tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),])
                          }
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path_img = '.\images.zip\images\'
        img = Path(path_img, self.label.iloc[index, 0])
        img_label = self.label.iloc[index, [1,2]]

        img= gray2rgb(img)
        if self._transform:
            img = self._transform[self.mode](img)

        image_label =[]
        for x, y in zip(img, img_label):
            image_label.append(img[x], img_label[y])

        return image_label




