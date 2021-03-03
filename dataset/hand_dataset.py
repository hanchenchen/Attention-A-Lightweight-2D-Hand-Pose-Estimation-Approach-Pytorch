import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt


parts = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]


groups6 = [
    [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19], [0, 4, 8, 12, 16],
]

groups1 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
]


class HandDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        self.img_size = 224
        self.joints = 21  # 21 heat maps
        self.label_size = 2  # (x, y)
        self.mode = mode

        self.data_root = data_root
        self.img_names = json.load(open(os.path.join(self.data_root, 'partitions.json')))[mode]
        self.all_labels = json.load(open(os.path.join(self.data_root, 'labels.json')))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]  # '00000001.jpg'

        # ********************** get image **********************
        im = Image.open(os.path.join(self.data_root, 'imgs', img_name))
        w, h = im.size

        im = im.resize((self.img_size, self.img_size))

        image = transforms.ToTensor()(im)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)

        # ******************** get label  **********************
        img_label = self.all_labels[img_name]  # origin label list  21 * 2

        label = np.asarray(img_label)  # 21 * 2
        label[:, 0] = label[:, 0] * self.img_size / w
        label[:, 1] = label[:, 1] * self.img_size / h

        return image, label, img_name, w, h


# test case
if __name__ == "__main__":
    data_root = 'data_sample/cmuhand'

    print('Dataset ===========>')
    data = HandDataset(data_root=data_root, mode='train')
    image, label, img_name, w, h = data[0]
    # ***************** draw Limb map *****************
    print(image.shape, label.shape, img_name, w, h)





