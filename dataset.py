from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, img_path='/home/josh/Data/', label_path='/home/josh/Data/gt.txt'):
        super(CustomDataset, self).__init__()

        self.img_path = img_path
        self.label_path = label_path

        file_list = os.listdir(self.img_path)
        self.file_list_jpg = [file for file in file_list if file.endswith(".jpg")]

        gt_file = open(self.label_path, 'r')

        lines = gt_file.readlines()
        lines = [line.strip() for line in lines]

        self.gt_dict = dict()
        for line in lines:
            line = line.split('\t')
            self.gt_dict[line[0] + '.jpg'] = line[1]

    def __len__(self):
        return len(self.file_list_jpg)

    def __getitem__(self, idx):
        img_path = self.file_list_jpg[idx]
        img_path = os.path.join(self.img_path, img_path)

        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = transforms.ToTensor()(img)

        gt = self.gt_dict[self.file_list_jpg[idx]]
        one_hot_gt = np.zeros(1)
        one_hot_gt = int(gt)
        # one_hot_gt = np.eye(self.num_classes)[gt]
        # print(one_hot_gt)

        return img, one_hot_gt

if __name__ == '__main__':
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

    for step, data in enumerate(dataloader):
        img, gt = data
        print(img.shape)
        print(gt.shape)