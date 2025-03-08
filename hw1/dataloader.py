import os
import glob
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class TrainValDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.mode = mode
        self.folder_list = glob.glob(f'./data/{mode}/*')
        self.img_list = []
        self.preprocess = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        for folder in self.folder_list:
            print(f'processing {folder}...')
            for idx, file in enumerate(os.listdir(folder)):
                src = f'{folder}/{file}'
                # dst = f"{folder}/{folder.split('/')[-1]}_{idx}.jpg"

                # print(f'rename {src} to {dst}...')
                # os.rename(src, dst)

                self.img_list.append(src)
        print(f'=> {len(self.img_list)} images for training.')
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        label = int(self.img_list[index].split('_')[0].split('/')[-1])

        processed_img = self.preprocess(img)
        one_hot_label = torch.zeros(100)
        one_hot_label[label] = 1

        return processed_img, one_hot_label
    
class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.img_list = glob.glob(f'./data/test/*.jpg')
        self.preprocess = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f'=> {len(self.img_list)} images for testing.')
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index].split('/')[-1].split('.')[0]
        img = Image.open(self.img_list[index]).convert('RGB')

        processed_img = self.preprocess(img)

        return img_name, processed_img


