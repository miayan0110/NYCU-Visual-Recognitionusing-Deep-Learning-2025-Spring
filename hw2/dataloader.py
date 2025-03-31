import glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class HW2Data(Dataset):
    def __init__(self, img_root, json_path=''):
        super().__init__()
        self.mode = img_root.split('/')[-1]
        self.img_root = img_root
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.mode != 'test':
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.img_list = data['images']
                self.anno_list = data['annotations']
                self.cat_list = data['categories']
        else:
            self.img_list = glob.glob(f'{img_root}/*.png')

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        if self.mode != 'test':
            img_id = self.img_list[index]['id']
            img_path = f"{self.img_root}/{self.img_list[index]['file_name']}"
            img = Image.open(img_path).convert('RGB')
            img = self.preprocess(img)

            annos = [anno for anno in self.anno_list if anno['image_id'] == img_id]
            target = {
                'boxes': torch.tensor([[anno['bbox'][0], anno['bbox'][1], anno['bbox'][0]+anno['bbox'][2], anno['bbox'][1]+anno['bbox'][3]] for anno in annos]),
                'labels': torch.tensor([anno['category_id'] for anno in annos])
            }

            return img, target
        
        img = Image.open(self.img_list[index]).convert('RGB')
        img = self.preprocess(img)
        
        return img, index+1
    