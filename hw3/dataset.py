import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import glob


class TrainDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.folder_names = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        img_id = self.folder_names[idx]
        folder_path = os.path.join(self.root, img_id)
        img_path = os.path.join(folder_path, 'image.tif')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(img.transpose(2, 0, 1))

        mask_paths = sorted(glob.glob(os.path.join(folder_path, 'class*.tif')))
        masks, labels = [], []
        for mp in mask_paths:
            class_id = int(os.path.basename(mp).split('class')[1].split('.tif')[0])
            mask_img = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            instances = np.unique(mask_img)
            instances = instances[instances != 0]
            for inst_id in instances:
                bin_mask = (mask_img == inst_id).astype(bool)
                masks.append(bin_mask)
                labels.append(class_id)

        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, image.size[1], image.size[0]), dtype=torch.uint8)

        boxes = []
        for m in masks:
            pos = m.nonzero()
            xmin = torch.min(pos[:, 1])
            xmax = torch.max(pos[:, 1])
            ymin = torch.min(pos[:, 0])
            ymax = torch.max(pos[:, 0])
            boxes.append([xmin, ymin, xmax, ymax])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])

        return image, target