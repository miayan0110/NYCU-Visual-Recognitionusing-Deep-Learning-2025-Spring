import torch
import torch.nn as nn
from torchvision import models


class ResNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnext101_64x4d(weights='DEFAULT')
        self.attention = nn.Sequential(
            nn.Linear(1000, 1000 // 2),
            nn.Tanh(),
            nn.Linear(1000 // 2, 1000),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        x  = self.resnet(img)
        attn_weights = self.attention(x)
        x = self.classifier(x * attn_weights)
        return x
    


def load_model(model, optimizer, save_path, device="cuda"):
    print(f"=> loading checkpoint '{save_path}'...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('=> finished.')

    return model, optimizer, epoch

def save_model(model, optimizer, epoch, save_path):
    print(f"=> saving checkpoint to '{save_path}'...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    print('=> finished.')