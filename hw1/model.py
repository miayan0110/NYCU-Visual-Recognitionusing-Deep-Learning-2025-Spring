import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.linear = nn.Linear(in_features=1000, out_features=100, bias=True)  # adapt output class to 100 categories
        # print(self.resnet)

    def forward(self, img):
        x  = self.resnet(img)
        x = self.linear(x)
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