import os
import torch
import torchvision
import json
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from dataloader import *


logdir = './records'
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 2048

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),) * 5)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                    num_classes=11,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()



def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in pbar:
        images = [images.to(device).squeeze(0)]
        targets = [{
            'boxes': targets['boxes'].to(device).squeeze(0),
            'labels': targets['labels'].to(device).squeeze(0)
        }]
        
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))
    return total_loss / len(data_loader)

def train_model(model, data_loader, num_epochs=10, device='cuda', checkpoint_dir='ckpt'):
    model.to(device)
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, data_loader, optimizer, device, epoch)
        
        writer.add_scalar('Loss', loss, epoch+1)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

def evaluate_model(model, data_loader, device='cuda', output_file='pred.json'):
    model.eval()
    model.to(device)
    results = []
    with torch.no_grad():
        for images, image_ids in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for img_id, output in zip(image_ids, outputs):
                result = {
                    'image_id': img_id,
                    'bbox': output['boxes'].cpu().tolist(),
                    'score': output['scores'].cpu().tolist(),
                    'category_id': output['labels'].cpu().tolist(),
                }
                results.append(result)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {output_file}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: {checkpoint_path}, Starting from epoch {start_epoch}")
    return start_epoch

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(HW2Data('./nycu-hw2-data/train', './nycu-hw2-data/train.json'), batch_size=1, shuffle=True)
    train_model(model, train_loader, num_epochs=10, device=device)
    
    test_loader = DataLoader(HW2Data('./nycu-hw2-data/test'), batch_size=1, shuffle=False)
    evaluate_model(model, test_loader, device=device)



if __name__ == "__main__":
    main()
