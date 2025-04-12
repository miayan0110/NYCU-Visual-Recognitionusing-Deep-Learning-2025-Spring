import os
import torch
import torchvision
import json
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import csv

from torch.utils.tensorboard import SummaryWriter
from dataloader import *

def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: {checkpoint_path}, Starting from epoch {start_epoch}")
    return model, optimizer, start_epoch

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def initialize_model(resume=False, device='cuda'):
    logdir = './records'
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
        trainable_backbone_layers=2
    )
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.005, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    start_epoch = 0

    if resume:
        ckpt_path = sorted(glob.glob('./ckpt/*.pth'))[-1]
        model, optimizer, start_epoch = load_checkpoint(
            model, 
            optimizer, 
            ckpt_path, 
            device=device
        )
    
    return model, optimizer, start_epoch, writer

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{
            'boxes': target['boxes'].to(device),
            'labels': target['labels'].to(device)
        } for target in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))
    return total_loss / len(data_loader)

def train_model(data_loader, num_epochs=10, device='cuda', checkpoint_dir='ckpt', resume=False):
    model, optimizer, start_epoch, writer = initialize_model(resume, device=device)

    model.to(device)
    for epoch in range(start_epoch, num_epochs):
        loss = train_one_epoch(model, data_loader, optimizer, device, epoch)
        
        writer.add_scalar('Loss', loss, epoch+1)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
        valid_loader = DataLoader(HW2Data('./nycu-hw2-data/valid', './nycu-hw2-data/valid.json'),batch_size=8, shuffle=False, collate_fn=collate_fn)
        mAP = valid_model(valid_loader, device=device)
        writer.add_scalar('mAP', mAP, epoch+1)
        print(f"Epoch {epoch+1}, mAP: {mAP:.4f}")

def valid_model(data_loader, device='cuda'):
    model, _, _, _ = initialize_model(resume=True, device=device)
    model.to(device)
    model.eval()
    all_preds, all_gts = [], []


    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Validation")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{
                'boxes': target['boxes'].to(device),
                'labels': target['labels'].to(device)
            } for target in targets]
            
            preds = model(images)
            all_preds.extend(pred for pred in preds)
            all_gts.extend(targets)

        metric = MeanAveragePrecision(iou_type="bbox")
        metric.update(preds=all_preds, target=all_gts)
        result = metric.compute()
        mAP = result["map"]
    return mAP

def evaluate_model(data_loader, device='cuda', task1_output_file='pred.json', task2_output_file='pred.csv'):
    model, _, _, _ = initialize_model(resume=True, device=device)
    model.eval()
    model.to(device)
    t1_results = []
    t2_results = []
    
    with torch.no_grad():
        for images, image_ids in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img_id, output in zip(image_ids, outputs):
                img_id = int(img_id) if isinstance(img_id, torch.Tensor) else img_id
                
                pred_label_dict = [img_id]
                valid_results = []
                for bbox, score, category_id in zip(output['boxes'].cpu().tolist(),
                                                     output['scores'].cpu().tolist(),
                                                     output['labels'].cpu().tolist()):
                    if score >= 0.7:
                        xmin, ymin, xmax, ymax = bbox
                        bbox_converted = [xmin, ymin, xmax - xmin, ymax - ymin]
                        result = {
                            'image_id': img_id,
                            'bbox': bbox_converted,
                            'score': score,
                            'category_id': category_id,
                        }
                        valid_results.append(result)
                        t1_results.append(result)
                valid_results_sorted = sorted(valid_results, key=lambda r: r['bbox'][0])
                label_str = ''.join(str(int(r['category_id']) - 1) for r in valid_results_sorted)
                if label_str == '':
                    pred_label_dict.append(-1)
                else:
                    pred_label_dict.append(int(label_str))
                t2_results.append(pred_label_dict)
    
    with open(task1_output_file, 'w') as f:
        json.dump(t1_results, f, indent=4)

    with open(task2_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        for t2_result in t2_results:
            writer.writerow(t2_result)

    print(f"Evaluation results saved to {task1_output_file}, {task2_output_file}")

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def main():
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(HW2Data('./nycu-hw2-data/train', './nycu-hw2-data/train.json'),batch_size=8, shuffle=True, collate_fn=collate_fn)
    train_model(train_loader, num_epochs=10, device=device, resume=False)
    
    test_loader = DataLoader(HW2Data('./nycu-hw2-data/test'), batch_size=1, shuffle=False)
    evaluate_model(test_loader, device=device)

if __name__ == "__main__":
    main()
