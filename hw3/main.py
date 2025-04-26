import os
import argparse
import tempfile
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from PIL import Image
from torchvision import transforms
import json

from dataset import *
from torch.utils.tensorboard import SummaryWriter



def get_maskrcnn_model_v2(args, num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)        

    start_epoch = 0
    if args.resume_path:
        print(f"Loaded model weights from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    print(f'model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, optimizer, start_epoch

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path} (epoch {epoch})")

def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    scaler = GradScaler(device=device)
    total_loss = 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for imgs, targets in pbar:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(device_type='cuda'):
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))
    return total_loss / len(data_loader)
    
def train(args, num_classes, train_loader, val_loader):
    logdir = './records/loss'
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    model, optimizer, start_epoch = get_maskrcnn_model_v2(args, num_classes=num_classes)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.8*args.epochs), int(0.9*args.epochs)], gamma=0.1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss = train_one_epoch(model, optimizer, train_loader, args.device, epoch)
        writer.add_scalar('Loss', loss, epoch+1)
        torch.cuda.empty_cache()
        validation(model, val_loader, args.device, epoch+1)
        lr_scheduler.step()
        save_checkpoint(model, optimizer, epoch+1, f"{args.save_dir}/epoch_{epoch+1:03d}.pth")

def validation(model, data_loader, device, epoch):
    logdir = './records/mAP'
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    model.eval()
    results = []
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 5)]
    }
    ann_id = 1

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validating')
        for imgs, targets in pbar:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds   = model(imgs)

            for i, output in enumerate(preds):
                image_id = int(targets[i]["image_id"].item())
                h, w = imgs[i].shape[-2:]
                coco_gt["images"].append({
                    "id": image_id,
                    "height": h,
                    "width": w,
                    "file_name": f"{image_id}.jpg"
                })

                gt_masks = targets[i]["masks"].cpu().numpy()
                gt_labels = targets[i]["labels"].cpu().numpy()
                for j in range(len(gt_masks)):
                    mask = gt_masks[j]
                    encoded_mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(gt_labels[j]),
                        "segmentation": encoded_mask,
                        "iscrowd": 0,
                        "area": int(mask.sum()),
                        "bbox": list(mask_utils.toBbox(encoded_mask))
                    })
                    ann_id += 1

                pred_masks = output["masks"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                for k in range(len(pred_scores)):
                    if pred_scores[k] >= 0.5:
                        mask = pred_masks[k, 0] > 0.5
                        encoded_mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                        encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")
                        results.append({
                            "image_id": image_id,
                            "category_id": int(pred_labels[k]),
                            "segmentation": encoded_mask,
                            "score": float(pred_scores[k])
                        })

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as pred_f, \
         tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as gt_f:

        json.dump(results, pred_f)
        json.dump(coco_gt, gt_f)
        pred_f.flush()
        gt_f.flush()

        cocoGt = COCO(gt_f.name)
        cocoDt = cocoGt.loadRes(pred_f.name)

        cocoEval = COCOeval(cocoGt, cocoDt, iouType='segm')
        cocoEval.params.iouThrs = np.array([0.5], dtype=float)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP = cocoEval.stats[0]
        writer.add_scalar('segm_mAP', mAP, epoch)

def inference(args, num_classes):
    model, _, _ = get_maskrcnn_model_v2(args, num_classes=num_classes)
    model.eval()
    transform = transforms.ToTensor()

    with open(os.path.join(args.data_root, "test_image_name_to_ids.json"), "r") as f:
        image_info = json.load(f)

    test_dir = os.path.join(args.data_root, "test_release")
    results = []
    for info in tqdm(image_info, desc="Inferencing"):
        file_name = info["file_name"]
        image_id = info["id"]
        image_path = os.path.join(test_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device=args.device)

        with torch.no_grad():
            preds = model(image_tensor)[0]

        for i in range(len(preds["scores"])):
            mask = preds["masks"][i, 0].cpu().numpy() > 0.8
            encoded_mask = encode_mask(mask)

            results.append({
                "image_id": image_id,
                "bbox": preds["boxes"][i].tolist(),
                "score": float(preds["scores"][i]),
                "category_id": preds["labels"][i].item(),
                "segmentation": encoded_mask
            })

    with open("test-results.json", "w") as f:
        json.dump(results, f)



def parse_args():
    parser = argparse.ArgumentParser("Train Mask R-CNN on custom dataset")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/maskrcnn')
    parser.add_argument('--result_path', type=str, default='test-results.json')
    parser.add_argument('--mode', type=str, default='train')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'train':
        # training
        dataset = TrainDataset('./dataset/train')
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size

        torch.manual_seed(0)
        dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        train(args, num_classes=5, train_loader=train_loader, val_loader=val_loader)
    else:
        # inferencing
        inference(args, num_classes=5)