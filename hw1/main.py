import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import csv

from dataloader import *
from model import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=7, type=int)   # id of usage gpu
    parser.add_argument('--mode', default='train', type=str)    # execute mode (train/val/test)
    parser.add_argument('--ckpt_root', default='./ckpt', type=str)    # checkpoint save root
    parser.add_argument('--ckpt_path', default='', type=str)    # checkpoint save path (for evaluating)
    parser.add_argument('--save_per_epoch', default=10, type=int)    # save checkpoint per epoch
    parser.add_argument('--result_path', default='prediction.csv', type=str)    # save path of prediction.csv

    parser.add_argument('--lr', default=1e-4, type=float)   # learning rate
    parser.add_argument('--batch_size', default=64, type=int)   # batch size
    parser.add_argument('--num_epochs', default=1000, type=int)   # training number of epochs
    parser.add_argument('--resume', action='store_true')    # whether keeping training the previous model or not

    args = parser.parse_args()
    return args

def train(args, model, dataloader, val_dataloader, resume):
    device = f'cuda:{args.gpu_id}'

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    start_epoch = 0

    if resume:
        ckpt_list = glob.glob(f'{args.ckpt_root}/*.pth')
        ckpt_list.sort()
        model, optimizer, start_epoch = load_model(model, optimizer, ckpt_list[-1], device)

    print('=> start training...')
    model.train()

    val_loss_list = [0.003998014703392982]
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())        
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(dataloader)}")

        val_loss = val(args, model, val_dataloader)
        val_loss_list.append(val_loss)
        if (epoch+1) % args.save_per_epoch == 0 and max(val_loss_list) == val_loss:
            save_model(model, optimizer, epoch + 1, f'{args.ckpt_root}/checkpoint_{epoch+1:04d}.pth')

def val(args, model, dataloader):
    device = f'cuda:{args.gpu_id}'
    criterion = nn.MSELoss()

    if args.mode == 'val':
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        ckpt_list = glob.glob(f'{args.ckpt_root}/*.pth')
        ckpt_list.sort()
        model, _, _ = load_model(model, optimizer, ckpt_list[-1], device)

    print('=> start validation...')
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader)

        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)

            pred = model(img)
            loss = criterion(pred, label)

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())        
        
        print(f"Val Avg Loss: {total_loss / len(dataloader)}")
        return total_loss / len(dataloader)

def eval(args, model, dataloader):
    device = f'cuda:{args.gpu_id}'

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_list = glob.glob(f'{args.ckpt_root}/*.pth')
    ckpt_list.sort()
    load_path = ckpt_list[-1]
    if args.ckpt_path != '':
        load_path = args.ckpt_path
    model, _, _ = load_model(model, optimizer, load_path, device)
    get_model_size(model)

    with open(args.result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred_label'])

        print('=> start evaluation...')
        model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)

            for img_name, img in pbar:
                img = img.to(device)

                pred = model(img)
                writer.writerow([img_name[0], torch.argmax(pred).item()])

def get_model_size(model):
    param_num = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f'#Parameters: {param_num:.2f}M')


if __name__ == '__main__':
    args = get_args()
    
    print(f'Mode: {args.mode}')
    if args.mode == 'test':
        dataset = TestDataset()
        args.batch_size = 1
    else:
        dataset = TrainValDataset(args.mode)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(args.mode in ['train', 'val']))
    model = ResNet(num_classes=100)

    match(args.mode):
        case 'train':
            val_dataset = TrainValDataset('val')
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=(args.mode in ['train', 'val']))
            train(args, model, dataloader, val_dataloader, resume=args.resume)
        case 'val':
            val(args, model, dataloader)
        case _:
            eval(args, model, dataloader)
