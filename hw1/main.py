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
    parser.add_argument('--mode', default='train', type=str)    # execute mode
    parser.add_argument('--ckpt_root', default='./ckpt', type=str)    # checkpoint save path
    parser.add_argument('--save_per_epoch', default=10, type=int)    # save checkpoint per epoch
    parser.add_argument('--result_path', default='prediction.csv', type=str)    # path of prediction.csv

    parser.add_argument('--lr', default=1e-4, type=float)   # learning rate
    parser.add_argument('--batch_size', default=128, type=int)   # batch size
    parser.add_argument('--num_epochs', default=1000, type=int)   # training number of epochs
    parser.add_argument('--resume', action='store_true')    # whether keep training the previous model or not

    args = parser.parse_args()
    return args

def train(args, model, dataloader, resume):
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
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(model, optimizer, epoch + 1, f'{args.ckpt_root}/checkpoint_{epoch+1:04d}.pth')

def val(args, model, dataloader):
    device = f'cuda:{args.gpu_id}'

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

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
        
        print(f"Avg Loss: {total_loss / len(dataloader)}")

def eval(args, model, dataloader):
    device = f'cuda:{args.gpu_id}'

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_list = glob.glob(f'{args.ckpt_root}/*.pth')
    ckpt_list.sort()
    model, _, _ = load_model(model, optimizer, ckpt_list[-1], device)

    with open(args.save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred_label'])

        print('=> start evaluation...')
        model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)

            for img_name, img in pbar:
                img = img.to(device)

                pred = model(img)
                writer.writerow([img_name, pred])


if __name__ == '__main__':
    args = get_args()
    
    print(f'Mode: {args.mode}')
    if args.mode != 'test':
        dataset = TrainValDataset(args.mode)
    else:
        dataset = TestDataset()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(args.mode in ['train', 'val']))
    model = ResNet()

    match(args.mode):
        case 'train':
            train(args, model, dataloader, resume=args.resume)
        case 'val':
            val(args, model, dataloader)
        case _:
            eval(args, model, dataloader)
