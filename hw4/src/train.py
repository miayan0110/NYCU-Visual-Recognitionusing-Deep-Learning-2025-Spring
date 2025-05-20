import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from torchmetrics import StructuralSimilarityIndexMeasure


class TVLoss(nn.Module):
    """
    Total variation loss to encourage smoothness in the output image.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        h_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        v_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return h_tv + v_tv

class ResBlock(nn.Module):
    """
    A simple residual block with two 3x3 conv layers.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x

class PromptIRModel(pl.LightningModule):
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.84,
        tv_weight: float = 0.1
    ):
        super().__init__()
        # Base restoration network
        self.net = PromptIR(decoder=True)
        # Refinement head: adds residual detail corrections
        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        # Loss functions
        self.l1_loss   = nn.L1Loss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv_loss   = TVLoss()
        # Loss weights
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.tv_weight   = tv_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Core restoration
        restored = self.net(x)
        # Refinement: learn residual corrections
        residual = self.refine(restored)
        # Combine and clamp to valid [0,1]
        out = torch.clamp(restored + residual, 0.0, 1.0)
        return out

    def training_step(self, batch, batch_idx):
        # Unpack batch; ignore clean_name/de_id
        (_, _), degraded, clean = batch
        # Forward
        output = self(degraded)
        # Compute losses
        l1_val   = self.l1_loss(output, clean)
        ssim_val = 1.0 - self.ssim_loss(output, clean)
        tv_val   = self.tv_loss(output)
        # Weighted sum
        loss = (
            self.l1_weight   * l1_val +
            self.ssim_weight * ssim_val +
            self.tv_weight   * tv_val
        )
        # Logging
        self.log('train/l1', l1_val, prog_bar=True)
        self.log('train/ssim', ssim_val)
        self.log('train/tv', tv_val)
        self.log('train/total_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = {
            'scheduler': LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=15,
                max_epochs=150
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}



def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=[0],strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



