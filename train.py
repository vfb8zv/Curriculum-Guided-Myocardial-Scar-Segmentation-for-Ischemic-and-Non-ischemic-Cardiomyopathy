import scipy.io as sio
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import shutil, os, json
import torch
import random
from torch.utils.data import Dataset, Sampler, DataLoader
import argparse
import torch.nn.functional as F


from scipy import ndimage
import torchvision

def std_img(tens):
    t_ = (tens-tens.min())/(tens.max()-tens.min()+1e-14)
    return t_


def resize_volume(img, ex=64, order=1):
    current_depth = img.shape[0]
    current_width = img.shape[1]            

    depth_factor = ex / current_depth
    width_factor = ex / current_width

    factors = (depth_factor, width_factor)

    return ndimage.zoom(img, factors, order=order)


def load_matching_weights(model, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:  
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()

    matched_weights = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(matched_weights)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(matched_weights)}/{len(model_dict)} layers from checkpoint")
    return model


######### data loader ###########

class DataLoading(Dataset):
    def __init__(self, data_dir, test_flag, image_size=64, stage='0'):
        
        if not test_flag:
            self.data_dir = os.path.join(data_dir, 'train', stage)
        else:
            self.data_dir = os.path.join(data_dir, 'test')
                        
        self.image_size = image_size

        self.filenames = [f for f in os.listdir(self.data_dir) if "raw_" in f]
        self.labels = []
        for f in self.filenames:
            lge_file = f.replace("raw_", "lge_")
            lge_path = os.path.join(self.data_dir, lge_file)
            lge_np = np.load(lge_path)
            label = 1 if np.sum(lge_np) >= 1 else 0
            self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    
    def __getitem__(self, idx):
        raw_file = self.filenames[idx]
        raw_path = os.path.join(self.data_dir, raw_file)

        lge_file = raw_file.replace("raw_", "lge_")
        lge_path = os.path.join(self.data_dir, lge_file)

        raw_np = np.load(raw_path)
        lge_np = np.load(lge_path)

        raw_np = resize_volume(raw_np, ex=self.image_size, order=1)
        lge_np = resize_volume(lge_np, ex=self.image_size, order=0)

        image = torch.from_numpy(raw_np).float()
        seg = std_img(torch.from_numpy(np.nan_to_num(lge_np, copy=False, nan=0.0)).float())
        
        subject = raw_file.split("_")[0]
        
        label = 1 if seg.sum() >= 1 else 0

        seg[seg>=0.5]=1
        seg[seg<0.5]=0
        

        return (
            std_img(image).unsqueeze(0),  # image
            seg.unsqueeze(0),    # segmentation
            torch.tensor(label, dtype=torch.long),
            subject
        )

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_indices = np.where(self.labels == 1)[0].tolist()
        self.neg_indices = np.where(self.labels == 0)[0].tolist()
        self.num_batches = min(len(self.pos_indices), len(self.neg_indices)) * 2 // batch_size

    def __iter__(self):
        pos = self.pos_indices.copy()
        neg = self.neg_indices.copy()
        random.shuffle(pos)
        random.shuffle(neg)

        for _ in range(self.num_batches):
            batch = []
            batch.extend(random.sample(pos, self.batch_size // 2))
            batch.extend(random.sample(neg, self.batch_size // 2))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def get_data(data_dir="./dataset/", test_flag=False, batch_size=2, image_size=64, stage='0', shuffle=True):
    ds = DataLoading(data_dir, test_flag=test_flag, image_size=image_size, stage=stage)
    sampler = BalancedBatchSampler(ds.labels, batch_size)
    datal = DataLoader(ds, batch_sampler=sampler)
    return datal



############### model utils ###################

import torch.nn.functional as F

def weighted_dice_loss(y_true, y_pred, myo_mask, w_fg=0.8, w_bg=0.2, smooth=1e-10):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    myo_mask = myo_mask.view(-1)
    
    fg_mask = myo_mask == 1
    y_true_f = y_true_f[fg_mask]
    y_pred_f = y_pred_f[fg_mask]
    
    intersection_fg = (y_true_f * y_pred_f).sum()
    dice_fg = (2 * intersection_fg + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

    y_true_bg = 1 - y_true_f
    y_pred_bg = 1 - y_pred_f
    intersection_bg = (y_true_bg * y_pred_bg).sum()
    dice_bg = (2 * intersection_bg + smooth) / (y_true_bg.sum() + y_pred_bg.sum() + smooth)

    return 1 - (w_fg * dice_fg + w_bg * dice_bg)


# Import desired model

from models_scar.scarnet import ScarNet 
# from get_net import get_transunet
# from attention_model import AttentionUNet

def model_main(args):
    
    loss_type= 'cl'
    
    DEVICE= 'cuda:0'
    batch_size = args.batch_size
    stage = args.stage
    epochs= args.epochs
    w_fg= args.w_fg
    w_bg= 1-w_fg
    model_save_path= f'{args.model_type}_{loss_type}_{epochs}.pt'

    datal= get_data('./dataset/', test_flag=False, batch_size=batch_size, image_size=224, stage=stage)
    d_= iter(datal)

    # select desired model

    model = ScarNet(pretrained_path = './models_scar/medsam_vit_b.pth', num_classes=1).cuda()
    # model = get_transunet().cuda()
    # model = AttentionUNet(drop_out_prob=0).cuda()

    loss_scale = 0.6


    ################################## Model Training ##############################################

    os.makedirs(f'./results/models_{loss_type}_{args.model_type}/', exist_ok=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay= 1e-14, lr=1e-3)

    loss_pre= nn.Sigmoid()

    for ep in range(epochs):
        losses=[]
        focal_losses, dice_losses = [], []

        d_= iter(datal)

        #### STEPS ####
        for _ in range(len(datal)):
            try:
                x, y, labels, subject = next(d_)
            except:
                d_ = iter(datal)
                x, y, labels, subject = next(d_)
                
            if y.sum()==0:
                continue
            myo_mask = x.clone()
            myo_mask[myo_mask>0]=1

            optimizer.zero_grad()
            out = model.forward(x.to(DEVICE))
            y= torch.nan_to_num(y, nan=0.0)


            f_loss = loss_scale * torchvision.ops.sigmoid_focal_loss(out, y.to(DEVICE), reduction='mean')

            d_loss = weighted_dice_loss(loss_pre(out), y.to(DEVICE), myo_mask.to(DEVICE), w_fg=w_fg, w_bg=w_bg)
            
            loss= (f_loss + d_loss)

            focal_losses.append(f_loss.item())
            dice_losses.append(d_loss.item())

            loss.backward()
            optimizer.step()
            losses.append(loss.item())



        mean_focal = torch.mean(torch.tensor(focal_losses))
        mean_dice = torch.mean(torch.tensor(dice_losses))
        mean_total = torch.mean(torch.tensor(losses))

        print(f"Epoch {ep}: "
              f"focal={mean_focal:.5f}, "
              f"dice={mean_dice:.5f}, "
              f"total={mean_total:.5f}")
        

        if ep%20==0:
            torch.save(model.state_dict(), f'./results/models_{loss_type}_{args.model_type}/{ep}_'+model_save_path)
        torch.save(model.state_dict(), f'./results/models_{loss_type}_{args.model_type}/'+model_save_path)

    torch.save(model.state_dict(), f'./results/models_{loss_type}_{args.model_type}/'+model_save_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default = 36, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training Epochs")
    parser.add_argument("--stage", type=str, default = '0', help="Curriculum stage as string")
    parser.add_argument("--w_fg", type=float, default = 0.5, help="Weight on scar pixels in Dice loss")
    parser.add_argument("--model_type", type=str,  default='scarnet', help="Base model architecture")

    args = parser.parse_args()
    
    model_main(args)

if __name__ == "__main__":
    main()