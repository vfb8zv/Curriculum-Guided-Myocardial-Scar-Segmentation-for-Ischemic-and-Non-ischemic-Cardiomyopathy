import scipy.io as sio
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import shutil, os, json
import torch, scipy
import random
from torch.utils.data import Dataset, Sampler, DataLoader
import argparse
import torch.nn.functional as F


from scipy import ndimage
import torchvision

DEVICE= 'cuda:0'

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


def acti_pred(pred):
    act= nn.Sigmoid().to(DEVICE)
    return act(pred)
    # return pred

def scar_dice(y_true_f, y_pred_f, myo_mask):
    fg_mask = myo_mask == 1
    y_true_f = y_true_f[fg_mask].view(-1)
    y_pred_f = y_pred_f[fg_mask].view(-1)
    return ((2 * (y_true_f[y_true_f==1] * y_pred_f[y_true_f==1]).sum()) / (y_true_f[y_true_f==1].sum() + y_pred_f[y_true_f==1].sum() + 1e-14))

def scar_burden_error(Y, Y_hat, M):    
    Y_myo = Y[M == 1]
    Yhat_myo = Y_hat[M == 1]
    
    gt_scar_vol = np.sum(Y_myo)
    pred_scar_vol = np.sum(Yhat_myo)
    total_myo_vol = np.sum(M)

    scar_burden_error = abs(gt_scar_vol - pred_scar_vol) / total_myo_vol * 100
    return scar_burden_error


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


def get_data(data_dir="./dataset/", test_flag=True, batch_size=2, image_size=64, stage='0', shuffle=False):
    ds = DataLoading(data_dir, test_flag=test_flag, image_size=image_size)
    datal = DataLoader(ds)
    return datal



# Import desired model

# from models_scar.scarnet import ScarNet 
# from get_net import get_transunet
from attention_model import AttentionUNet


def model_main(args):

    loss_type= 'cl'
    DEVICE= 'cuda:0'
    batch_size = args.batch_size
    epochs= args.epochs
    model_save_path= f'{args.model_type}_{loss_type}_{epochs}.pt'

    os.makedirs('./fin_results/', exist_ok=True)

    datal= get_data('./dataset/', test_flag=True, batch_size=batch_size, image_size=224)
    d_= iter(datal)

    # select desired model

    # model = ScarNet(pretrained_path = './models_scar/medsam_vit_b.pth', num_classes=1).cuda()
    # model = get_transunet().cuda()
    model = AttentionUNet(drop_out_prob=0).cuda()

    model.load_state_dict(torch.load(f'./results/models_{loss_type}_{args.model_type}/'+model_save_path, map_location=DEVICE))
    model.eval();

    d_= iter(datal)
        
    slices=[]
    gt=[]
    subjects=[]
    scar_dices=[]
    burdens=[]
    pred=[]
    
    for _ in range(len(datal)):
        x, y, label, subject = next(d_)
 
        mask = x.clone()
        mask[mask>0]=1
        mask[mask<=0]=0
        y*=mask

        y_pred = model(x.to(DEVICE))
        
        y_pred= acti_pred(y_pred).detach().cpu()*mask
        y_pred[y_pred<0.5]=0
        y_pred[y_pred>=0.5]=1
        scar_dsc= scar_dice(y, y_pred, mask)

        if torch.sum(y)==0:
            continue
        else:
            scar_dices.append(scar_dsc)
            burdens.append(scar_burden_error(y.numpy(), y_pred.detach().cpu().numpy(), mask.numpy()))
            subjects.append(subject[0])
        
        slices.append(x.squeeze())
        y_gt_viz = y.detach().cpu().squeeze().clone().numpy()
        y_gt_viz[y_gt_viz==0]=np.nan
        gt.append(y_gt_viz)
        
        y_viz = y_pred.detach().cpu().squeeze().clone().numpy()
        y_viz[y_viz==0]=np.nan
        pred.append(y_viz)

    torch.save(pred, f'./fin_results/predictions_{loss_type}_{args.model_type}.pt')
    torch.save(gt, f'./fin_results/gt_{loss_type}_{args.model_type}.pt')
    torch.save(slices, f'./fin_results/slices_{loss_type}_{args.model_type}.pt')
    torch.save(subjects, f'./fin_results/pred_subjects_{loss_type}_{args.model_type}.pt')
    torch.save(scar_dices, f'./fin_results/dice_{loss_type}_{args.model_type}.pt')
    torch.save(burdens, f'./fin_results/burden_{loss_type}_{args.model_type}.pt')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default = 1, help="Batch size")
    parser.add_argument("--model_type", type=str,  default='scarnet', help="Base model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Training Epochs")

    args = parser.parse_args()
    model_main(args)

if __name__ == "__main__":
    main()




