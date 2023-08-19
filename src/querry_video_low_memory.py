import torch
import torch.nn as nn
import argparse
import scipy.io as scio
from functions.model import STFormer
from functions.prepare_mask import build_mask
from functions.utils import load_checkpoints
from functions.utils import compute_ssim_psnr
from functions.utils import dismantleMeas,assemblyMeas
#from functions.siren_utils import *
from functions.model_siren import Siren
from functions.utils import implay,plot
from torch.utils.data import DataLoader
from functions.videoLoaderSingle import VideoFramesDataset
import torch.optim as opt
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os

parser = argparse.ArgumentParser(description='Settings, Data agumentation')

## Demultiplexing arguments
parser.add_argument('--siren_video', default="./implicit_results/clip_5.pth", type=str)
parser.add_argument('--mask_path', default="./masks/xinyuan_mask.mat", type=str)
parser.add_argument('--fullmask_path', default="./masks/mask_spix4_nucmas(512,512,8).mat", type=str)
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--spix', default=4, type=int)
parser.add_argument('--resolution', default=[2048,2048], type=eval, help='Dataset resolution')
parser.add_argument('--crop_size', default=[2048,2048], type=eval, help='Dataset resolution []')
parser.add_argument('--siren_batch_size', default=150000, type=int)
args = parser.parse_args()


## IMplicit fit
siren_model = Siren(in_features=3, out_features=1, hidden_features=512, 
                  hidden_layers=4, outermost_linear=True)
siren_model.cuda()

siren_model = torch.load(args.siren_video).cuda()
# pr



batch_size = args.siren_batch_size

# Full 
xv = torch.linspace(-1,-1,args.resolution[0])
yv = torch.linspace(-1,-1,args.resolution[1])
tv = torch.linspace(-1,-1,args.frames*args.spix**2)


out_tensor = torch.zeros((args.resolution[0],args.resolution[1],args.frames*args.spix**2),dtype=torch.float)
with torch.no_grad():
    for xi in tqdm(xv,
                                 desc ="Querrying... ",colour="red",
                                 total=len(xv),
                                 ascii=' 123456789═'):
            for yi in yv:
                 for ti in tv:
                    coord_in = torch.unsqueeze(torch.unsqueeze(torch.stack([xi,yi,ti]),0),0).float().cuda() 
                    pred_img, coords_out = siren_model(coord_in) # (1, 3, 256, 256)

                    xidx = int((xi+1)/2*(args.resolution[0]-1))
                    yidx = int((yi+1)/2*(args.resolution[0]-1))
                    tidx = int((yi+1)/2*(args.frames*args.spix**2-1))
                    out_tensor[xidx,yidx,tidx] = pred_img.cpu()
 

ten_out = torch.reshape(out_x2[:,:,0],(args.resolution[0],args.resolution[0],args.frames*args.spix**2))

implay(ten_out.cpu().numpy())# 0.00687