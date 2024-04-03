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
from functions.utils import implay,plot, save_mp4
from torch.utils.data import DataLoader
from functions.videoLoaderSingle import VideoFramesDataset
import torch.optim as opt
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os

parser = argparse.ArgumentParser(description='Settings, Data agumentation')

## Demultiplexing arguments
parser.add_argument('--test_data', default="./clips/clip 5.mp4", type=str)
parser.add_argument('--siren_video', default="./implicit_results/clip_5.pth", type=str)
parser.add_argument('--mask_path', default="./masks/xinyuan_mask.mat", type=str)
parser.add_argument('--fullmask_path', default="./masks/mask_spix4_nucmas(512,512,8).mat", type=str)
parser.add_argument('--frames', default=128, type=int)
parser.add_argument('--spix', default=1, type=int)
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')

parser.add_argument('--resolution', default=[2048,2048], type=eval, help='Dataset resolution')
parser.add_argument('--crop_size', default=[2048,2048], type=eval, help='Dataset resolution []')
parser.add_argument('--siren_batch_size', default=15000, type=int)
parser.add_argument('--FrameStart',type=int, default=3,help='data agumentation')
parser.add_argument('--frame_skip', default=3, type=int)
parser.add_argument('--color_mode', default="gray", type=str)
parser.add_argument('--randomFlip', default=False, action="store_true",help='data agumentation')
parser.add_argument('--randomRotation', default=False, action="store_true",help='data agumentation')

args = parser.parse_args()
args.device = torch.device(args.device)

## IMplicit fit
#siren_model = Siren(in_features=3, out_features=1, hidden_features=512, 
#                  hidden_layers=4, outermost_linear=True)
#siren_model.cuda()

siren_model = torch.load(args.siren_video).cuda()
# pr



batch_size = args.siren_batch_size

# Full 
coords_x2 = torch.unsqueeze(torch.nonzero(torch.ones((args.resolution[0],args.resolution[0],args.frames*args.spix**2))),0).float()
coords_x2[:,:,0] = (coords_x2[:,:,0]/(args.resolution[0]-1)*2)-1
coords_x2[:,:,1] = (coords_x2[:,:,1]/(args.resolution[0]-1)*2)-1
coords_x2[:,:,2] = (coords_x2[:,:,2]/(args.frames*args.spix**2-1)*2)-1

coords_x2 = torch.tensor_split(coords_x2,coords_x2.shape[1]//batch_size,dim=1)
out_x2 = None
with torch.no_grad():
    for k in tqdm(range(len(coords_x2)),
                                 desc ="Querrying... ",colour="red",
                                 total=len(coords_x2),
                                 ascii=' 123456789═'):
        pred_img, coords_out = siren_model(coords_x2[k].float().cuda()) # (1, 3, 256, 256)
        if out_x2 is None:
            out_x2 = pred_img.cpu()
        else:
            out_x2 = torch.cat((out_x2,pred_img.cpu()),dim=1)

ten_out = torch.reshape(out_x2[:,:,0],(args.resolution[0],args.resolution[0],args.frames*args.spix**2))



dataset = VideoFramesDataset(args.test_data,args.frames,args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
full_mask_data = scio.loadmat(args.fullmask_path)
full_mask = torch.from_numpy(full_mask_data['mask'])
full_mask = torch.unsqueeze(full_mask.permute(2,0,1),0)
order = full_mask_data['order']
meas = torch.zeros(size=(1,1,args.resolution[0],args.resolution[1]))
imsh = None
for k in tqdm(range(128),
                                 desc ="Generating measurement... ",colour="red",
                                 total=128,
                                 ascii=' 123456789═'):
    # Get one batch from the dataloader
    iframe = next(iter(dataloader),k)
    modulated_frame = compute_ssim_psnr(iframe,ten_out[:,k:k+1])
    meas = meas + modulated_frame