import torch
import torch.nn as nn
import argparse
import scipy.io as scio
from functions.model import STFormer
from functions.prepare_mask import build_mask
from functions.utils import load_checkpoints
from functions.utils import compute_ssim_psnr,generate_compressed_coordinates
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
parser.add_argument('--test_data', default="./clips/clip 5.mp4", type=str)
parser.add_argument('--mask_path', default="./masks/xinyuan_mask.mat", type=str)
parser.add_argument('--fullmask_path', default="./masks/mask_spix4_nucmas(512,512,8).mat", type=str)
parser.add_argument('--frames', default=2, type=int)
parser.add_argument('--spix', default=1, type=int)
parser.add_argument('--resolution', default=[256,256], type=eval, help='Dataset resolution')
parser.add_argument('--crop_size', default=[2048,2048], type=eval, help='Dataset resolution []')
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--checkpoint', default='./checkpoints/stformer_base.pth', type=str)
parser.add_argument('--FrameStart',type=int, default=3,help='data agumentation')
parser.add_argument('--frame_skip', default=3, type=int)
parser.add_argument('--color_mode', default="gray", type=str)
parser.add_argument('--randomFlip', default=False, action="store_true",help='data agumentation')
parser.add_argument('--randomRotation', default=False, action="store_true",help='data agumentation')

## Implicit representation arguments
parser.add_argument('--siren_batch_size', default=500, type=int)
parser.add_argument('--siren_iterations', default=10000, type=int)

args = parser.parse_args()
args.device = torch.device(args.device)


save_path = "./implicit_results/"
if not os.path.exists(save_path):
    os.makedirs(save_path)


total_frame = args.frames*args.spix**2
batch_size = args.siren_batch_size

dataset = VideoFramesDataset(args.test_data,args.frames,args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
full_mask_data = scio.loadmat(args.fullmask_path)
full_mask = (torch.rand((args.resolution[0],args.resolution[1],total_frame))>0.5)*1
full_mask = torch.unsqueeze(full_mask.permute(2,0,1),0)
meas = torch.zeros(size=(1,1,args.resolution[0],args.resolution[1]))
imsh = None
for k in tqdm(range(full_mask.shape[1]),
                                 desc ="Generating measurement... ",colour="red",
                                 total=full_mask.shape[1],
                                 ascii=' 123456789═'):
    # Get one batch from the dataloader
    iframe = next(iter(dataloader),k)
    modulated_frame = iframe*full_mask[:,k:k+1]
    meas = meas + modulated_frame
   


## IMplicit fit
siren_model = Siren(in_features=3, out_features=1, hidden_features=512, 
                  hidden_layers=4, outermost_linear=True)
siren_model.cuda()
lossfn = nn.MSELoss() 
lossfn = lossfn.cuda()
optim = torch.optim.AdamW(lr=1e-4, params=siren_model.parameters())
scheduler = opt.lr_scheduler.StepLR(optim, step_size=500, gamma=0.9)
# pr
for iter in range(args.siren_iterations):
    coords = generate_compressed_coordinates(batch_size,args.resolution[0],total_frame)
    idxXY = torch.squeeze(coords.clone().detach())[0:total_frame*batch_size:total_frame]
    coords = coords.float().cuda()
    coords[:,:,0] = (coords[:,:,0]/(args.resolution[0]-1)*2)-1
    coords[:,:,1] = (coords[:,:,1]/(args.resolution[0]-1)*2)-1
    coords[:,:,2] = (coords[:,:,2]/((total_frame))*2)-1

    model_output, coords_out = siren_model(coords.cuda())
    compressed_filaments = torch.tensor_split(model_output,batch_size,dim=1)

    loss_batch = []
    for mm in range(batch_size):
        filament_mask = full_mask[:,:,idxXY[mm,0],idxXY[mm,1]].unsqueeze(-1).cuda().float()
        syntethic_meas = torch.sum(filament_mask*compressed_filaments[mm])
        real_meas_vector = meas[0,0,idxXY[mm,0],idxXY[mm,1]].cuda().float()
        loss_batch.append(lossfn(real_meas_vector,syntethic_meas))
    loss_batch = torch.mean(torch.stack(loss_batch))
    optim.zero_grad()
    loss_batch.backward()
    optim.step()
    scheduler.step()
    print(f"iter: {iter} - loss: {loss_batch.item()}")


torch.save(siren_model,save_path + "clip_5.pth")

del coords, optim, scheduler, coords_out
torch.cuda.empty_cache()
gc.collect()

# Full 
coords_x2 = torch.unsqueeze(torch.nonzero(torch.ones((args.resolution[0],args.resolution[0],args.frames*args.spix**2))),0).float()
coords_x2[:,:,0] = (coords_x2[:,:,0]/(args.resolution[0]-1)*2)-1
coords_x2[:,:,1] = (coords_x2[:,:,1]/(args.resolution[0]-1)*2)-1
coords_x2[:,:,2] = (coords_x2[:,:,2]/(total_frame)*2)-1

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



implay(ten_out.cpu().numpy())# 0.00687