import torch
import torch.nn as nn
import argparse
import scipy.io as scio
from functions.model import STFormer
from functions.prepare_mask import build_mask
from functions.utils import load_checkpoints
from functions.utils import compute_ssim_psnr
from functions.utils import dismantleMeas,assemblyMeas
from functions.siren_utils import *
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
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--spix', default=4, type=int)
parser.add_argument('--resolution', default=[2048,2048], type=eval, help='Dataset resolution')
parser.add_argument('--crop_size', default=[2048,2048], type=eval, help='Dataset resolution []')
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--checkpoint', default='./checkpoints/stformer_base.pth', type=str)
parser.add_argument('--FrameStart',type=int, default=3,help='data agumentation')

## Implicit representation arguments
parser.add_argument('--frame_skip', default=3, type=int)
parser.add_argument('--color_mode', default="gray", type=str)
parser.add_argument('--randomFlip', default=False, action="store_true",help='data agumentation')
parser.add_argument('--randomRotation', default=False, action="store_true",help='data agumentation')

args = parser.parse_args()
args.device = torch.device(args.device)


save_path = "./implicit_results/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset = VideoFramesDataset(args.test_data,args.frames,args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
full_mask_data = scio.loadmat(args.fullmask_path)
full_mask = torch.from_numpy(full_mask_data['mask'])
full_mask = torch.unsqueeze(full_mask.permute(2,0,1),0)
order = full_mask_data['order']
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

    # stack_im = torch.cat((iframe[0,0,0:32,0:32],modulated_frame[0,0,0:32,0:32],meas[0,0,0:32,0:32]/5),dim=1)
    # if imsh is None:
    #     imsh = plt.imshow(stack_im, cmap='gray')
    # else:
    #     imsh.set_data(stack_im)
    # plt.show(block=False)
    # plt.pause(0.01)    


## Decompose measurement according to spix
meas_batch = dismantleMeas(meas,order,args)

args.Phi, args.Phi_s = build_mask(args,args.mask_path,args.batchSize)
args.Phi_test, args.Phi_s_test = build_mask(args,args.mask_path,1)

model = STFormer(color_channels=1,units=4,dim=64,frames=args.frames)
model = model.to(args.device)

if args.checkpoint is not None:
    print("Load pre_train model...")
    resume_dict = torch.load(args.checkpoint)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict)
else:            
    print("No pre_train model")


demul_tensor = []
with torch.no_grad():
    for bt in range(len(meas_batch)):
        tm_tensor = []
        for sb in range(meas_batch[bt].shape[0]):
            out = model(meas_batch[bt][sb:sb+1],args.Phi,args.Phi_s)
            out = out[0][0].cpu()
            tm_tensor.append(out)
        demul_tensor.append(tm_tensor)    


kernels = torch.from_numpy(full_mask_data['kernel'])
                     
Full_TM = assemblyMeas(demul_tensor,order,kernels,args)


coords = torch.unsqueeze(torch.nonzero(Full_TM).float().cuda(),0)
gt = torch.unsqueeze(torch.unsqueeze(Full_TM[Full_TM.nonzero(as_tuple=True)].float().cuda(),0),-1)

## Prepare coords
coords[:,:,0] = (coords[:,:,0]/(args.resolution[0]-1)*2)-1
coords[:,:,1] = (coords[:,:,1]/(args.resolution[0]-1)*2)-1
coords[:,:,2] = (coords[:,:,2]/(args.frames*args.spix**2-1)*2)-1




## Purge memory
del model,tm_tensor, demul_tensor, dataset, dataloader, iframe, meas_batch, resume_dict, model_state_dict
del modulated_frame, out
args.Phi,args.Phi_test,args.Phi_s,args.Phi_s_test = None,None,None,None
torch.cuda.empty_cache()
gc.collect()


## IMplicit fit
siren_model = Siren(in_features=3, out_features=1, hidden_features=256, 
                  hidden_layers=3, outermost_linear=True)
siren_model.cuda()
lossfn = nn.MSELoss() 
lossfn = lossfn.cuda()
optim = torch.optim.AdamW(lr=1e-4, params=siren_model.parameters())
scheduler = opt.lr_scheduler.StepLR(optim, step_size=500, gamma=0.9)
# pr



batch_size = 80000


for iter in range(2000):
    batch_idx = torch.randperm(coords.shape[1])[0:batch_size]
    model_output, coords_out = siren_model(coords[:,batch_idx,:])
    loss = lossfn(gt[:,batch_idx,:],model_output)
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()
    print(f"iter: {iter} - loss: {loss.item()}")


del coords,gt, loss, optim, scheduler, batch_idx, coords_out
torch.cuda.empty_cache()
gc.collect()

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

torch.save(siren_model,save_path + "clip_5.pth")

implay(ten_out.cpu().numpy())# 0.00687