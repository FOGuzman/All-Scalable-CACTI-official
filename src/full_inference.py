import torch
import torch.nn as nn
import argparse
import scipy.io as scio
from functions.model import STFormer
from functions.prepare_mask import build_mask
from functions.utils import load_checkpoints
from functions.utils import compute_ssim_psnr
from functions.utils import implay
from functions.siren_utils import *
from functions.model_siren import Siren
from functions.utils import implay,plot
import torch.optim as opt
parser = argparse.ArgumentParser(description='Settings, Data agumentation')

parser.add_argument('--test_data', default="./dataset_mat/train/tensor_1.mat", type=str)
parser.add_argument('--mask_path', default="./masks/xinyuan_mask.mat", type=str)
parser.add_argument('--fullmask_path', default="./masks/mask_spix4_nucmas(512,512,8).mat", type=str)
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--checkpoint', default='./checkpoints/stformer_base.pth', type=str)

args = parser.parse_args()
args.device = torch.device(args.device)




args.Phi, args.Phi_s = build_mask(args,args.batchSize)
args.Phi_test, args.Phi_s_test = build_mask(args,1)

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



siren_model = Siren(in_features=3, out_features=1, hidden_features=256, 
                  hidden_layers=3, outermost_linear=True)
siren_model.cuda()

lossfn = nn.MSELoss() 
lossfn = lossfn.cuda()
optim = torch.optim.AdamW(lr=1e-4, params=siren_model.parameters())
scheduler = opt.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.9)
latent = nn.Parameter(torch.zeros(512).normal_(0, 1)).cuda()



data_in = scio.loadmat(args.test_data)
gt = data_in['tensor']
gt = torch.from_numpy(gt).float().cuda()
gt = torch.unsqueeze(gt,0)

meas = torch.sum(gt*args.Phi,dim=1,keepdim=True)

with torch.no_grad():
    out = model(meas,args.Phi,args.Phi_s)
out = out[0][0]


# pr
sir_in = out[:,0:256,0:256].permute(1,2,0)

Hs,Ws,Ch = sir_in.shape
# Implicit fit
x = np.linspace(-1, 1, Hs)
y = np.linspace(-1, 1, Hs)
t = np.linspace(-1,1,Ch)
xv, yv, t = np.meshgrid(x, y,t)

coords = np.stack((xv.flatten(),yv.flatten(),t.flatten()),axis=1)
coords = torch.unsqueeze(torch.from_numpy(coords),0).float().cuda()

gt_sir = torch.unsqueeze(torch.unsqueeze(sir_in.flatten(),0),-1)

batch_size = 60000


for iter in range(1000):
    batch_idx = torch.randperm(coords.shape[1])[0:batch_size]
    model_output, coords_out = siren_model(coords[:,batch_idx,:])
    loss = lossfn(gt_sir[:,batch_idx,:],model_output)
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()
    print(f"iter: {iter} - loss: {loss.item()}")

with torch.no_grad():
    pred_img, coords_out = siren_model(coords) # (1, 3, 256, 256)

ten_out = torch.reshape(pred_img[:,:,0],(256,256,8))

implay(ten_out.cpu().numpy())# 0.00687


x = np.linspace(-1, 1, Hs*2)
y = np.linspace(-1, 1, Hs*2)
t = np.linspace(-1,1,12)
xv, yv, t = np.meshgrid(x, y,t)

coords_x2 = np.stack((xv.flatten(),yv.flatten(),t.flatten()),axis=1)
coords_x2 = torch.unsqueeze(torch.from_numpy(coords_x2),0).float().cuda()

batch_img = torch.tensor_split(coords_x2,coords_x2.shape[1]//batch_size,dim=1)
out_x2 = None
with torch.no_grad():
    for k in range(len(batch_img)):
        pred_img, coords_out = siren_model(batch_img[k]) # (1, 3, 256, 256)
        if out_x2 is None:
            out_x2 = pred_img
        else:
            out_x2 = torch.cat((out_x2,pred_img),dim=1)

ten_out = torch.reshape(out_x2[:,:,0],(512,512,12))


implay(ten_out.cpu().numpy())# 0.00687