import torch
import argparse
import scipy.io as scio
from functions.model import STFormer
from functions.prepare_mask import build_mask
from functions.utils import load_checkpoints
from functions.utils import compute_ssim_psnr
from functions.utils import implay,plot


parser = argparse.ArgumentParser(description='Settings, Data agumentation')

parser.add_argument('--test_data', default="./dataset_mat/train/tensor_1.mat", type=str)
parser.add_argument('--mask_path', default="./masks/xinyuan_mask.mat", type=str)
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--checkpoint', default='./checkpoints/stformer_base.pth', type=str)

args = parser.parse_args()
args.device = torch.device(args.device)




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




data_in = scio.loadmat(args.test_data)
gt = data_in['tensor']
gt = torch.from_numpy(gt).float().cuda()
gt = torch.unsqueeze(gt,0)

meas = torch.sum(gt*args.Phi,dim=1,keepdim=True)

with torch.no_grad():
    out = model(meas,args.Phi,args.Phi_s)
out = out[0]


compute_ssim_psnr(gt, out)

implay(out[0].permute(1,2,0).cpu().numpy())