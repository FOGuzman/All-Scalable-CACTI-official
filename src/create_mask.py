import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from functions.utils import plot,implay
from functions.videoLoader import VideoFramesDataset
import argparse
from functions.savers import save_as_mat_np, save_as_mp4_np, save_as_torch_np
import scipy.io as scio
import numpy as np



parser = argparse.ArgumentParser(description='Settings, Data agumentation')

parser.add_argument('--spix', default=4, type=int)
parser.add_argument('--nuclear_mask', default="./masks/mask512_8.mat", type=str)
parser.add_argument('--save_path', default="./masks/", type=str)
parser.add_argument('--color_mode', default="gray", type=str)
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--FileFormat', default="mat", type=str)
args = parser.parse_args()


mask_data = scio.loadmat(args.nuclear_mask)
mask = mask_data['mask'].astype(bool)

final_shape = tuple(np.array(mask.shape)*np.array((args.spix,args.spix,args.spix**2)))

out_mask = []

stacked_kernel = []
idxs = np.arange(0,args.spix**2).reshape(args.spix,args.spix)
for ki in range(idxs.shape[0]):
    if not(ki % 2 == 0):
        idxs[ki] = np.flip(idxs[ki])

for kx in range(args.spix**2):
    (x,y) = np.where(idxs==kx)
    kernel = np.zeros((args.spix,args.spix),dtype=bool)
    kernel[x,y] = True
    stacked_kernel.append(kernel)

stacked_kernel = np.stack(stacked_kernel,axis=2).astype(bool)



for mk in range(args.spix**2):
    for fm in range(mask.shape[2]):
        out_mask.append(np.kron(mask[:,:,fm],stacked_kernel[:,:,mk]))

out_mask = np.stack(out_mask,axis=2)       

file_name = os.path.join(args.save_path, f"mask_spix{args.spix}_nucmas({mask.shape[0]},{mask.shape[1]},{mask.shape[2]}).{args.FileFormat}")
if args.FileFormat == "mat":
    save_as_mat_np(out_mask, file_name,"mask")

elif args.FileFormat == "torch":
    save_as_torch_np(out_mask, file_name)

elif args.FileFormat == "mp4":
    save_as_mp4_np(out_mask, file_name)