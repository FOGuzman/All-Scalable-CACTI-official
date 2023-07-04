import scipy.io as scio
from utils import A, At
import torch
import einops
import numpy as np


def build_mask(args):
    data = scio.loadmat(args.mask_path)
    mask = data['mask']   
    mask_s = np.sum(mask,2)
    Phi = einops.repeat(mask,'h w cr->b cr h w',b=args.batchSize).astype(float)
    Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=args.batchSize).astype(float)

    Phi = torch.from_numpy(Phi).to(args.device).float()
    Phi_s = torch.from_numpy(Phi_s).float().to(args.device).float()

    return Phi, Phi_s