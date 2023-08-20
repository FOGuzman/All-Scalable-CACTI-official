import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
   

def save_mp4(tensor,vid_name):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(vid_name, fourcc, 20.0, (tensor.shape[0],tensor.shape[1]))

    for fr in range(tensor.shape[2]):
        frame = (255*torch.unsqueeze(tensor[:,:,fr],-1).cpu().numpy()).astype(np.uint8)
        cframe = np.concatenate((frame,frame,frame),axis=2)
        out.write(cframe)
    out.release()


def load_checkpoints(model,pretrained_dict,strict=False):
    # pretrained_dict = torch.load(checkpoints)
    if strict is True:
        try: 
            model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)

        
        
def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,dim=1,keepdim=True)
    return y

def At(y,Phi):
    x = y*Phi
    return x



def expand_tensor(x):
    if x.dim() == 4:
        x= torch.squeeze(x)
    
    x = x.permute(1,2,0)
    h,w,ch = x.shape 
    
    for k in range(ch):
        if k == 0:
            out = x[:,:,k]
        else:
          out = torch.cat((out,x[:,:,k]),dim=1)
    
    return(out)    





def compute_ssim_psnr(original, reconstructed):
    # Calculate mean over spatial dimensions
    original_mean = original.mean(dim=(2, 3), keepdim=True)
    reconstructed_mean = reconstructed.mean(dim=(2, 3), keepdim=True)

    # Calculate variance and covariance
    original_var = ((original - original_mean) ** 2).mean(dim=(2, 3), keepdim=True)
    reconstructed_var = ((reconstructed - reconstructed_mean) ** 2).mean(dim=(2, 3), keepdim=True)
    covariance = ((original - original_mean) * (reconstructed - reconstructed_mean)).mean(dim=(2, 3), keepdim=True)

    # SSIM constants
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Calculate SSIM
    numerator = (2 * original_mean * reconstructed_mean + c1) * (2 * covariance + c2)
    denominator = (original_mean ** 2 + reconstructed_mean ** 2 + c1) * (original_var + reconstructed_var + c2)
    ssim_score = numerator / denominator

    # Average SSIM over channels and batches
    ssim_score = ssim_score.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)

    # Compute PSNR
    mse = torch.mean((original - reconstructed) ** 2)
    psnr = 10 * torch.log10(1 / mse)

    return ssim_score.item(), psnr.item()



def plot(x):
    plt.imshow(x, cmap='gray')
    plt.show(block=False)
    plt.pause(0.1)


def implay(x):
    image = None
    for idx in range(x.shape[2]):
        if image is None:
             image = plt.imshow(x[:,:,idx], cmap='gray')
        else:
             image.set_data(x[:,:,idx])
        plt.show(block=False)
        plt.pause(0.1)


def dismantleMeas(xin,order,args):
    meas_batch = torch.zeros((args.spix**2,1,args.resolution[0]//args.spix,args.resolution[1]//args.spix),device=args.device)
    for kx in range(args.spix**2):
        (x,y) = np.where(order==kx)
        meas_batch[kx] = xin[:,:,int(x):args.resolution[0]:args.spix,int(y):args.resolution[1]:args.spix]
    meas_batch2 = []
    meas_batch2.append(meas_batch[:,:,0:256,0:256])
    meas_batch2.append(meas_batch[:,:,0:256,256:512])   
    meas_batch2.append(meas_batch[:,:,256:512,0:256]) 
    meas_batch2.append(meas_batch[:,:,256:512,256:512])
    return meas_batch2 


def assemblyMeas(demul_tensor,order,kernels,args):
    TM_tensor11 = torch.zeros((args.resolution[0]//2,args.resolution[1]//2,args.spix**2*args.frames))
    cont = 0
    for kx in range(len(demul_tensor[0])):
        (x,y) = np.where(order==kx)
        for ts in range(args.frames):
                span = np.kron(demul_tensor[0][kx][ts],kernels[:,:,kx])
                TM_tensor11[:,:,cont] = torch.from_numpy(span)
                cont += 1

    TM_tensor12 = torch.zeros((args.resolution[0]//2,args.resolution[1]//2,args.spix**2*args.frames))
    cont = 0
    for kx in range(len(demul_tensor[1])):
        (x,y) = np.where(order==kx)
        for ts in range(args.frames):
                span = np.kron(demul_tensor[1][kx][ts],kernels[:,:,kx])
                TM_tensor12[:,:,cont] = torch.from_numpy(span)
                cont += 1

    TM_tensor21 = torch.zeros((args.resolution[0]//2,args.resolution[1]//2,args.spix**2*args.frames))
    cont = 0
    for kx in range(len(demul_tensor[2])):
        (x,y) = np.where(order==kx)
        for ts in range(args.frames):
                span = np.kron(demul_tensor[2][kx][ts],kernels[:,:,kx])
                TM_tensor21[:,:,cont] = torch.from_numpy(span)
                cont += 1   

    TM_tensor22 = torch.zeros((args.resolution[0]//2,args.resolution[1]//2,args.spix**2*args.frames))
    cont = 0
    for kx in range(len(demul_tensor[3])):
        (x,y) = np.where(order==kx)
        for ts in range(args.frames):
                span = np.kron(demul_tensor[3][kx][ts],kernels[:,:,kx])
                TM_tensor22[:,:,cont] = torch.from_numpy(span)
                cont += 1                       


    Full_TM = torch.cat((torch.cat((TM_tensor11,TM_tensor12),dim=1),torch.cat((TM_tensor21,TM_tensor22),dim=1)),dim=0)
    return Full_TM






def generate_compressed_coordinates(batch_size,size1,frames):
    # Generate random XY coordinates
    xy_coords = torch.randint(0, size1, (1, batch_size, 2))
    
    # Expand XY coordinates to create the initial tensor
    initial_tensor = xy_coords.unsqueeze(2).expand(1, batch_size, frames, 2)
    
    # Generate third coordinate
    third_coords = torch.arange(0, frames).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(1, batch_size, frames, 1)
    
    # Concatenate XY and third coordinates
    final_tensor = torch.cat((initial_tensor, third_coords), dim=-1)
    
    # Reshape to get the final tensor shape (1, batch_size*15, 3)
    final_tensor = final_tensor.view(1, batch_size*frames, 3)
    
    return final_tensor