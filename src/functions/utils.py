import torch

    
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