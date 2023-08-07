import os
import os.path as osp
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from functions.model import STFormer
from torch.utils.tensorboard import SummaryWriter
from functions.loader import Imgdataset
from functions.prepare_mask import build_mask
from functions.utils import A, At, expand_tensor
from functions.utils import compute_ssim_psnr

parser = argparse.ArgumentParser(description='Settings, Data agumentation')

parser.add_argument('--videos_dir', default="./clips/", type=str)
parser.add_argument('--training_dir', default="./dataset/train/", type=str)
parser.add_argument('--validation_dir', default="./dataset/val/", type=str)
parser.add_argument('--mask_path', default="./masks/mask128_8.mat", type=str)
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--batchSize', default=2, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--checkpoint', default='./checkpoints/stformer_base.pth', type=str)
parser.add_argument('--experimentName', default="exp_exmaple", type=str)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--saveModelEach', default=5, type=int, help='Number of epochs')

args = parser.parse_args()
args.device = torch.device(args.device)

tb_path = "./training_results/" + "tensorboard/" + args.experimentName 
check_path = "./training_results/" + args.experimentName + "/"

if not os.path.exists(tb_path):
    os.makedirs(tb_path)
if not os.path.exists(check_path):
    os.makedirs(check_path)

writer = SummaryWriter(log_dir = tb_path)



args.Phi, args.Phi_s = build_mask(args,args.batchSize)
args.Phi_test, args.Phi_s_test = build_mask(args,1)

lr = args.learning_rate

model = STFormer(color_channels=1,units=2,dim=32,frames=args.frames)
model = model.to(args.device)
loss = torch.nn.MSELoss()
loss = loss.to(args.device)

optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=lr)


train_dataset = Imgdataset(args.training_dir)
test_dataset = Imgdataset(args.validation_dir)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batchSize, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

iter_num = len(train_data_loader)
for epoch in range(args.Epochs):
    epoch_loss = 0
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=lr)
    model = model.train()
    start_time = time.time()
    for iteration, data in tqdm(enumerate(train_data_loader),
                                desc ="Training... ",colour="red",
                                total=iter_num,
                                ascii=' 123456789═'):
        gt = data.to(args.device)

        meas = A(gt,args.Phi)

        optimizer.zero_grad()

        out = model(meas,args.Phi,args.Phi_s)
        out = out[0]

        loss_val = loss(out, gt)
        loss_nm  = loss_val.item()
        epoch_loss += loss_nm

        loss_val.backward()
        optimizer.step()


        if (iteration % 100) == 0:
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            iter_len = len(str(iter_num))
            print("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss_nm))
            writer.add_scalar("loss",loss_nm,epoch*len(train_data_loader) + iteration)
            
    end_time = time.time()   

    print("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

    if (epoch % args.saveModelEach) == 0:

        save_model = model
        checkpoint_dict = {
            "epoch": epoch, 
            "model_state_dict": save_model.state_dict(), 
            "optim_state_dict": optimizer.state_dict(), 
        }
        torch.save(checkpoint_dict,osp.join(check_path,"epoch_"+str(epoch)+".pth")) 

    if (epoch % 5) == 0:
        lr = lr*0.9

    
    # Validation
    model.eval()
    ssim_vec,psnr_vec = [],[]
    for iteration, data in tqdm(enumerate(test_data_loader),
                            desc ="Validation... ",colour="green",
                            total=len(test_data_loader),
                            ascii=' 123456789═'):
        
        with torch.no_grad():
            gt = data.to(args.device)

            meas = A(gt,args.Phi_test)

            optimizer.zero_grad()

            out = model(meas,args.Phi_test,args.Phi_s_test)
            out = out[0]

            ssim_val, psnr_val = compute_ssim_psnr(gt, out)

            ssim_vec.append(ssim_val)
            psnr_vec.append(psnr_val)
        
            
    vid_gt = expand_tensor(gt)
    vid_re = expand_tensor(out)
    vid_mk = expand_tensor(args.Phi_test)
    tensor_im = torch.cat((vid_gt,vid_re,vid_mk),dim=0)
    mean_ssim = sum(ssim_vec)/len(ssim_vec)
    mean_psnr = sum(psnr_vec)/len(psnr_vec)
    writer.add_scalar("PSNR",mean_psnr,epoch)
    writer.add_scalar("SSIM",mean_ssim,epoch)
    writer.add_image("Val recon",tensor_im,epoch,dataformats='HW')
    print(f"Validation results: mean SSIM: {mean_ssim:.4f} | mean PSNR: {mean_psnr:.2f} dB.")
    
               
writer.close()