import os
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
from functions.utils import A, At

parser = argparse.ArgumentParser(description='Settings, Data agumentation')

parser.add_argument('--training_dir', default="./dataset/train/", type=str)
parser.add_argument('--validation_dir', default="./dataset/val/", type=str)
parser.add_argument('--mask_path', default="./masks/mask512_8.mat", type=str)
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--experimentName', default="exp_exmaple", type=str)
parser.add_argument('--learning_rate', default=0.0001, type=float)

args = parser.parse_args()
args.device = torch.device(args.device)

tb_path = "./training_results/" + "tensorboard/" + args.experimentName 
check_path = "./training_results/" + args.experimentName + "/"
writer = SummaryWriter(log_dir = tb_path)



args.Phi, args.Phi_s = build_mask(args)


lr = args.learning_rate

model = STFormer(color_channels=1,units=4,dim=64,frames=args.frames)
model = model.to(args.device)
loss = torch.nn.MSELoss()
loss = loss.to(args.device)

optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=lr)


train_dataset = Imgdataset(args.training_dir)
test_dataset = Imgdataset(args.validation_dir)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batchSize, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batchSize, shuffle=True)

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

            model_out = model(meas,args)
            

            loss_val = loss(model_out, gt)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


            if (iteration % 100) == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                iter_len = len(str(iter_num))
                print("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss.item()))
                writer.add_scalar("loss",loss.item(),epoch*len(train_data_loader) + iteration)
                
        end_time = time.time()

        

        print("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

        if (epoch % args.saveModelEach) == 0:
 
            save_model = model
            checkpoint_dict = {
                "epoch": epoch, 
                "model_state_dict": save_model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(), 
            }
            torch.save(checkpoint_dict,osp.join(paths.trained_model_path,"epoch_"+str(epoch)+".pth")) 
