import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from functions.videoLoader import VideoFramesDataset
import argparse
from functions.savers import save_as_mat, save_as_mp4, save_as_torch

parser = argparse.ArgumentParser(description='Settings, Data agumentation')

parser.add_argument('--videos_dir', default="./clips/", type=str)
parser.add_argument('--augmented_data_dir', default="./dataset/", type=str)
parser.add_argument('--frames', default=8, type=int)
parser.add_argument('--frame_skip', default=8, type=int)
parser.add_argument('--resolution', default=[128,128], type=eval, help='Dataset resolution')
parser.add_argument('--crop_size', default=[2048,2048], type=eval, help='Dataset resolution []')
parser.add_argument('--color_mode', default="gray", type=str)
parser.add_argument('--randomFlip', default=True, action="store_true",help='data agumentation')
parser.add_argument('--randomRotation', default=True, action="store_true",help='data agumentation')
parser.add_argument('--batchSize', default=5, type=int, help='Batch size for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device choice (cpu or cuda)')

parser.add_argument('--trainData', default=20, type=int)
parser.add_argument('--validationData', default=20, type=int)
parser.add_argument('--FileFormat', default="torch", type=str)
args = parser.parse_args()

args.device = torch.device(args.device)

# Check if CUDA is available and the device is set to CUDA
if args.device.type == 'cuda' and not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please select the 'cpu' device.")

train_path = args.augmented_data_dir + "/train/"
val_path   = args.augmented_data_dir + "/val/"
if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(val_path):
    os.makedirs(val_path)
# Create an instance of the dataset
dataset = VideoFramesDataset(args.videos_dir,args)

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

## Train data
cont = 1
for k in tqdm(range(args.trainData//args.batchSize),
                                 desc ="Generate training data... ",colour="red",
                                 total=args.trainData//args.batchSize,
                                 ascii=' 123456789═'):
    # Get one batch from the dataloader
    batch = next(iter(dataloader))

    for i, tensor in enumerate(batch):
        file_name = os.path.join(train_path, f"tensor_{cont}.{args.FileFormat}")

        if args.FileFormat == "mat":
            save_as_mat(tensor, file_name)

        elif args.FileFormat == "torch":
            save_as_torch(tensor, file_name)

        elif args.FileFormat == "mp4":
            save_as_mp4(tensor, file_name)

        cont += 1



## Validation
cont = 1
for k in tqdm(range(args.validationData//args.batchSize),
                                 desc ="Generate validation data... ",colour="green",
                                 total=args.trainData//args.batchSize,
                                 ascii=' 123456789═'):
    # Get one batch from the dataloader
    batch = next(iter(dataloader))

    for i, tensor in enumerate(batch):
        file_name = os.path.join(val_path, f"tensor_{cont}.{args.FileFormat}")

        if args.FileFormat == "mat":
            save_as_mat(tensor, file_name)

        elif args.FileFormat == "torch":
            save_as_torch(tensor, file_name)

        elif args.FileFormat == "mp4":
            save_as_mp4(tensor, file_name)

        cont += 1