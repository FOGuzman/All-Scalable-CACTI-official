import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as transforms
import numpy as np


class VideoFramesDataset(Dataset):
    def __init__(self, video_dir,frames, args):
        if video_dir[-4:] == ".mp4":
            self.video_paths = video_dir
        else:
            self.video_paths = glob.glob(video_dir + "/*.mp4")
        self.n_frames = frames
        self.frame_count_list = []
        self.color_mode = args.color_mode
        self.resolution = args.resolution
        self.crop_size = args.crop_size
        self.spix = args.spix
        self.randomFlip = args.randomFlip
        self.randomRotation = args.randomRotation
        self.FrameStart = args.FrameStart
        self.frame_skip = args.frame_skip
        self.device = args.device
        self.idx  = 0
        self.cropflag = 0
        self.x = 0
        self.y = 0
        
        if args.color_mode == "gray":
            self.grayscale_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])

        if args.randomRotation:
            self.rotation_transforms = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            #transforms.ToTensor(),
        ])


        cap = cv2.VideoCapture(self.video_paths)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count_list.append(frame_count)
        cap.release()    

    def __len__(self):
        return sum(self.frame_count_list)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range.")
        
        video_path = self.video_paths
        


        cap = cv2.VideoCapture(video_path)
        vid_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        init_rand_limit = np.arange(vid_total_frames,
                                    vid_total_frames-self.n_frames*self.frame_skip*(self.spix**2),step=-self.frame_skip)-1
        
        if self.FrameStart>init_rand_limit[-1]:
            frame_idx = random.randint(0, init_rand_limit[-1])
        else:
            frame_idx = self.FrameStart
        
        frame_idx_list = np.arange(0,self.n_frames*self.frame_skip*(self.spix**2),step=self.frame_skip)
        frame_idx_list = frame_idx_list+frame_idx

        frames = []
        flag = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_list[self.idx])
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to read frame.")
  
        self.idx += 1


        # Croping
        h, w, _ = frame.shape
        if self.cropflag == 0:
            self.x = random.randint(0, w - self.crop_size[0])
            self.y = random.randint(0, h - self.crop_size[1])
            self.cropflag = 1

        process_frame = frame[self.y:self.y + self.crop_size[0], self.x:self.x + self.crop_size[1]]
        # Convert frame to PyTorch tensor

        # Monochrome or RGB
        if self.color_mode == "gray":
            process_frame = self.grayscale_transforms(process_frame)              
            
        tensor_vid = process_frame

        cap.release()

        # Random ratation
        if self.randomRotation:
            rot_ang = random.choice([0, 90, 180, 270])
            rand_flip = random.choice([0,1,2])

            if rand_flip == 1:
                tensor_vid = transforms.functional.hflip(tensor_vid)
            if rand_flip == 2:
                tensor_vid = transforms.functional.vflip(tensor_vid)

            tensor_vid = transforms.functional.rotate(tensor_vid,rot_ang)

        out_vid = transforms.Resize(size=self.resolution)(tensor_vid)

        return out_vid