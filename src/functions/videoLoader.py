import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as transforms
import numpy as np


class VideoFramesDataset(Dataset):
    def __init__(self, video_dir, args):
        self.video_paths = glob.glob(video_dir + "/*.mp4")
        self.n_frames = args.frames
        self.frame_count_list = []
        self.color_mode = args.color_mode
        self.resolution = args.resolution
        self.crop_size = args.crop_size
        self.randomFlip = args.randomFlip
        self.randomRotation = args.randomRotation
        self.frame_skip = args.frame_skip
        self.device = args.device
        
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

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count_list.append(frame_count - args.frames + 1)
            cap.release()    

    def __len__(self):
        return sum(self.frame_count_list)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range.")

        video_idx = 0
        frame_idx = idx

        for i, frame_count in enumerate(self.frame_count_list):
            if frame_idx < frame_count:
                video_idx = i
                break
            else:
                frame_idx -= frame_count

        video_path = self.video_paths[video_idx]
        


        cap = cv2.VideoCapture(video_path)
        vid_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        init_rand_limit = np.arange(vid_total_frames,
                                    vid_total_frames-self.n_frames*self.frame_skip,step=-self.frame_skip)-1
        
        frame_idx = random.randint(0, init_rand_limit[-1])
        
        frame_idx_list = np.arange(0,self.n_frames*self.frame_skip,step=self.frame_skip)
        frame_idx_list = frame_idx_list+frame_idx

        frames = []
        flag = 0
        for fr in range(self.n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_list[fr])
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Failed to read frame.")
  

            # Croping
            h, w, _ = frame.shape
            if flag == 0:
                x = random.randint(0, w - self.crop_size[0])
                y = random.randint(0, h - self.crop_size[1])
                flag = 1

            process_frame = frame[y:y + self.crop_size[0], x:x + self.crop_size[1]]
            # Convert frame to PyTorch tensor

            # Monochrome or RGB
            if self.color_mode == "gray":
                process_frame = self.grayscale_transforms(process_frame)              
                
            frames.append(process_frame.to(self.device))

        cap.release()

        tensor_vid = torch.squeeze(torch.stack(frames))
        # Random ratation
        if self.randomRotation:
            rot_ang = random.choice([0, 90, 180, 270])
            rand_flip = random.choice([0,1,2])

            if rand_flip == 1:
                tensor_vid = transforms.functional.hflip(tensor_vid)
            if rand_flip == 2:
                tensor_vid = transforms.functional.vflip(tensor_vid)

            tensor_vid = transforms.functional.rotate(tensor_vid,rot_ang)

        tensor_vid = transforms.Resize(size=self.resolution)(tensor_vid)

        return tensor_vid