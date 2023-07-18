from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time
import ffmpy

from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import SoccerNetDownloader

from Features.VideoFeatureExtractor import PCAReducer, VideoFeatureExtractor

from torch.utils.data import Dataset

import os
import time


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]

class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", 
                framerate=2, window_size=15):
        self.path = path
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17

        ff = ffmpy.FFmpeg(
             inputs={self.path: ""},
             outputs={"inference/outputs/videoLQ.mkv": '-y -r 25 -vf scale=-1:224 -max_muxing_queue_size 9999'})
        print(ff.cmd)
        ff.run()

        print("Initializing feature extractor")
        myFeatureExtractor = VideoFeatureExtractor(
            feature="ResNET",
            back_end="TF2",
            transform="crop",
            grabber="opencv",
            FPS=self.framerate)

        print("Extracting frames")
        myFeatureExtractor.extractFeatures(path_video_input="inference/outputs/videoLQ.mkv",
                                           path_features_output="inference/outputs/features.npy",
                                           overwrite=True)

        print("Initializing PCA reducer")
        myPCAReducer = PCAReducer(pca_file="inference/Features/pca_512_TF2.pkl",
                                  scaler_file="inference/Features/average_512_TF2.pkl")

        print("Reducing with PCA")
        myPCAReducer.reduceFeatures(input_features="inference/outputs/features.npy",
                                    output_features="inference/outputs/features_PCA.npy",
                                    overwrite=True)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
        """
        # Load features
        feat_half1 = np.load(os.path.join("inference/outputs/features_PCA.npy"))

        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
        print("Shape half 1: ", feat_half1.shape)

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        
        return feat_half1

    def __len__(self):
        return 1