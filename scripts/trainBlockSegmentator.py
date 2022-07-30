import random
import json
import requests
import shutil
import PIL
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

sys.path.append('../models')
from MovBlockSegmentationeClassifier import BlockSegmentator
sys.path.append('../helpers')
from Generators import SegmentationGenerator

imageSize = (640, 360, 3)
numClasses = 

input_img_paths = []
target_img_paths = []

path = "/Volumes/Extreme SSD/Extra Datasets/video-depth-colormap/"
for run in os.listdir(path):
    newPath = path + run + "/"
    for index in range(0, len(os.listdir(newPath + "video_frames/")[20:40])):
        if os.path.exists(newPath + "video_frames/frame" + str(index) + ".png") and os.path.exists(newPath + "depth_frames/frame" + str(index) + ".png") and os.path.exists(newPath + "colormap_frames/frame" + str(index) + ".png"):
            input_img_paths.append(newPath + "video_frames/frame" + str(index) + ".png")
            target_img_paths.append(newPath + "colormap_frames/frame" + str(index) + ".png")

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = SegmentationGenerator(
    16, imageSize, train_input_img_paths, train_target_img_paths
)
val_gen = SegmentationGenerator(16, imageSize, val_input_img_paths, val_target_img_paths)

segmentator = BlockSegmentator(imageSize, numClasses)