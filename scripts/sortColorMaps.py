import PIL.Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import shutil
from multiprocessing import Process


#Process to be run in parallel
def workerTask(samples, i):
    savePath = '/Volumes/Extreme SSD/Extra Datasets/video-depth-colormap/'
    #frames, width, height = getFrames(sample[0])
    video_width = 640
    video_height = 360
    colorData = []
    for index, row in tqdm(samples.iterrows(), total=len(samples)):
        image = PIL.Image.open(row["ColorMap"])
        imgarr = np.asarray(image)
        colorCount = {"ColorMap": row["ColorMap"]}
        for r in range(0, video_height):
            for c in range(0, video_width):
                color = str(imgarr[r][c][0]) + "," + str(imgarr[r][c][1]) + "," + str(imgarr[r][c][2])
                if color not in colorCount:
                    colorCount.update({color: 1})
                else:
                    colorCount[color] += 1
        colorData.append(colorCount)
    colorData = pd.DataFrame.from_dict(colorData)
    colorData.to_csv(savePath + "colors" + str(i) + ".csv", index=False)
            
#Runs body in i parallel fragements
def runBlocks(cores, samples):
    listOfProcesses = []
    for i in range(0, cores):
        start = int(i*(len(samples)/cores))
        end = int((i+1)*(len(samples)/cores))
        process = Process(target=workerTask, args=(samples.iloc[start:end], i))
        process.start()
        listOfProcesses.append(process)
    
    for process in listOfProcesses:
        process.join()


def main():
    path = "/Volumes/Extreme SSD/Extra Datasets/video-depth-colormap/"
    frames = {"RGB": [], "Depth": [], "ColorMap": []}
    for run in os.listdir(path):
        newPath = path + run + "/"
        for index in range(0, len(os.listdir(newPath + "video_frames/"))):
            if os.path.exists(newPath + "video_frames/frame" + str(index) + ".png") and os.path.exists(newPath + "depth_frames/frame" + str(index) + ".png") and os.path.exists(newPath + "colormap_frames/frame" + str(index) + ".png"):
                frames['RGB'].append(newPath + "video_frames/frame" + str(index) + ".png")
                frames['Depth'].append(newPath + "depth_frames/frame" + str(index) + ".png")
                frames['ColorMap'].append(newPath + "colormap_frames/frame" + str(index) + ".png")
    df = pd.DataFrame.from_dict(frames)
    
    cores = 8
    runBlocks(cores, df)
    
if __name__ == '__main__':
    main()
    
    