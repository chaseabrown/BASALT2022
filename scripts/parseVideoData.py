
import os
import cv2
import json
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process

def getFrames(videoPath):
    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoFPS = int(cap.get(cv2.CAP_PROP_FPS))

    buf = np.empty((
        frameCount,
        frameHeight,
        frameWidth,
        3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount):
        ret, frame = cap.read()
        buf[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fc += 1

    cap.release()
    videoArray = buf
    return videoArray, frameWidth, frameHeight

def getMoves(movesPath):
    with open(movesPath, "r") as json_file:
        imagePath = '/'.join(movesPath.split('/')[:6]).replace("Agent Moves", "Move Classifier Data").replace(".txt", "") + "/"
        listOfMoves = []
        counter = 0
        for line in json_file.readlines():
            try:
                moves = json.loads(line.replace(":", '":').replace("{", '{"').replace("], ", '], "'))
                clean = []
                for key in moves.keys():
                    if key == "camera":
                        clean = str(moves[key]).replace("[", "").replace("]", "").split(", ")
                    else:
                        moves[key] = int(str(moves[key]).replace("[", "").replace("]", ""))
                moves.update({"camera1": float(clean[0]), "camera2": float(clean[1])})
                moves.update({"startImage": counter})
                moves.pop('camera', None)
                counter += 1
                listOfMoves.append(moves)
            except Exception as e:
                print(e)
    return listOfMoves

def gatherData(dataPathList):
    samples = []
    for dataPath in dataPathList:
        for file in os.listdir(dataPath):
            if file.endswith(".txt"):
                try:
                    videoFile = os.listdir(dataPath + file.split('.')[0] + ".mp4/")[0]
                    sample = [dataPath + file.split('.')[0] + ".mp4/" + videoFile, dataPath + file]
                    samples.append(sample)
                except Exception as e:
                    print(file, e)

    return samples



#Process to be run in parallel
def workerTask(samples, i):
    for sample in tqdm(samples):
        savePath = '/'.join(sample[0].split('/')[:6]).replace("Agent Moves", "Move Classifier Data").replace(".mp4", "") + "/"
        frames, width, height = getFrames(sample[0])
        
        moves = getMoves(sample[1])
        moves
        hotbar = 1
        for move in moves:
            move.update({"hotbar": hotbar})
            for key in move.keys():
                if "hotbar." in key and move[key]==1:
                    hotbar = int(key[-1])
        
        moves = pd.DataFrame.from_dict(moves)
        
        
        
        if not os.path.exists(savePath):
            os.makedirs(savePath)
            
        for index in range(0, len(moves)):
            Image.fromarray(frames[index]).save(savePath + str(index) + ".jpg")
        moves = moves.drop(['hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9'], axis=1)
        moves.to_csv(savePath + "moves.csv", index=False)
            
#Runs body in i parallel fragements
def runBlocks(cores, samples):
    listOfProcesses = []
    for i in range(0, cores):
        start = int(i*(len(samples)/cores))
        end = int((i+1)*(len(samples)/cores))
        process = Process(target=workerTask, args=(samples[start:end], i))
        process.start()
        listOfProcesses.append(process)
    
    for process in listOfProcesses:
        process.join()


def main():
    dataPathList = ["../assets/datasets/Agent Moves/MineRLBasaltFindCave-v0/", 
                            "../assets/datasets/Agent Moves/MineRLBasaltBuildVillageHouse-v0/", 
                            "../assets/datasets/Agent Moves/MineRLBasaltCreateVillageAnimalPen-v0/", 
                            "../assets/datasets/Agent Moves/MineRLBasaltMakeWaterfall-v0/"]
    
    samples = gatherData(dataPathList)
    
    cores = 8
    runBlocks(cores, samples)
    
if __name__ == '__main__':
    main()
    
    