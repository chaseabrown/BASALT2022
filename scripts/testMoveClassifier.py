import sys
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.callbacks import ModelCheckpoint
sys.stderr = stderr

sys.path.append('../models')
from MoveClassifier import MoveClassifier
sys.path.append('../helpers')
from Generators import Generator2Images, Generator1Image

def gatherData(dataPathList, framesViewed, FEATURE, BALANCED):
    folders = []
    images = []
    labels = []
    
    for dataPath in dataPathList:
        for folder in os.listdir(dataPath):
            folders.append(dataPath + folder)
            if not ".DS_Store" in folder:
                newMoves = pd.read_csv(dataPath + folder + "/moves.csv")
                if BALANCED:
                    positives = newMoves[newMoves[FEATURE] == 1]
                    negatives = newMoves[newMoves[FEATURE] == 0]
                    if len(positives) > len(negatives):
                        newMoves = pd.concat([negatives, positives[:len(negatives)]], ignore_index=True)
                    else:
                        newMoves = pd.concat([positives, negatives[:len(positives)]], ignore_index=True)

                balanced = newMoves.sample(frac=1, random_state=42)
                for index, move in newMoves.iterrows():

                    framesToInclude = []
                    end = False
                    for i in range(0, framesViewed):
                        if not os.path.exists(dataPath + folder + "/" + str(int(move["startImage"] + i)) + ".jpg"):
                            end = True
                        framesToInclude.append(dataPath + folder + "/" + str(int(move["startImage"] + i)) + ".jpg")
                    if not end:
                        images.append(framesToInclude)
                        labels.append(move[FEATURE])

    print("# of Actions:", len(labels))
    return images, labels


def main(FEATURE, BATCHSIZE, EPOCHS, BALANCED, INPUTSHAPE):
    print("Loading Test Data...")
    if FEATURE in ["attack", "forward", "backward", "left", "right", "jump", "sneak", "sprint", "use", "drop"]:
        images, labels = gatherData(DATAPATHS, 2, FEATURE, False)
        generator = Generator2Images(images, labels, batch_size=BATCHSIZE, inputShape=INPUTSHAPE)
        preds = []
        
        print("Loading Model...")
        MC = MoveClassifier(INPUTSHAPE)
        model = MC.build_model_2Images(regress=False)

        print("Loading Weights...")
        checkpoint_path = "../weights/Move Classifier/" + FEATURE + "-BAL" + str(BALANCED) + "-BS" + str(BATCHSIZE) + "-IW" + str(INPUTSHAPE[0])+ "-IH" + str(INPUTSHAPE[0]) + "/cp-{epoch:04d}.ckpt"
        checkpoint_path = checkpoint_path.format(epoch=EPOCHS)
        model.load_weights(checkpoint_path)

        print("Model Predicting...")
        for i in range(0, len(generator)):
            for item in generator.__getitem__(i):
                pred = model.predict(item[0])
                preds.append(pred)
    else:
        images, labels = gatherData(DATAPATHS, 1, FEATURE, True)
        generator = Generator1Image(images, labels, batch_size=BATCHSIZE, inputShape=INPUTSHAPE)
        preds = []

        print("Loading Model...")
        MC = MoveClassifier(INPUTSHAPE)
        model = MC.build_model_1Image(regress=False)

        checkpoint_path = "../weights/Move Classifier/" + FEATURE + "-BAL" + str(BALANCED) + "-BS" + str(BATCHSIZE) + "-IW" + str(INPUTSHAPE[0])+ "-IH" + str(INPUTSHAPE[0]) + "/cp-{epoch:04d}.ckpt"
        checkpoint_path = checkpoint_path.format(epoch=EPOCHS)
        model.load_weights(checkpoint_path)

        print("Model Predicting...")
        for i in range(0, len(generator)):
            for item in generator.__getitem__(i):
                pred = model.predict(item[0])
                preds.append(pred)

if __name__ == "__main__":
    parser = ArgumentParser("Train Move Classifier on Feature")

    DATAPATHS = ["../assets/datasets/Move Classifier Data/MineRLBasaltFindCave-v0/", 
                        "../assets/datasets/Move Classifier Data/MineRLBasaltBuildVillageHouse-v0/", 
                        "../assets/datasets/Move Classifier Data/MineRLBasaltCreateVillageAnimalPen-v0/", 
                        "../assets/datasets/Move Classifier Data/MineRLBasaltMakeWaterfall-v0/"]

    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--balanced", type=int, required=True)
    parser.add_argument("--shrink", type=int, required=True)

    args = parser.parse_args()

    BALANCED = False
    if args.balanced == 1:
        BALANCED = True
    
    INPUTSHAPE = (640, 360, 3)
    if args.shrink == 1:
        INPUTSHAPE = (320, 180, 3)

    main(args.feature, args.batchsize, args.epochs, BALANCED, INPUTSHAPE)
