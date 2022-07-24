import sys
import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.callbacks import EarlyStopping
sys.stderr = stderr
import PIL
from tqdm import tqdm
import time
import json

sys.path.append('../models')
from MoveClassifier import MoveClassifier
sys.path.append('../helpers')
from Generators import Generator2Images, GeneratorEndImage

# Takes in data paths and args and returns data for generators.
def gatherData(dataPathList, framesViewed, FEATURE, BALANCED, SUBSET):
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
                newMoves = newMoves.sample(frac=SUBSET, random_state=42)
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

# Takes in images from gatherData and returns input for model
def processImage(image):
    image = PIL.Image.open(image)
    image.thumbnail((INPUTSHAPE[0],INPUTSHAPE[1]), PIL.Image.ANTIALIAS)
    image = np.array(image)
    image = image.astype('float32')
    return image/255.

# Splits data into training, validation, and testing sets after sufffling them
def splitTrainData(images, labels):
    temp = list(zip(images, labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    images, labels = list(res1), list(res2)
    
    X_train = images[:int(len(images) * 0.7)]
    Y_train = labels[:int(len(labels) * 0.7)]
    X_val = images[int(len(images) * 0.7):int(len(images) * 0.9)]
    Y_val = labels[int(len(labels) * 0.7):int(len(labels) * 0.9)]
    X_test = images[int(len(images) * 0.9):]
    Y_test = labels[int(len(labels) * 0.9):]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def main(FEATURE, BATCHSIZE, EPOCHS, BALANCED, INPUTSHAPE, SUBSET):

    # --------------------------------------------- Load Model and Data -------------------------------------------------
    # Set Output Path
    outputPath = "../model_tests/Move Classifier/" + FEATURE + "-BAL" + str(BALANCED) + "-BS" + str(BATCHSIZE) + "-IW" + str(INPUTSHAPE[0]) + "-IH" + str(INPUTSHAPE[0]) + "-SUB" + str(SUBSET) + "/"
    
    # Time Check and Update User
    trainDataTime = time.time()
    print("Loading Training Data...")
    X_test = []
    Y_test = []

    # If Feature uses 2 images model and is categorically binary, use this
    if FEATURE in ["attack", "forward", "backward", "left", "right", "jump", "sneak", "sprint", "use", "drop"]:

        # Load Data
        images, labels = gatherData(DATAPATHS, 2, FEATURE, BALANCED, SUBSET)

        # Split Data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = splitTrainData(images, labels)

        # Load Data into Generators
        generator = Generator2Images(X_train, Y_train, batch_size=BATCHSIZE, inputShape=INPUTSHAPE)
        val_generator = Generator2Images(X_val, Y_val, batch_size=BATCHSIZE, inputShape=INPUTSHAPE)

        # Create Model
        print("Building " + FEATURE + " Model...")
        MC = MoveClassifier(INPUTSHAPE)
        model = MC.build_model_2Images(regress=False)
    
    # If Feature uses 1 images model and is categorically binary, use this
    else:
        # Load Data
        images, labels = gatherData(DATAPATHS, 2, FEATURE, BALANCED, SUBSET)

        # Split Data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = splitTrainData(images, labels)

        # Load Data into Generators
        generator = GeneratorEndImage(X_train, Y_train, batch_size=BATCHSIZE, inputShape=INPUTSHAPE)
        val_generator = GeneratorEndImage(X_val, Y_val, batch_size=BATCHSIZE, inputShape=INPUTSHAPE)

        # Create Model
        print("Building " + FEATURE + " Model...")
        MC = MoveClassifier(INPUTSHAPE)
        model = MC.build_model_1Image(regress=False)


    # --------------------------------------------- Train Model -------------------------------------------------
    # Time Check and Update User
    trainDataTime = time.time() - trainDataTime
    trainTime = time.time()
    print("Training " + FEATURE + " Model...")


    # Setup Environment and Callbacks for Training
    if not os.path.exists(outputPath):
            os.makedirs(outputPath)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)

    # Train Model
    history = model.fit_generator(generator=generator,
                validation_data=val_generator,
                use_multiprocessing=True,
                callbacks = [early_stopping],
                workers=6,
                epochs=EPOCHS)
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(outputPath + "/loss.png")

    

    # --------------------------------------------- Test the Model -------------------------------------------------

    # Time Check and Update User
    trainTime = time.time() - trainTime
    timePerEpoch = trainTime/EPOCHS
    testDataTime = time.time()
    predictTime = time.time()
    print("Loading Test Data...")

    # Variables for Output
    predictions = []
    output = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # If Feature uses 2 images model and is categorically binary, use this
    if FEATURE in ["attack", "forward", "backward", "left", "right", "jump", "sneak", "sprint", "use", "drop"]:

        # Load Test Data
        startImages = []
        endImages = []
        for x in tqdm(X_test):
            startImages.append(processImage(x[0]))
            endImages.append(processImage(x[1]))

        testImages = [np.array(startImages, np.float32), np.array(endImages, np.float32)]

        # Time Check and Update User
        testDataTime = time.time() - testDataTime
        predictTime = time.time()
        print("Model Predicting...")
        predictions = model.predict(testImages)            
        

        # Check Predictions
        for i in tqdm(range(0, len(predictions))):
            if int(Y_test[i]) == round(predictions[i][0]):
                if Y_test[i] == 0:
                    output.append({"image1": X_test[i][0], "image2": X_test[i][1], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "TN"})
                    TN += 1
                else:
                    output.append({"image1": X_test[i][0], "image2": X_test[i][1], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "TP"})
                    TP += 1
            else:
                if Y_test[i] == 0:
                    output.append({"image1": X_test[i][0], "image2": X_test[i][1], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "FP"})
                    FP += 1
                else:
                    output.append({"image1": X_test[i][0], "image2": X_test[i][1], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "FN"})
                    FN += 1

    # If Feature uses 1 images model and is categorically binary, use this
    else:

        # Load Test Data
        testImages = []
        for x in tqdm(X_test):
            startImages.append(processImage(x[1]))
        testImages = np.array(testImages, np.float32)

        # Time Check and Update User
        testDataTime = time.time() - testDataTime
        predictTime = time.time()
        print("Model Predicting...")
        predictions = model.predict(testImages)            
        
        # Check Predictions
        for i in tqdm(range(0, len(predictions))):
            if int(Y_test[i]) == round(predictions[i][0]):
                if Y_test[i] == 0:
                    output.append({"image": X_test[i], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "TN"})
                    TN += 1
                else:
                    output.append({"image": X_test[i], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "TP"})
                    TP += 1
            else:
                if Y_test[i] == 0:
                    output.append({"image": X_test[i], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "FP"})
                    FP += 1
                else:
                    output.append({"image": X_test[i], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "FN"})
                    FN += 1

    # Time Check and Update User
    print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    predictTime = time.time() - predictTime



    # -------------------------------------------------- Output Results -------------------------------------------------

    # Write Times to File
    with open(outputPath + "/time.json", "w") as outfile:
        json.dump({"trainDataTime": trainDataTime, "trainTime": trainTime, "timePerEpoch": timePerEpoch, "testDataTime": testDataTime, "predictTime": predictTime}, outfile)
    
    # Write Predictions to File
    df = pd.DataFrame.from_dict(output)
    df.to_csv(outputPath + "/predictions.csv", index=False)
    
    # Save Model
    model.save(outputPath + "/model.h5")



# -------------------------------------------------- Initialize main -------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser("Train Move Classifier on Feature")

    # Paths to Data
    DATAPATHS = ["../assets/datasets/Move Classifier Data/MineRLBasaltFindCave-v0/", 
                        "../assets/datasets/Move Classifier Data/MineRLBasaltBuildVillageHouse-v0/", 
                        "../assets/datasets/Move Classifier Data/MineRLBasaltCreateVillageAnimalPen-v0/", 
                        "../assets/datasets/Move Classifier Data/MineRLBasaltMakeWaterfall-v0/"]

    #Setup Arguments
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--balanced", type=int, default=0)
    parser.add_argument("--shrink", type=float, default=1.0)
    parser.add_argument("--subset", type=float, default=1.0)

    args = parser.parse_args()

    BALANCED = False
    if args.balanced == 1:
        BALANCED = True
    
    INPUTSHAPE = (int(640*args.shrink), int(360*args.shrink), 3)

    # Call Main
    main(args.feature, args.batchsize, args.epochs, BALANCED, INPUTSHAPE, args.subset)


