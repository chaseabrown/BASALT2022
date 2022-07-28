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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
sys.stderr = stderr
import PIL
from tqdm import tqdm
import time
import json
import cv2
import multiprocessing

sys.path.append('../models')
from MoveClassifier import MoveClassifier
sys.path.append('../helpers')
from Generators import Generator2Images, GeneratorStartImage


def getFrames(videoPath, startFrame, numFrames, COLOR):
    cap = cv2.VideoCapture(videoPath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(1,startFrame)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = []

    fc = startFrame
    ret = True

    counter = numFrames-1
    while (fc < startFrame + numFrames):
        
        ret, frame = cap.read()
        if COLOR:
          buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
          buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fc += 1
        counter -= 1

    cap.release()
    return buf

def gatherData(DATAPATH, FEATURE, BALANCED, SUBSET):
    moves = pd.read_csv(DATAPATH + "moves.csv")
    if BALANCED:
        positives = moves[moves[FEATURE] == 1]
        negatives = moves[moves[FEATURE] == 0]
        if len(positives) > len(negatives):
            moves = pd.concat([moves, negatives.sample(len(positives) - len(negatives), replace=True)], ignore_index=True)
        else:
            moves = pd.concat([moves, positives.sample(len(negatives) - len(positives), replace=True)], ignore_index=True)
    
    if SUBSET:
        moves = moves.sample(frac=SUBSET)

    videoPaths = moves['videoFile'].tolist()
    startFrames = moves['tick'].tolist()
    labels = moves[FEATURE].tolist()

    images = []
    for i in range(len(videoPaths)):
        images.append((videoPaths[i], startFrames[i]))

    print("# of Actions:", len(labels))
    return images, labels

# Takes in images from gatherData and returns input for model
def processImage(image, INPUTSHAPE):
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


def main(FEATURE, BATCHSIZE, EPOCHS, BALANCED, INPUTSHAPE, SUBSET, COLOR):

    # --------------------------------------------- Load Model and Data -------------------------------------------------
    # Set Output Path
    outputPath = "../model_tests/Move Classifier/" + FEATURE + "-BAL" + str(BALANCED) + "-BS" + str(BATCHSIZE) + "-IW" + str(INPUTSHAPE[0]) + "-IH" + str(INPUTSHAPE[1]) + "-SUB" + str(SUBSET) + "-RGB" + str(COLOR) + "/"
    # Paths to Data
    DATAPATH = "../assets/datasets/BASALT Contractor Dataset/"

    # Time Check and Update User
    trainDataTime = time.time()
    print("Loading Training Data...")

     # Load Data
    images, labels = gatherData(DATAPATH, FEATURE, BALANCED, SUBSET)

    # Split Data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = splitTrainData(images, labels)

    # If Feature uses 2 images model and is categorically binary, use this
    if FEATURE in ["attack", "forward", "backward", "left", "right", "jump", "sneak", "sprint", "use", "drop"]:

        # Load Data into Generators
        generator = Generator2Images(X_train, Y_train, batch_size=BATCHSIZE, inputShape=INPUTSHAPE, COLOR=COLOR)
        val_generator = Generator2Images(X_val, Y_val, batch_size=BATCHSIZE, inputShape=INPUTSHAPE, COLOR=COLOR)

        # Create Model
        print("Building " + FEATURE + " Model...")
        MC = MoveClassifier(INPUTSHAPE)
        model = MC.build_model_2Images(regress=False)
    
    # If Feature uses 1 images model and is categorically binary, use this
    else:

        # Load Data into Generators
        generator = GeneratorStartImage(X_train, Y_train, batch_size=BATCHSIZE, inputShape=INPUTSHAPE, COLOR=COLOR)
        val_generator = GeneratorStartImage(X_val, Y_val, batch_size=BATCHSIZE, inputShape=INPUTSHAPE, COLOR=COLOR)


        # Create Model
        print("Building " + FEATURE + " Model...")
        MC = MoveClassifier(INPUTSHAPE)
        model = MC.build_model_1Image(regress=False)


    # --------------------------------------------- Train Model -------------------------------------------------
    # Time Check and Update User
    trainDataTime = time.time() - trainDataTime
    trainTime = time.time()
    print("Training " + FEATURE + " Model...")
    print("Using", multiprocessing.cpu_count(), "cores.")

    # Setup Environment and Callbacks for Training
    if not os.path.exists(outputPath):
            os.makedirs(outputPath)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
    # Train Model
    history = model.fit_generator(generator=generator,
                validation_data=val_generator,
                use_multiprocessing=True,
                callbacks = [reduce_lr, early_stopping],
                workers=multiprocessing.cpu_count(),
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
        for path, startFrame in X_test:
            frames = getFrames(path, startFrame, 2, COLOR)
            startImages.append(processImage(PIL.Image.fromarray(frames[0]), INPUTSHAPE))
            endImages.append(processImage(PIL.Image.fromarray(frames[1]), INPUTSHAPE))

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
        for path, startFrame in X_test:
            frames = getFrames(path, startFrame, 1, COLOR)
            testImages.append(processImage(PIL.Image.fromarray(frames[0]), INPUTSHAPE))
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
                    output.append({"image": X_test[i][0], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "TN"})
                    TN += 1
                else:
                    output.append({"image": X_test[i][0], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "TP"})
                    TP += 1
            else:
                if Y_test[i] == 0:
                    output.append({"image": X_test[i][0], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "FP"})
                    FP += 1
                else:
                    output.append({"image": X_test[i][0], "prediction": predictions[i][0], "true": int(Y_test[i]), "results": "FN"})
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

    #Setup Arguments
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--balanced", type=int, default=0)
    parser.add_argument("--shrink", type=float, default=1.0)
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--color", type=int, default=1)

    args = parser.parse_args()

    BALANCED = False
    if args.balanced == 1:
        BALANCED = True
    COLOR = False
    dims = 1
    if args.color == 1:
        COLOR = True
        dims = 3
    
    INPUTSHAPE = (int(640*args.shrink), int(360*args.shrink), dims)

    # Call Main
    main(args.feature, args.batchsize, args.epochs, BALANCED, INPUTSHAPE, args.subset, COLOR)





