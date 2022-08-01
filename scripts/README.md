# BASALT 2022 - Python Scripts

## [Build Item Classifier Data - FINISHED](<https://github.com/chaseabrown/BASALT2022/scripts/buildItemClassifierData.py>) 

A script that edits the minerl package before importing to edit which items will start in the inventory. Those items are then logged by name and quantity and an image of the inventory is stored. This is used to train both the item classifier and the item quantity classifier in *InvClassifier.py*.

**Packages Used: PIL, Logging, ColoredLogs, DateTime, OS, Random, Sys, Pandas**
  
## [Block Prediction Solution Test - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/blockPredSolutionTest.py>) 

A script testing the item classifier in *InvClassifer.py*. 

**Packages Used: OS, Numpy, PIL, MatPlotLib, SkLearn**
  
## [Quantity Prediction Solution Test - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/quantitySolutionTest.py>) 

A script testing the item quantity classifier in *InvClassifer.py*. 

**Packages Used: OS, Numpy, PIL, MatPlotLib, SkLearn**

## [Clean Inventory Data - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/cleanInventoryData.py>) 

A script that takes an image of the inventory and saves a bunch of small images of each item in the inventory.

**Packages Used: PIL, OS, Numpy, Pandas**

## [Simulate Cursor In Dataset - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/simulateCursorInDataset.py>) 

A script that takes in a list of inventory images with no cursor (I moved it off of the screen for this purpose) and pastes a picture of the cursor at a random point on the screen and stores that location with a saved picture. This is done multiple times to create the dataset for my cursor detector model in *InvClassifier.py*

**Packages Used: PIL, Numpy, OS, Random**

## [Test InvClassifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/testInvClassifier.py>) 

A scripts that tests both the item classifier and the quantity classifer by importing them from their class in  *InvClassifier.py*

**Packages Used: Sys, OS, Numpy, PIL, SkLearn**
  
## [Train Move Classifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/trainMoveClassifier.py>) 

 ```
  Arguments:
  --features (required) ["attack", "forward", "backward", "left", "right", "jump", "sneak", "sprint", "use", "drop", "inventory"] These are the different  models we need to train. There are a few more, but they don't work at the moment. Will update as those are ready

  --batchsize (default = 16) [1:64] Make sure to change batchsize to fit your machine (If your graphics card can't handle it, drop the batchsize, if it can, increase it until your training speed stops increasing)

  --epochs (default = 10) [1:200] It has auto stopping, so just make sure its high enough to meet it. If you are running a speed test and accuracy doesn't matter then drop it

  --balance (default = 0) [1 or 0] Balances the number of positive and negative cases (Right now it drops a bunch of data, but I need to fix that to  duplicate smaller dataset for final version)
  
  --shrink (default = 1.0) (0.0, 1.0] Ratio for shrinking an image. 1.0 keep it at original size and .5 shrinks it to half its size for example
  
  --subset (default = 1.0) (0.0, 1.0] Samples a subset of the dataset for faster tests using less data
  
  Example
  python testMoveClassifier.py --features "inventory" --batchsize 16 --epochs 10 --balanced 1 --shrink .5 --subset .1
  ```
  
A script used to train *MoveClassifier.py*. There are many arguments used for testing different versions of the model. 

**Packages Used: Sys, Os, Pandas, Random, Numpy, MatPlotLib, ArgParse, Keras, PIL, TQDM, Time, Json, CV2, Multiprocessing**
  
## [Test Move Classifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/testMoveClassifier.py>) 
  
  ```
  Arguments:
  --features (required) ["attack", "forward", "backward", "left", "right", "jump", "sneak", "sprint", "use", "drop", "inventory"] These are the different  models we need to train. There are a few more, but they don't work at the moment. Will update as those are ready

  --batchsize (default = 16) [1:64] Make sure to change batchsize to fit your machine (If your graphics card can't handle it, drop the batchsize, if it can, increase it until your training speed stops increasing)

  --epochs (default = 10) [1:200] It has auto stopping, so just make sure its high enough to meet it. If you are running a speed test and accuracy doesn't matter then drop it

  --balance (default = 0) [1 or 0] Balances the number of positive and negative cases (Right now it drops a bunch of data, but I need to fix that to  duplicate smaller dataset for final version)
  
  --shrink (default = 1.0) (0.0, 1.0] Ratio for shrinking an image. 1.0 keep it at original size and .5 shrinks it to half its size for example
  
  --subset (default = 1.0) (0.0, 1.0] Samples a subset of the dataset for faster tests using less data
  
  Example
  python testMoveClassifier.py --features "inventory" --batchsize 16 --epochs 10 --balanced 1 --shrink .5 --subset .1
  ```
A script used to test the *MoveClassifier.py* models created by *trainMoveClassifier.py*. Matching the arguments used on *trainMoveClassifer.py* will test that trained model. 
 
**Packages Used: Sys, OS, Pandas, Random, MatPlotLib, ArgParse, Keras, Tensorflow/PlaidML**
  
## [Parse Video Data - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/parseVideoData.py>) 

A script that breaks up video data into more easily accessable move->frame pairs.
  
**Packages Used: OS, CV2, Json, Numpy, PIL, Pandas, TQDM, Multiprocessing**
  
## [Sort Color Maps - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/sortColorMaps.py>) 

A script that reads all of the ColorMap data and labels which colors are in which ColorMap and saves those values in an CSV for easy reading for later analysis.

**Packages Used: PIL, Numpy, Pandas, OS, TQDM, Json, ShUtil, Multiprocessing**
  
## [Download Youtube Transcripts - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/downloadYoutubeTranscripts.py>) 

A script that loops through the MineDojo Youtube video and downloads the transcripts so I can search videos by what is said in them.

**Packages Used: Minedojo, Youtube Transcript API, TQDM, Pandas, Multiprocessing**
  
## [Train Block Segmentator - IN PROGRESS](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/trainBlockSegmentator.py>) 

A script that starts training for *BlockSegmentation.py*. 

**Packages Used: Random, Json, Requests, ShUtil, PIL, Numpy, Pandas, OS, TQDM**
