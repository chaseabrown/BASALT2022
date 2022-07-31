# BASALT2022
## These are the competition files for BASALT 2022

**This project is still in development.**

### [Models](https://github.com/chaseabrown/BASALT2022/blob/master/models/)
* [Inventory Classifier - FINISHED](https://github.com/chaseabrown/BASALT2022/blob/master/models/InvClassifier.py) is a class with 3 functions: read image of inventory items and predict the item type and quantity using a `Gaussian Naive Bayes` model and read an image of the full screen and get the cursor location using a `Convolutional Neural Network (CNN)`.

  **Packages Used: OS, NumPy, PIL, SkLearn, Detecto, MatPlotLib, CV2, GDown**

* [DQN - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/blob/master/models/DQN.py) a standard `Deep Q-Learning Network` which I intend to use for specific training situations that OpenAI's VPT Model isn't designed to handle.

  **Packages Used: OS, NumPy, Random, Collections, Keras, Tensorflow/PlaidML**

* [Move Classifier - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/blob/master/models/MoveClassifier.py) is a class with the sole purpose of reading 2 frame images from a video of gameplay and return the predicted move that was made between frames. I got pretty far with this, before I realized the work was already done by OpenAI's VPT Inverse Dynamics Model and decided to use that instead. My model uses 2 `Convolutional Neural Networks (CNNs)` that are combined at the ends finished with an activation function specific to each move type.

  **Packages Used: OS, Sys, ArgParse, Locale, JSON, Random, Keras, Tensorflow/PlaidML**

* [Block Segmentation - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/blob/master/models/BlockSegmentation.py) is a class with the sole purpose of reading in an RGB Frame and returning a ColorMap that represents the block at each pixel. The model uses `Keras` for `Semantic Segmentation` with a `U-Net Architecture`.

  **Packages Used: OS, Sys, Keras, Tensorflow/PlaidML**

* [Depth Estimator - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/blob/master/models/DepthEstimator.py) is a class with the sole purpose of reading in an RGB Frame and returning a depth map at each pixel. The model uses `Keras` for a `Deep Learning Autoencoder`.

  **Packages Used: OS, Sys, NumPy, Pandas, MatPlotLib, CV2, Keras, Tensorflow/PlaidML**


### [Generators](https://github.com/chaseabrown/BASALT2022/blob/master/helpers/Generators.py)

* [GeneratorStartImage - FINISHED](https://github.com/chaseabrown/BASALT2022/blob/master/helpers/Generators.py) is a generator class that serves some of the models in *MoveClassifier.py*. For example, the model that detects whether or not the inventory is open only needs the frame before a move is given. 

* [GeneratorEndImage - FINISHED](https://github.com/chaseabrown/BASALT2022/blob/master/helpers/Generators.py) is a generator class that serves some of the models in *MoveClassifier.py*. For example, the model that detects whether or not the character is told to attack needs the frame after a move is given. 

* [Generator2Images - FINISHED](https://github.com/chaseabrown/BASALT2022/blob/master/helpers/Generators.py) is a generator class that serves some of the models in *MoveClassifier.py*. For example, the model that detects whether or not the character is moving in a specific direction needs 2 frames so it can compare the two.

* [SegmentationGenerator - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/blob/master/helpers/Generators.py) is a generator class that serves the model in *BlockSegmentation.py*. It serves the model both the RGB image and the ColorMap images for training.

* [DepthEstGenerator - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/blob/master/helpers/Generators.py) is a generator class that serves the model in *DepthEstimator.py*. It serves the model both the RGB image and the Depth Map images for training.

  **Packages Used: NumPy, PIL, Random, CV2, Math, Keras, Tensorflow/PlaidML**

### [Notebooks](https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/):

* [Item Classifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Item Classifier.ipynb>) is a notebook demonstration of the item classifier in *InvClassifier.py* with an explaination of the process.

  **Packages Used: OS, NumPy, PIL, SkLearn, MatPlotLib, CV2**

* [Item Quantity Classifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Item Quantity Classifier.ipynb>) is a notebook demonstration of the item quantity classifier in *InvClassifier.py* with an explaination of the process.

  **Packages Used: OS, NumPy, PIL, SkLearn, MatPlotLib, CV2**

* [Data Generator Class Setup - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Data Generator Class Setup.ipynb>) is a notebook demonstration of how I step up and tested *StartImageGenerator* and *EndImageGenerator*. 

  **Packages Used: NumPy, PIL, Json, TQDM, Pandas, Random, CV2, Math, Keras**

* [Move Classifier Data Exploration - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Move Classifier Data Exploration.ipynb>) is a notebook visualization/exploration of my automatically generated gameplay dataset. This one is fun because it has a lot of pictures and data preprocessing, but this dataset did not end up chosen in the final model.

  **Packages Used: OS, Sys, Json, NumPy, IPython.Display, TQDM, Random**

* [Visualize Model Results (Move Classifier) - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Visualize Model Results (Move Classifier).ipynb>) is a notebook demonstrating the model outputs from *MoveClassifier.py*. 

  **Packages Used: OS, NumPy, Pandas, MatPlotLib, PIL, Json, Random, IPython.display**

* [Dataset Exploration (BASALTAgent Moves) - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Dataset Exploration (BASALTAgent Moves).ipynb>) is a notebook visualization/exploration of BASALT 2022 competition gameplay dataset. This one is fun because it has a lot of pictures and data preprocessing.

  **Packages Used: OS, NumPy, Pandas, Json, CV2, PIL, IPython.display**
  
* [ColorMap Smoothing - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/ColorMap Smoothing.ipynb>) is a notebook exploration at an attempt to smooth the ColorMap data collected. Concluded to not be a great choice for this project, but the smoothing works and is clearly demonstrated and visualized.

  **Packages Used: OS, NumPy, Pandas, TQDM, PIL, CV2**
  
* [KNN Grouping to Match Similar Colors in Image - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/KNN Grouping to Match Similar Colors in Image.ipynb>) is a notebook exploration at an attempt to smooth the ColorMap data collected using KNN. Concluded to not be a great choice for this project again, and smoothing was fine, but not great. *ColorMap Smoothing.ipynb* was better.

  **Packages Used: OS, NumPy, PIL, CV2, TDQM**
  
* [Explore Depth and ColorMaps - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Explore Depth and ColorMaps.ipynb>) is a notebook visualization of a Malmo dataset I generated for training data to feed into *BlockSegmentator.py* and *DepthEstimator.py*. Lots of visuals, tables, and pretty pictures.

  **Packages Used: OS, NumPy, Pandas, PIL, TQDM, IPython.display, Json, Requests, ShUtil**
  
* [TF-IDF with Spark MineWiki Search Engine - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/TF-IDF with Spark MineWiki Search Engine.ipynb>) is a notebook demonstration of how I use TF-IDF to search through the many pages of MineWiki Data distributed by MineDojo. Works really well and uses PySpark for all the data preprocessing, transformation and TF-IDF work which makes it really fast. This is the start of me introducing Natural Language Processing to get more information for my Agent.

  **Packages Used: OS, Pandas, PySpark SQL, PySparkML**

* [Create Search Engine for MineDojo Youtube Dataset - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Create Search Engine for MineDojo Youtube Dataset.ipynb>) is a notebook demonstration of how I use TF-IDF to search through transcripts of 70k gameplay Youtube Videos distributed by MineDojo. After the success on the MineWiki search, I brought that to this problem. Doesn't work great because the words said are not as consistent in a transcript as a Wiki page. It still uses PySpark for all the data preprocessing, transformation and TF-IDF work which makes it really fast. After this I need to look into some other options for gathering information from transcripts.

  **Packages Used: Minedojo, IPython.display, Random, TDQM, OS, RE, Pandas, PySpark SQL, PySparkML**

* [Explore MineDojo Gym Observations and Actions - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Explore MineDojo Gym Observations and Actions.ipynb>) is a notebook visualization of a the MineDojo Gym Environment's Observation and Action set. 

  **Packages Used: Minedojo, NumPy, Random, PIL**
  
* [Explore MineDojo Tasks - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/notebooks/Explore Depth and ColorMaps.ipynb>) is a notebook visualization of a the MineDojo Gym Tasks and Instructions.

  **Packages Used: Minedojo**

### [Scripts](https://github.com/chaseabrown/BASALT2022/blob/master/scripts/):

* [Build Item Classifier Data - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/buildItemClassifierData.py>) is a script that edits the minerl package before importing to edit which items will start in the inventory. Those items are then logged by name and quantity and an image of the inventory is stored. This is used to train both the item classifier and the item quantity classifier in *InvClassifier.py*.

  **Packages Used: PIL, Logging, ColoredLogs, DateTime, OS, Random, Sys, Pandas**
  
* [Block Prediction Solution Test - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/blockPredSolutionTest.py>) is a script testing the item classifier in *InvClassifer.py*. 

  **Packages Used: OS, Numpy, PIL, MatPlotLib, SkLearn**
  
* [Quantity Prediction Solution Test - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/quantitySolutionTest.py>) is a script testing the item quantity classifier in *InvClassifer.py*. 

  **Packages Used: OS, Numpy, PIL, MatPlotLib, SkLearn**

* [Clean Inventory Data - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/cleanInventoryData.py>) is a script that takes an image of the inventory and saves a bunch of small images of each item in the inventory.

  **Packages Used: PIL, OS, Numpy, Pandas**

* [Simulate Cursor In Dataset - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/simulateCursorInDataset.py>) is a script that takes in a list of inventory images with no cursor (I moved it off of the screen for this purpose) and pastes a picture of the cursor at a random point on the screen and stores that location with a saved picture. This is done multiple times to create the dataset for my cursor detector model in *InvClassifier.py*

  **Packages Used: PIL, Numpy, OS, Random**

* [Test InvClassifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/testInvClassifier.py>) is a scripts that tests both the item classifier and the quantity classifer by importing them from their class in  *InvClassifier.py*

  **Packages Used: Sys, OS, Numpy, PIL, SkLearn**
  
* [Train Move Classifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/trainMoveClassifier.py>) is the script used to train *MoveClassifier.py*. There are many arguments used for testing different versions of the model. 

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

  **Packages Used: Sys, Os, Pandas, Random, Numpy, MatPlotLib, ArgParse, Keras, PIL, TQDM, Time, Json, CV2, Multiprocessing**
  
* [Test Move Classifier - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/testMoveClassifier.py>) is the script used to test the *MoveClassifier.py* models created by *trainMoveClassifier.py*. Matching the arguments used on *trainMoveClassifer.py* will test that trained model.
  
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
  
  **Packages Used: Sys, OS, Pandas, Random, MatPlotLib, ArgParse, Keras, Tensorflow/PlaidML**
  
* [Parse Video Data - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/parseVideoData.py>) is a script that breaks up video data into more easily accessable move->frame pairs.
  
  **Packages Used: OS, CV2, Json, Numpy, PIL, Pandas, TQDM, Multiprocessing**
  
* [Sort Color Maps - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/sortColorMaps.py>) is a script that reads all of the ColorMap data and labels which colors are in which ColorMap and saves those values in an CSV for easy reading for later analysis.

  **Packages Used: PIL, Numpy, Pandas, OS, TQDM, Json, ShUtil, Multiprocessing**
  
* [Download Youtube Transcripts - FINISHED](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/downloadYoutubeTranscripts.py>) is a script that loops through the MineDojo Youtube video and downloads the transcripts so I can search videos by what is said in them.

  **Packages Used: Minedojo, Youtube Transcript API, TQDM, Pandas, Multiprocessing**
  
* [Train Block Segmentator - IN PROGRESS](<https://github.com/chaseabrown/BASALT2022/blob/master/scripts/trainBlockSegmentator.py>) is the script that starts training for *BlockSegmentation.py*. 

  **Packages Used: Random, Json, Requests, ShUtil, PIL, Numpy, Pandas, OS, TQDM**

 
