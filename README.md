# BASALT2022
## These are the competition files for BASALT 2022

**This Project is still in progress, so it is not very well documentmented**

## Quick Look - 

### Models:
* [Inventory Classifier - FINISHED](https://github.com/chaseabrown/BASALT2022/models/InvClassifier.py) is a class with 3 functions: read image of inventory items and predict the item type and quantity using a `Gaussian Naive Bayes` model and read an image of the full screen and get the cursor location using a `Convolutional Neural Network (CNN)`.

  **Packages Used: OS, NumPy, PIL, SkLearn, Detecto, MatPlotLib, CV2, GDown**

* [DQN - FINISHED (But will need work later)](https://github.com/chaseabrown/BASALT2022/models/DQN.py) is a class with 3 functions: read image of inventory items and predict the item type and quantity using a `Gaussian Naive Bayes` model and read an image of the full screen and get the cursor location using a `Convolutional Neural Network (CNN)`.

  **Packages Used: OS, NumPy, Random, Collections, Keras, Tensorflow/PlaidML**

* [Move Classifier - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/models/MoveClassifier.py) is a class with the sole purpose of reading 2 frame images from a video of gameplay and return the predicted move that was made between frames. I got pretty far with this, before I realized the work was already done by OpenAI's VPT Inverse Dynamics Model and decided to use that instead. My model uses 2 `Convolutional Neural Networks (CNNs)` that are combined at the ends finished with an activation function specific to each move type.

  **Packages Used: OS, Sys, ArgParse, Locale, JSON, Random, Keras, Tensorflow/PlaidML**

* [Block Segmentation - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/models/BlockSegmentation.py) is a class with the sole purpose of reading in an RGB Frame and returning a ColorMap that represents the block at each pixel. The model uses `Keras` for `Semantic Segmentation` with a `U-Net Architecture`.

  **Packages Used: OS, Sys, Keras, Tensorflow/PlaidML**

* [Depth Estimator - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/models/DepthEstimator.py) is a class with the sole purpose of reading in an RGB Frame and returning a depth map at each pixel. The model uses `Keras` for a `Deep Learning Autoencoder`.

  **Packages Used: OS, Sys, NumPy, Pandas, MatPlotLib, CV2, Keras, Tensorflow/PlaidML**


### Generators:

* [GeneratorStartImage - FINISHED](https://github.com/chaseabrown/BASALT2022/helpers/Generators.py) is a generator class that serves some of the models in *MoveClassifier.py*. For example, the model that detects whether or not the inventory is open only needs the frame before a move is given. 

* [GeneratorEndImage - FINISHED](https://github.com/chaseabrown/BASALT2022/helpers/Generators.py) is a generator class that serves some of the models in *MoveClassifier.py*. For example, the model that detects whether or not the character is told to attack needs the frame after a move is given. 

* [Generator2Images - FINISHED](https://github.com/chaseabrown/BASALT2022/helpers/Generators.py) is a generator class that serves some of the models in *MoveClassifier.py*. For example, the model that detects whether or not the character is moving in a specific direction needs 2 frames so it can compare the two.

* [SegmentationGenerator - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/helpers/Generators.py) is a generator class that serves the model in *BlockSegmentation.py*. It serves the model both the RGB image and the ColorMap images for training.

* [DepthEstGenerator - IN PROGRESS](https://github.com/chaseabrown/BASALT2022/helpers/Generators.py) is a generator class that serves the model in *DepthEstimator.py*. It serves the model both the RGB image and the Depth Map images for training.

**Packages Used: NumPy, PIL, Random, CV2, Math, Keras, Tensorflow/PlaidML**

### Notebooks:

* [Item Classifier - FINISHED]("https://github.com/chaseabrown/BASALT2022/notebooks/Item Classifier.py") is a notebook demonstration of the item classifier in *InvClassifier.py* with an explaination of the process.

  **Packages Used: OS, NumPy, PIL, SkLearn, MatPlotLib, CV2**

* [Item Quantity Classifier - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Item Quantity Classifier.py) is a notebook demonstration of the item quantity classifier in *InvClassifier.py* with an explaination of the process.

  **Packages Used: OS, NumPy, PIL, SkLearn, MatPlotLib, CV2**

* [Data Generator Class Setup - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Data Generator Class Setup.py) is a notebook demonstration of how I step up and tested *StartImageGenerator* and *EndImageGenerator*. 

  **Packages Used: NumPy, PIL, Json, TQDM, Pandas, Random, CV2, Math, Keras**

* [Move Classifier Data Exploration - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Move Classifier Data Exploration.py) is a notebook visualization/exploration of my automatically generated gameplay dataset. This one is fun because it has a lot of pictures and data preprocessing, but this dataset did not end up chosen in the final model.

  **Packages Used: OS, Sys, Json, NumPy, IPython.Display, TQDM, Random**

* [Visualize Model Results (Move Classifier) - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Visualize Model Results (Move Classifier).py) is a notebook demonstrating the model outputs from *MoveClassifier.py*. 

  **Packages Used: OS, NumPy, Pandas, MatPlotLib, PIL, Json, Random, IPython.display**

* [Dataset Exploration (BASALTAgent Moves) - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Dataset Exploration (BASALTAgent Moves).py) is a notebook visualization/exploration of BASALT 2022 competition gameplay dataset. This one is fun because it has a lot of pictures and data preprocessing.

  **Packages Used: OS, NumPy, Pandas, Json, CV2, PIL, IPython.display**
  
* [ColorMap Smoothing - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/ColorMap Smoothing.py) is a notebook exploration at an attempt to smooth the ColorMap data collected. Concluded to not be a great choice for this project, but the smoothing works and is clearly demonstrated and visualized.

  **Packages Used: OS, NumPy, Pandas, TQDM, PIL, CV2**
  
* [KNN Grouping to Match Similar Colors in Image - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/KNN Grouping to Match Similar Colors in Image.py) is a notebook exploration at an attempt to smooth the ColorMap data collected using KNN. Concluded to not be a great choice for this project again, and smoothing was fine, but not great. *ColorMap Smoothing.ipynb* was better.

  **Packages Used: OS, NumPy, PIL, CV2, TDQM**
  
* [Explore Depth and ColorMaps - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Explore Depth and ColorMaps.py) is a notebook visualization of a Malmo dataset I generated for training data to feed into *BlockSegmentator.py* and *DepthEstimator.py*. Lots of visuals, tables, and pretty pictures.

  **Packages Used: OS, NumPy, Pandas, PIL, TQDM, IPython.display, Json, Requests, ShUtil**
  
* [TF-IDF with Spark MineWiki Search Engine - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/TF-IDF with Spark MineWiki Search Engine.py) is a notebook demonstration of how I use TF-IDF to search through the many pages of MineWiki Data distributed by MineDojo. Works really well and uses PySpark for all the data preprocessing, transformation and TF-IDF work which makes it really fast. This is the start of me introducing Natural Language Processing to get more information for my Agent.

  **Packages Used: OS, Pandas, PySpark SQL, PySparkML**

* [Create Search Engine for MineDojo Youtube Dataset - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Create Search Engine for MineDojo Youtube Dataset.py) is a notebook demonstration of how I use TF-IDF to search through transcripts of 70k gameplay Youtube Videos distributed by MineDojo. After the success on the MineWiki search, I brought that to this problem. Doesn't work great because the words said are not as consistent in a transcript as a Wiki page. It still uses PySpark for all the data preprocessing, transformation and TF-IDF work which makes it really fast. After this I need to look into some other options for gathering information from transcripts.

  **Packages Used: Minedojo, IPython.display, Random, TDQM, OS, RE, Pandas, PySpark SQL, PySparkML**

* [Explore MineDojo Gym Observations and Actions - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Explore MineDojo Gym Observations and Actions.py) is a notebook visualization of a the MineDojo Gym Environment's Observation and Action set. 

  **Packages Used: Minedojo, NumPy, Random, PIL**
  
* [Explore MineDojo Tasks - FINISHED](https://github.com/chaseabrown/BASALT2022/notebooks/Explore Depth and ColorMaps.py) is a notebook visualization of a the MineDojo Gym Tasks and Instructions.

  **Packages Used: Minedojo**

### Scripts:
