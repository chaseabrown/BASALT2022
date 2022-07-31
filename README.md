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

### Notebooks:

### Scripts:
