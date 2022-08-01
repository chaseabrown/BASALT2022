# BASALT 2022 Models

## Inventory Classifier (InvClassifier.py)

InventoryClassifier is a class with 3 functions: 
1. Read image of inventory items and predict the item type (Gaussian Naive Bayes model)
2. Read image of inventory items and predict the item quantity (Gaussian Naive Bayes model)
3. Read an image of the full screen and get the cursor location using a Convolutional Neural Network (CNN)

[![Inventory Classifier](https://github.com/chaseabrown/BASALT2022/blob/master/assets/github-images/InvClassifier%20Diagram.jpeg)](https://github.com/chaseabrown/BASALT2022/blob/master/models/InvClassifier.py)

This works really well. Basically only makes mistakes when the items are identical such as Chests and Trapped Chests which can't be helped.

## Deep Q-Learning Network (DQN.py)

DQN is a standard Deep Q-Learning Network which I intend to use for specific training situations that OpenAI's VPT Model isn't designed to handle.

[![Inventory Classifier](https://github.com/chaseabrown/BASALT2022/blob/master/assets/github-images/DQN%20Diagram.png)](https://github.com/chaseabrown/BASALT2022/blob/master/models/DQN.py)

This hasn't been implimented into this project yet, but I used this same model to beat the Cartpole game with great success.

## Move Classifier (MoveClassifier.py)

MoveClassifier is a class with the sole purpose of reading 2 frame images from a video of gameplay and return the predicted move that was made between frames. My model uses 2 Convolutional Neural Networks (CNNs) that are combined at the ends finished with an activation function specific to each move type.

[![Move Classifier](https://github.com/chaseabrown/BASALT2022/blob/master/assets/github-images/MoveClassifier%20Diagram.jpeg)](https://github.com/chaseabrown/BASALT2022/blob/master/models/MoveClassifier.py)

I got pretty far with this, before I realized the work was already done by OpenAI's VPT Inverse Dynamics Model and decided to use that instead. My model had a really high success rate for inventory, hotbar, use and attack. I still needed some work with movements but didn't get that far.

## Block Segmentation (BlockSegmentation.py)

BlockSegmentator is a class with the sole purpose of reading in an RGB Frame and returning a ColorMap that represents the block at each pixel. The model uses Keras for Semantic Segmentation with a U-Net Architecture.

[![Block Segmentation](https://github.com/chaseabrown/BASALT2022/blob/master/assets/github-images/Block%20Segmentation%20Diagram.jpg)](https://github.com/chaseabrown/BASALT2022/blob/master/models/BlockSegmentation.py)

As I write this, I am at the early stages of this process.

## Depth Estimator (DepthEstimator.py)

DepthEstimator is a class with the sole purpose of reading in an RGB Frame and returning a depth map at each pixel. The model uses Keras for a Deep Learning Autoencoder.

[![Depth Estimator](https://github.com/chaseabrown/BASALT2022/blob/master/assets/github-images/Depth%20Estimation%20Diagram.jpg)](https://github.com/chaseabrown/BASALT2022/blob/master/models/DepthEstimator.py)

As I write this, I am at the early stages of this process.
