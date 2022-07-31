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

MoveClassifier is a class with the sole purpose of reading 2 frame images from a video of gameplay and return the predicted move that was made between frames. I got pretty far with this, before I realized the work was already done by OpenAI's VPT Inverse Dynamics Model and decided to use that instead. My model uses 2 Convolutional Neural Networks (CNNs) that are combined at the ends finished with an activation function specific to each move type.

[![Move Classifier](https://github.com/chaseabrown/BASALT2022/blob/master/assets/github-images/MoveClassifier%20Diagram.jpeg)](https://github.com/chaseabrown/BASALT2022/blob/master/models/MoveClassifier.py)

