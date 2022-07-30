# U-Net based segmentation model class called BlockSegmentation.
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras import layers
sys.stderr = stderr


class BlockSegmentator():
   
    def __init__(self, inputShape, numClasses):
        self.inputShape = inputShape
        self.numClasses = numClasses
        # Build model
        model = self.__get_model(inputShape, numClasses)
        model.summary()

    def __get_model(img_size, num_classes):
        inputs = keras.Input(shape=img_size + (3,))

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
        # Add a per-pixel classification layer    
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
        # Define the model
        model = keras.Model(inputs, outputs)
        return model
    
    def train(self, train_gen, val_gen, epochs=15):
        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.
        model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

        callbacks = [
            keras.callbacks.ModelCheckpoint("block_segmentation.h5", save_best_only=True)
        ]

        # Train the model, doing validation at the end of each epoch.
        model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

