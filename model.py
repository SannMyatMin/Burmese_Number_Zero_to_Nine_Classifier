import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models

class BurmeseNumberPredictor:
    def __init__(self, train_x, train_y, valix_x, valid_y, test_x, test_y, class_names):
        self.train_x       = train_x
        self.train_y       = train_y
        self.valid_x       = valix_x
        self.valid_y       = valid_y
        self.test_x        = test_x
        self.test_y        = test_y
        self.class_names   = class_names
        self.total_classes = len(class_names)
        self.model         = None

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation="relu", padding="same", input_reshape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.total_classes, activation="softmax")
        ])

        self.model = model
        return self.model
    