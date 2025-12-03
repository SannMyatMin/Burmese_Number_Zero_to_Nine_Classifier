import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

class BurmeseNumberClassifier:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, test_y, class_names, epochs=50, batch_size=64):
        self.train_x       = train_x
        self.train_y       = train_y
        self.valid_x       = valid_x
        self.valid_y       = valid_y
        self.test_x        = test_x
        self.test_y        = test_y
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.class_names   = class_names
        self.total_classes = len(class_names)
        self.early_stop    = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.model         = None

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(28, 28, 1)),
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
    
    def _compile_model(self):
        self.model.compile(
            optimizer = "adam",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        )
        print("Model is compiled succesfully")

    def train_model(self):
        self._compile_model()
        training_history = self.model.fit(
            self.train_x,
            self.train_y,
            epochs = self.epochs,
            batch_size = self.batch_size,
            validation_data = (self.valid_x, self.valid_y),
            callbacks = [self.early_stop]
        )
        return training_history

    def evaluate_model_performance(self):
        loss, accuracy = self.model.evaluate(self.test_x, self. test_y)
        print(f"Model testing accuracy : {accuracy*100:.3f}%")
        return accuracy, loss
    
    