import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, dataset_dir="data.pkl"):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = ["၀","၁","၂","၃","၄","၅","၆","၇","၈","၉"]
        with open(self.dataset_dir, "rb") as file:
            self.dataset = pickle.load(file)

    def _load_data(self):
        training_dataset = self.dataset["trainDataset"]
        testing_dataset  = self.dataset["testDataset"]

        train_xs = np.array([td["image"] / 255 for td in training_dataset])
        train_ys = np.array([td["label"] for td in training_dataset])

        train_x, valid_x, train_y, valid_y = train_test_split(
            train_xs, 
            train_ys, 
            test_size = 0.1,
            random_state = 42,
            stratify = train_ys,
            shuffle=True
        )

        test_x = np.array([td["image"] / 255 for td in testing_dataset])
        test_y = np.array([td["label"] for td in testing_dataset])

        return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    def get_train_valid_test(self):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self._load_data()

        train_x = np.expand_dims(train_x, -1)
        valid_x = np.expand_dims(valid_x, -1)
        test_x  = np.expand_dims(test_x, -1)

        num_classes = len(self.class_names)
        train_y = keras.utils.to_categorical(train_y, num_classes)
        valid_y = keras.utils.to_categorical(valid_y, num_classes)
        test_y  = keras.utils.to_categorical(test_y, num_classes)

        return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    def display_img(self, data_x, data_y, img_num=5):
        plt.figure(figsize=(20,20))
        for i in range(img_num):
            plt.subplot(1, img_num, i+1)
            plt.title(f"Label - {data_y[i]}")
            plt.imshow(data_x[i].reshape(28,28), cmap="Greys")
        plt.show()