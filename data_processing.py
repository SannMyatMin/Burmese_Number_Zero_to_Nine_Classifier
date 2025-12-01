import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class DataProcesser:
    def __init__(self, dataset_dir="data.pkl"):
        self.dataset_dir = Path(dataset_dir)
        dataset = []
        with open(self.dataset_dir, "rb") as file:
            dataset = pickle.load(file)
        self.dataset = dataset

    def load_data(self):
        training_dataset = self.dataset["trainDataset"]
        testing_dataset  = self.dataset["testDataset"]

        train_x = np.array([list(td["image"].flatten() / 255) for td in training_dataset])
        train_y = np.array([td["label"] for td in training_dataset])

        test_x = np.array([list(td["image"].flatten() / 255) for td in testing_dataset])
        test_y = np.array([td["label"] for td in testing_dataset])

        return train_x, train_y, test_x, test_y
    
    def display_img(self, data_x, data_y, img_num=5):
        plt.figure(figsize=(20,20))
        for i in range(img_num):
            plt.subplot(1, img_num, i+1)
            plt.title(f"Label - {data_y[i]}")
            plt.imshow(data_x[i].reshape(28,28), cmap="gray")
        plt.show()