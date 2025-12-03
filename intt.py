import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image  # for processing new images

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

        train_xs = np.array([list(td["image"].flatten() / 255) for td in training_dataset])
        train_ys = np.array([td["label"] for td in training_dataset])

        train_x, valid_x, train_y, valid_y = train_test_split(
            train_xs, 
            train_ys, 
            test_size = 0.1,
            random_state = 42,
            stratify = train_ys,
            shuffle=True
        )

        test_x = np.array([list(td["image"].flatten() / 255) for td in testing_dataset])
        test_y = np.array([td["label"] for td in testing_dataset])

        return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    def display_img(self, data_x, data_y, img_num=5):
        plt.figure(figsize=(20,20))
        for i in range(img_num):
            plt.subplot(1, img_num, i+1)
            plt.title(f"Label - {data_y[i]}")
            plt.imshow(data_x[i].reshape(28,28), cmap="Greys")
        plt.show()

    def predict_new_image(self, img_path, model):
        """
        Predict a handwritten digit from an image file using the trained model.
        img_path : str / Path : path to image
        model    : trained Keras CNN model
        """
        # Open image
        img = Image.open(img_path).convert("L")  # convert to grayscale
        img = img.resize((28,28))                # resize to 28x28

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # add channel
        img_array = np.expand_dims(img_array, axis=0)   # add batch

        # Predict
        pred = model.predict(img_array)
        predicted_digit = np.argmax(pred)
        
        # Optionally display image
        plt.imshow(img_array[0,:,:,0], cmap="Greys")
        plt.title(f"Predicted: {predicted_digit}")
        plt.show()
        
        return predicted_digit
