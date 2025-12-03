import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

class BurmeseNumberPredictor:
    def __init__(self, model_path, class_names):
        self.model       = keras.models.load_model(model_path)
        self.class_names = class_names
       
    def predict_number(self, img):

        if img.endswith((".jpg", ".png", ".jpeg")):
            image = Image.open(img).convert("L")
            image = image.resize((28,28))
            img_array = np.array(image)
        else:
            raise ValueError("Invalid file type")
        
        img_array = img_array / 255
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        confidense = float(np.max(prediction))
        predicted_index = np.argmax(prediction)
        predicted_label = self.class_names[predicted_index]

        return predicted_label, confidense

