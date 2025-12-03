import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models

# Assuming you already have this class imported
dp = DataProcesser("data.pkl")
train_x, train_y, valid_x, valid_y, test_x, test_y = dp.load_data()

# Reshape data to add channel dimension (grayscale)
train_x = np.expand_dims(train_x, -1)
valid_x = np.expand_dims(valid_x, -1)
test_x  = np.expand_dims(test_x, -1)

# One-hot encode labels
num_classes = 10  # digits 0-9
train_y = keras.utils.to_categorical(train_y, num_classes)
valid_y = keras.utils.to_categorical(valid_y, num_classes)
test_y  = keras.utils.to_categorical(test_y, num_classes)

# Build a strong custom CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_x, train_y, epochs=15, batch_size=64, validation_data=(valid_x, valid_y))

# Evaluate
test_loss, test_acc = model.evaluate(test_x, test_y)
print(f"Test Accuracy: {test_acc:.4f}")
