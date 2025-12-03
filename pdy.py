# Pick the first test image
new_img = test_x[0]  # shape (28,28,1)
new_img_exp = np.expand_dims(new_img, axis=0)  # add batch dimension

pred = model.predict(new_img_exp)
predicted_digit = np.argmax(pred)
print("Predicted digit:", predicted_digit, "Actual:", np.argmax(test_y[0]))




dp = DataProcesser("data.pkl")

# After training your CNN (model)
predicted_digit = dp.predict_new_image("my_handwritten_digit.jpg", model)
print("Predicted digit:", predicted_digit)
