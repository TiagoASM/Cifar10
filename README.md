# 🧠 Image Classification with CIFAR-10

This project aims to train and provide a deep learning model for **image classification** using the well-known **CIFAR-10** dataset.

The final trained model is saved as: `modelo_cifar10.h5`.
Go to master branch.

---

##  About CIFAR-10

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a dataset consisting of **60,000 color images** (32x32 pixels), divided into 10 classes:

- Airplane ✈️  
- Automobile 🚗  
- Bird 🐦  
- Cat 🐱  
- Deer 🦌  
- Dog 🐶  
- Frog 🐸  
- Horse 🐎  
- Ship 🚢  
- Truck 🚛  

---

## 🚀 How to Get the Project

To clone this repository, run the following command in your terminal:

```bash
git clone https://github.com/TiagoASM/Cifar10.git
```
```bash
pip install tensorflow numpy matplotlib pillow
```

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("modelo_cifar10.h5")

# Load and preprocess your image
img_path = "your_image.jpg"  # Replace with your image path
img = Image.open(img_path).resize((32, 32))
img_array = np.array(img) / 255.0  # Normalize pixel values

# Ensure it's in the right shape: (1, 32, 32, 3)
if img_array.shape != (32, 32, 3):
    raise ValueError("Image must have 3 color channels (RGB).")

img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# CIFAR-10 class labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Show the image and prediction
plt.imshow(np.squeeze(img_array))
plt.title(f"Predicted class: {class_names[predicted_class]}")
plt.axis("off")
plt.show()
```

