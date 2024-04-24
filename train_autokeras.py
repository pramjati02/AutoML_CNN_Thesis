import numpy as np
import tensorflow as tf
import pickle as pkl

import autokeras as ak

# Loading pickle sets

# training set 
with open("/home/u395227/dataset/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)
    print("training set loaded")

# Validation set
with open("/home/u395227/dataset/validation.pkl", "rb") as f:
    x_val, y_val = pkl.load(f)
    print("validation set loaded")

# Initialize the image classifier.
clf = ak.ImageClassifier(overwrite=True, max_trials=15)

clf.fit(
    x_train,
    y_train,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    epochs=50,
)

# Export as a Keras Model.
model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

model.save("model_autokeras.keras")
