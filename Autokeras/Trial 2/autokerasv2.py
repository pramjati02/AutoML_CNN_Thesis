import numpy as np
import tensorflow as tf
import pickle as pkl
import autokeras as ak
#from autokeras.callbacks import BestModel

# Loading pickle sets

# training set 
with open("/home/u395227/dataset/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)
    print("training set loaded")

# Validation set
with open("/home/u395227/dataset/validation.pkl", "rb") as f:
    x_val, y_val = pkl.load(f)
    print("validation set loaded")

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Normalize the dataset.
    normalize=False,
    # Do not do data augmentation.
    augment=False,
)(input_node)
output_node = ak.ClassificationHead()(output_node)

# Initialize the image classifier.
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=50
)

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

