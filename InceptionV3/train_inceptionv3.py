import pickle as pkl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import pandas as pd
import numpy as np
import os 
import glob

# training set 
with open("/home/u395227/dataset/train_normalized.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)
    print("training set loaded")

# Validation set
with open("/home/u395227/dataset/validation_normalized.pkl", "rb") as f:
    x_val, y_val = pkl.load(f)
    print("validation set loaded")

# Setting up the pre-trained InceptionV3 model

model = Sequential()
model.add(InceptionV3(include_top=False, weights="imagenet", input_shape=(128,128,3), pooling="avg"))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(400, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

model.layers[0].trainable = False
model.summary()

# setting up calculations for f1 score, precision and recall during training 
def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Setting up data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy',f1_m,precision_m, recall_m])

# Fit the model using data augmentation
history1 = model.fit(datagen.flow(x_train, y_train, batch_size=50),
                      steps_per_epoch=len(x_train) / 50,  # Steps per epoch based on batch size
                      epochs=50,
                      validation_data=(x_val, y_val),
                      verbose=2)

# save the inceptionv3 classification model as a pickle file
model.save("inceptionv3_50epochsmodel.h5")

# Extract metrics from history1
training_loss = history1.history['loss']
training_accuracy = history1.history['accuracy']
validation_loss = history1.history['val_loss']
validation_accuracy = history1.history['val_accuracy']
precision_m_train = history1.history['precision_m']
recall_m_train = history1.history['recall_m']
f1_m_train = history1.history['f1_m']
precision_m_val = history1.history['val_precision_m']
recall_m_val = history1.history['val_recall_m']
f1_m_val = history1.history['val_f1_m']

# Combine metrics into a DataFrame
metrics_df = pd.DataFrame({
    'training_loss': training_loss,
    'training_accuracy': training_accuracy,
    'validation_loss': validation_loss,
    'validation_accuracy': validation_accuracy,
    'precision_m_train': precision_m_train,
    'recall_m_train': recall_m_train,
    'f1_m_train': f1_m_train,
    'precision_m_val': precision_m_val,
    'recall_m_val': recall_m_val,
    'f1_m_val': f1_m_val
})

# Save the DataFrame to a CSV file
metrics_df.to_csv('inceptionv3_50epochsmetrics.csv', index=False)
