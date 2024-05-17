import pickle as pkl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os 
import glob

# Load training set
with open("/home/u395227/dataset/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)
    print("Training set loaded")

# Load validation set
with open("/home/u395227/dataset/validation.pkl", "rb") as f:
    x_val, y_val = pkl.load(f)
    print("Validation set loaded")

print(x_train.shape)
print(x_val.shape)
#print(x_test.shape, "\n")

# Normalized sets
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_val_reshaped = x_val.reshape(x_val.shape[0], -1)
#x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

print(x_train_reshaped.shape)
print(x_val_reshaped.shape)
#print(x_test_reshaped.shape, "\n")

scaler = MinMaxScaler()
x_train_normalized = scaler.fit_transform(x_train_reshaped)
x_val_normalized = scaler.transform(x_val_reshaped)
#x_test_normalized = scaler.transform(x_test_reshaped)

# Reshape the normalized data back to its original shape
x_train_normalized = x_train_normalized.reshape(x_train.shape)
x_val_normalized = x_val_normalized.reshape(x_val.shape)
#x_test_normalized = x_test_normalized.reshape(x_test.shape)

print(x_train_normalized.shape, x_val_normalized.shape)

# Setting up the pre-trained ResNet50 model
model = Sequential()
model.add(ResNet50(include_top=False, weights="imagenet", input_shape=(128,128,3), pooling="avg"))
model.add(BatchNormalization())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(400,activation='relu'))
model.add(Dense(6,activation='softmax'))

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

# Setting up data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy',f1_m,precision_m, recall_m])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model using data augmentation
history1 = model.fit(datagen.flow(x_train_normalized, y_train, batch_size=50),
                      steps_per_epoch=len(x_train_normalized) / 50,  # Steps per epoch based on batch size
                      epochs=50,
                      validation_data=(x_val_normalized, y_val),
                      verbose=2)

# Save the model
model.save("resnet50_50epochsmodel.h5")

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
metrics_df.to_csv('resnet50_50epochsmetrics.csv', index=False)

