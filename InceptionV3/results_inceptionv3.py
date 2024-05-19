import pickle as pkl
from tensorflow.keras.models import load_model

# testing set
with open("/home/u395227/dataset/test_normalized.pkl", "rb") as f:
    x_test, y_test = pkl.load(f)
    print("training set loaded")

import tensorflow as tf
from sklearn.metrics import f1_score, classification_report
from keras import backend as K

# adapted from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

tf.keras.utils.get_custom_objects()['f1_m'] = f1_m
tf.keras.utils.get_custom_objects()['precision_m'] = precision_m
tf.keras.utils.get_custom_objects()['recall_m'] = recall_m

# loading model
model = load_model("inceptionv3_50epochsmodel.h5")
print("model loaded")

# Assuming 'model' is your trained model
predictions = model.predict(x_test)

# Convert probabilities to class labels (if necessary)
predicted_classes = predictions.argmax(axis=1)

# Compute F1 score

true_labels = []
for array in y_test:
  for i in range(len(array)):
    if array[i] == 1:
      #print(i)
      true_labels.append(i)

f1_weighted = f1_score(true_labels, predicted_classes, average="weighted")
f1 = f1_score(true_labels, predicted_classes, average=None)
f1_macro = f1_score(true_labels, predicted_classes, average="macro")
f1_micro = f1_score(true_labels, predicted_classes, average="micro")

print(f"F1 Score: {f1}")
print(f"F1 Score weighted: {f1_weighted:.4f}")
print(f"F1 Score macro: {f1_macro:.4f}")
print(f"F1 Score micro: {f1_micro:.4f}")

print(classification_report(true_labels, predicted_classes))
