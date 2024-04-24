import tensorflow as tf
import pickle as pkl
from tensorflow.keras.models import load_model
import autokeras as ak 
from sklearn.metrics import f1_score, classification_report

# testing set
with open("/home/u395227/dataset/test.pkl", "rb") as f:
    x_test, y_test = pkl.load(f)
    print("training set loaded")

# Load the saved model
#model = tf.saved_model.load('/home/u395227/dataset/autokeras/image_classifier/best_model/')

model = load_model(
    "model_autokeras.keras", custom_objects=ak.CUSTOM_OBJECTS
)

print("model loaded")
#print(type(model))
#print(dir(model)) 

model.summary()

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

