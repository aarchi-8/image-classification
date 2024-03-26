# image-classification
Machine learning project of image classification using cnn and datasets cifar10 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape
X_test.shape
y_train.shape
y_train[:5]y_train = y_train.reshape(-1,)
y_train[:5]
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plot_sample(X_train, y_train, 23)
plot_sample(X_train, y_train, 65)
X_train = X_train / 255.0
X_test = X_test / 255.0
ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=10)
#step-5:cnn evaluation
cnn.evaluate(X_test,y_test)
y_pred = cnn.predict(X_test)
y_pred[:5]
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]
y_test[:5]
plot_sample(X_test, y_test,6)
classes[y_classes[60]]
from flask import Flask, render_template, request, jsonify, send_file
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import decode_predictions
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        img_path = os.path.join('static/uploads/cifar10_cnn.h5')
        file.save(img_path)

        # Preprocess the image for the VGG16 model
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction with the VGG16 model
        predictions = model.predict(img_array)
        label = decode_predictions(predictions, top=1)[0][0][1]

        # Return the predicted label
        return jsonify({"label": label})

if __name__ == '__main__':
    app.run(debug=True)
import tensorflow as tf
import numpy as np

class Cifar10Classifier:
    def __init__(self):
        self.model = tf.keras.models.load_model('static/models/cifar10_cnn.h5')

    def predict(self, img_path):
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(32, 32)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255

        predictions = self.model.predict(img_array)
        top3_indices = np.argsort(predictions[0])[-3:]
        top3_classes = [
            ('Class {}: {:.2f}%'.format(i, predictions[0][i] * 100))
            for i in top3_indices
        ]

        return top3_classes
from IPython.display import HTML
HTML(filename='//content//index.html')
