import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import concat
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow.keras import regularizers
import os
os.environ['TF_CPP_MIN_LEVEL'] = '2'
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn import preprocessing
from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import KFold

physical_devices = tf.config.list_physical_devices('GPU')
data_train = pd.read_csv('data/mnist_train.csv')

x = data_train
x = x.drop(['label'], axis=1)
y = data_train.label

# scaling the training data
#scaler = StandardScaler()
#x= scaler.fit_transform(x)
scaler = Normalizer()
x = scaler.fit_transform(x)
x = pd.DataFrame(x)
#x = x - x.mean()

epoch = 30
splits = 5
kfold = KFold(n_splits=splits, shuffle=True, random_state=2)

loss = []
val_loss = []
accuracy = []
val_accuracy = []
for i, (train, test) in enumerate(kfold.split(x)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=397, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1)))
    model.add(tf.keras.layers.Dense(units=20, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1)))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
    #model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
    #model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.6), loss='mean_squared_error', metrics=['accuracy'])

    #callbacks = [EarlyStopping(monitor='val_accuracy', patience=3)]
    history = model.fit(x.iloc[train], y.iloc[train], epochs=epoch, validation_data=(x.iloc[test], y.iloc[test]))
    los, acc = model.evaluate(x.iloc[test], y.iloc[test])
    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    accuracy.append(history.history['accuracy'])
    val_accuracy.append(history.history['val_accuracy'])
    print(loss)
    print(val_loss)
    print(accuracy)
    print(val_accuracy)

mean_loss = []
mean_val_loss = []
mean_accuracy = []
mean_val_accuracy = []
for i in range(epoch):
    temp_loss = 0
    temp_val_loss = 0
    temp_accuracy = 0
    temp_val_accuracy = 0
    for j in range(splits):
        temp_loss = temp_loss + loss[j][i]
        temp_val_loss = temp_val_loss + val_loss[j][i]
        temp_accuracy = temp_accuracy + accuracy[j][i]
        temp_val_accuracy = temp_val_accuracy + val_accuracy[j][i]
    mean_loss.append(temp_loss/ splits)
    mean_val_loss.append(temp_val_loss / splits)
    mean_accuracy.append(temp_accuracy / splits)
    mean_val_accuracy.append(temp_val_accuracy / splits)

print(mean_loss)
print(mean_val_loss)
print(mean_accuracy)
print(mean_val_accuracy)

# Plot training and Validation
epochs = range(1, len(mean_loss) + 1)
plt.plot(epochs, mean_loss, 'y', label='Training loss')
plt.plot(epochs, mean_val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(["Train", 'Prediction'], loc='upper right')
plt.show()

# Plot training and validation accuracy values
plt.plot(epochs, mean_accuracy, 'y', label='Training acc')
plt.plot(epochs, mean_val_accuracy, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Prediction'], loc='lower right')
plt.show()






