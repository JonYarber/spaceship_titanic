import os
os.chdir('/Users/jonyarber/Documents/Projects/spaceship_titanic/scripts')
import carpentry
import subsampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


# Load the train and test set from the carpentry script
ss_titanic_train, ss_titanic_test = carpentry.get_data_frames()


def df_prep(df, scaler):
    
    df_copy = df.copy().drop('PassengerId', axis = 1)
    
    y = df_copy.pop('Transported')
    X = scaler.fit_transform(df_copy)
    
    X = pd.DataFrame(X, columns = df_copy.columns)
    
    return X, y


def build_model(df, neurons, optimizer, initializer):
    
    model = Sequential()
    
    model.add(Input(shape = (df.shape[1],)))
    
    for neuron_layer in neurons:
        
        model.add(Dense(neuron_layer, kernel_initializer = initializer, activation = 'relu'))
        
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = optimizer,
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model


X, y = df_prep(ss_titanic_train, StandardScaler())


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = .15, 
                                                    random_state = 100,
                                                    stratify = y)
    


optimizer = Adam(learning_rate = .01)
neurons = [32, 16, 8]
epochs = 150
batches = 75


model = build_model(df = X, neurons = neurons, optimizer = optimizer, initializer = 'random_normal')
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batches)


accuracy = history.history['accuracy']
loss = history.history['loss']

train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(max(accuracy))
print(min(loss))


#### Visualization ####
plt.figure(figsize = (6, 3))
plt.title("Training Loss")
plt.xlabel('Epochs'),
plt.ylabel('Loss')
plt.plot(np.arange(1, epochs + 1), loss)

plt.figure(figsize = (6, 3))
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(np.arange(1, epochs + 1), accuracy)
plt.show()



