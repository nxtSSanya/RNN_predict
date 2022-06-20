
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping

for dirname, _, filenames in os.walk('\\kaggle\\input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# read data
train = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=col_names)
test = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=col_names)
y_test = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

train.head()
test.head()


def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


train = add_remaining_useful_life(train)
train[index_names + ['RUL']].head()

drop_labels = index_names+setting_names
#dropping the columns except the sensor datas
X_train = train.drop(drop_labels, axis=1)
y_train = X_train.pop('RUL')

X_test = test.groupby('unit_nr').last().reset_index().drop(drop_labels, axis=1)

X_train.shape

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import SimpleRNNCell
from tensorflow.python.keras.layers import RNN
X_trains, X_val, y_trains, y_val = train_test_split(X_train, y_train, test_size=0.2)


model = keras.Sequential()
model.add(Dense(21, activation='relu', input_shape=(21,)))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

initial_learning_rate = 0.1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
optimizedGradient = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=0.3, use_locking=False, name='GradientDescent'
)
optimizer_A = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optimizer_A, loss='mean_absolute_error', metrics=['accuracy'])

model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-7, verbose=1)

history = model.fit(x=X_train, y=y_train,
                    validation_data = (X_val,y_val),
                    epochs = 50,
                    shuffle = True,
                    callbacks=[reduce_lr])


loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, 51)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss for Adam')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1,51)
plt.plot(epochs, acc_train, 'r', label='Training accuracy')
plt.plot(epochs, acc_val, 'g', label='validation accuracy')
plt.title('Training and Validation Accuracy for Custom Arch.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_pred = model.predict(X_test)
print(y_pred)


import sklearn
print(sklearn.metrics.r2_score(y_test, y_pred))
#print(sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred))
print(sklearn.metrics.mean_absolute_error(y_test, y_pred))

from autokeras import StructuredDataRegressor

search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error') #no of trial and errors allowed
search.fit(x=X_train, y=y_train, verbose=1) #fitting the model

mae, acc = search.evaluate(X_test, y_test, verbose=1)

yhat = search.predict(X_test)

print(sklearn.metrics.r2_score(y_test, yhat))
print(sklearn.metrics.mean_absolute_error(y_test, yhat))

model1 = search.export_model()

model1.summary()

model1.save('model1.tf') #saving the model
