import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile


import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

BASE_DIR = 'D:/NUS_1/PRS/project'

numbers_df = pd.read_csv(f"{BASE_DIR}/numbers.csv", dtype={'label': str})

fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(24,4))
numbers_df.groupby(['origin']).count().rename(columns={'file':'origin'})['origin'].plot.bar(ax = axes[0], subplots=True)
numbers_df.groupby(['group']).count().rename(columns={'file':'group'})['group'].plot.bar(ax = axes[1], subplots=True)
numbers_df.groupby(['label']).count().rename(columns={'file':'label'})['label'].plot.bar(ax = axes[2], subplots=True)

plt.show()

numbers_df.head()

def number_sample(df, group, number):
    return df[(df['group'] == group) & (numbers_df['label'] == number)]['file'].iloc[0]

def ax_plot(ax, df, group, number):
    img = number_sample(df, group, number)
    ax.imshow(plt.imread(f"{BASE_DIR}/numbers/{img}"))
    ax.title.set_text(group)

def plot_group_sample(df, groups, number):
    fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(24,4))
    for idx, group in enumerate(groups):
        ax_plot(axes[idx], df, group, number)
    plt.show()

groups = ['Hnd','Fnt','GoodImg','BadImag']
plot_group_sample(numbers_df, groups, '8')

SQUARE_SIZE = 28

def data_flow(data, batch_size):
    return ImageDataGenerator(rotation_range=40, rescale=1./255, shear_range=0.1, zoom_range=0.2, 
                              width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.4, 1.6])\
            .flow_from_dataframe(data, f"{BASE_DIR}/numbers", x_col='file', y_col='label', color_mode='grayscale',
                      target_size=(SQUARE_SIZE, SQUARE_SIZE), class_mode='categorical', batch_size= batch_size, seed=42)

def plot_group_sample_generator(df, group, number):
    
    sample_df = numbers_df[(numbers_df['group'] == group) & (numbers_df['label'] == number)].iloc[0:1]

    fig, axes = plt.subplots(nrows=1,ncols=4,figsize=(24,4))

    img = sample_df['file'].iloc[0]
    axes[0].imshow(plt.imread(f"{BASE_DIR}/numbers/{img}"))
    axes[0].title.set_text('Original')

    transformed = data_flow(sample_df, 1)

    sample_X, _ = next(transformed)
    axes[1].imshow(sample_X[0], cmap='gray')
    axes[1].title.set_text('Trasformed 1')

    sample_X, _ = next(transformed)
    axes[2].imshow(sample_X[0], cmap='gray')
    axes[2].title.set_text('Trasformed 2')

    sample_X, _ = next(transformed)
    axes[3].imshow(sample_X[0], cmap='gray')
    axes[3].title.set_text('Trasformed 3')

    plt.show()
    return sample_X[0]

last_sample = plot_group_sample_generator(numbers_df, 'GoodImg', '8')

print(f"Shape: {last_sample.shape}")
print("First 10 pixels values: ")
last_sample.flatten()[:10]

model = Sequential([
    Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Dropout(0.5),
    Flatten(),
    
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])


model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.summary()

sample_X, sample_y = next(data_flow(numbers_df[(numbers_df['group'] == 'Fnt')], 4))
sample_X.shape

sample_y_prob = model.predict(sample_X)
print(f"Shape: {sample_y_prob.shape}")
sample_y_prob

sample_y_pred = np.argmax(sample_y_prob, axis=1)
sample_y_pred

def plot_predictions(X, y_pred):
    fig, axes = plt.subplots(nrows = 1, ncols = len(X), figsize=(24,4))

    for i in range(len(X)):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].title.set_text(f"Predicted: {y_pred[i]}")

    plt.show()
    
plot_predictions(sample_X, sample_y_pred)

train_data, dummy_data = train_test_split(numbers_df, test_size=0.20, shuffle=True, random_state=42, stratify=numbers_df['label'])
val_data, test_data    = train_test_split(dummy_data, test_size=0.50, shuffle=True, random_state=42, stratify=dummy_data['label'])

train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

batch_size = 32
train_gen  = data_flow(train_data, batch_size)
val_gen    = data_flow(val_data, batch_size)
test_gen   = data_flow(test_data, batch_size)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

earlystop = EarlyStopping(monitor="val_loss", patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor= 'val_accuracy', patience= 2, verbose= 1,
                                            factor= 0.5, min_lr= 0.00001)
checkpoint = ModelCheckpoint('./models/sudoscan.h5', monitor='val_accuracy', save_best_only=True, 
                             save_weights_only=False, mode='auto', save_freq='epoch')

epochs = 100
history = model.fit(
    train_gen, epochs= epochs, validation_data= val_gen,
    callbacks= [checkpoint, earlystop, learning_rate_reduction]
)