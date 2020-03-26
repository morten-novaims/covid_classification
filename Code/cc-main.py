# ========================================================================================================== #
# ========================================================================================================== #
#
# Covid-19: Deep Diagnostics
#
# This project uses deep learning methods to classify lung conditions based on chest x-ray data.
# Specifically, the model differentiates between covid-19 and healthy cases.
#
# ========================================================================================================== #
# ========================================================================================================== #
# Setup ------------------------------------------------------------------------------------------------------
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Reshape, Activation, \
    BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_roc
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import neptune

# Set seed
seed = np.random.randint(100)

# Functions --------------------------------------------------------------------------------------------------

# Not working here and I couldn't figure out why. Same code in a separate notebook
#split_folders = True
#if split_folders:
#    print("Making folder split...")
#    shutil.rmtree("Data/split", ignore_errors=True)
#    split_folders.ratio("Data/train", output="Data/split", seed=1337) # default values
#    print("Done")

# Adding neptune to the project
neptune.init('morten/covid-classification')
neptune.create_experiment('covid-neptune-1')

class NeptuneLoggerCallback(Callback):
    def __init__(self, model, validation_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'batch_{log_name}', log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)
#        Leaving this commented as plots were not wokring with multiclass data            
#        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
#        y_true = self.validation_data[1]#

#        y_pred_class = np.argmax(y_pred, axis=1)

#        fig, ax = plt.subplots(figsize=(16, 12))
#        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
#        neptune.log_image('confusion_matrix', fig)

#        fig, ax = plt.subplots(figsize=(16, 12))
#        plot_roc(y_true, y_pred, ax=ax)
#        neptune.log_image('roc_curve', fig)

# Start ==================================================================================================== #
# Load data --------------------------------------------------------------------------------------------------
# Set up directories
main_dir = os.getcwd()
code_dir = os.path.join(main_dir, 'Code')
data_dir = os.path.join(main_dir, 'Data')
output_dir = os.path.join(main_dir, 'Output')

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')
train_n = os.path.join(train_dir, 'NORMAL')
train_c = os.path.join(train_dir, 'CORONA')
train_p = os.path.join(train_dir, 'PNEUMONIA')
train_o = os.path.join(train_dir, 'OTHER')
# test_n = os.path.join(test_dir, 'NORMAL')
# test_c = os.path.join(test_dir, 'CORONA')
# test_p = os.path.join(test_dir, 'PNEUMONIA')
# test_o = os.path.join(test_dir, 'OTHER')
# val_n = os.path.join(val_dir, 'NORMAL')
# val_c = os.path.join(val_dir, 'CORONA')
# val_p = os.path.join(val_dir, 'PNEUMONIA')
# val_o = os.path.join(val_dir, 'OTHER')

# Explore data (folder) structure
print(main_dir)
print(os.listdir(main_dir))
print(train_dir)
print(len(os.listdir(train_n)))
print(len(os.listdir(train_c)))
print(len(os.listdir(train_p)))
print(len(os.listdir(train_o)))

# Plot images ------------------------------------------------------------------------------------------------
# Save file names
train_n_names = os.listdir(train_n)
train_c_names = os.listdir(train_c)
train_p_names = os.listdir(train_p)
train_o_names = os.listdir(train_o)


# test_n_names = os.listdir(test_n)
# test_c_names = os.listdir(test_c)
# test_p_names = os.listdir(test_p)
# test_o_names = os.listdir(test_o)
# val_n_names = os.listdir(val_n)
# val_c_names = os.listdir(val_c)
# val_p_names = os.listdir(val_p)
# val_o_names = os.listdir(val_o)


# Plot first 10 (or individually set amount of) images of each normal, covid-19, and pneumonia x-rays from a given directory
def plt_classes(img_dir=train_dir, img_per_class=10, img_size=(10, 10)):
    plt.figure(figsize=img_size)
    for i in range(img_per_class):
        plt.subplot(6, img_per_class // 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x = plt.imread(os.path.join(os.path.join(img_dir, 'NORMAL'), os.listdir(os.path.join(img_dir, 'NORMAL'))[i]))
        plt.imshow(x, cmap='gray')
        plt.xlabel('Normal, no. {:02}'.format(i + 1))
    for i in range(img_per_class):
        plt.subplot(6, img_per_class // 2, i + 1 + img_per_class)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x = plt.imread(os.path.join(os.path.join(img_dir, 'CORONA'), os.listdir(os.path.join(img_dir, 'CORONA'))[i]))
        plt.imshow(x, cmap='gray')
        plt.xlabel('Covid-19, no. {:02}'.format(i + 1))
    for i in range(img_per_class):
        plt.subplot(6, img_per_class // 2, i + 1 + img_per_class * 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x = plt.imread(
            os.path.join(os.path.join(img_dir, 'PNEUMONIA'), os.listdir(os.path.join(img_dir, 'PNEUMONIA'))[i]))
        plt.imshow(x, cmap='gray')
        plt.xlabel('Pneumonia, no. {:02}'.format(i + 1))
    plt.tight_layout()  # Ensure image labels are not concealed
    plt.suptitle('Chest x-rays of 10 normal, covid-19, and pneumonia cases, respectively.', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Space between suptitle and images
    plt.show()
    plt.savefig(os.path.join(output_dir, 'fig-xray-classes.png'))
    plt.close()

plt_classes()

# Data pre-processing ----------------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.15,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=33,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  validation_split=0.2)

# Set parameters
image_size = (128, 128)
train_batch_size = 16
vt_batch_size = 16

# Now actually preprocess training, test, and validation data
# To Do: Either store data in separate physical folders, or slightly modify source code from ImageDataGenerator (e.g. see here: https://stackoverflow.com/questions/51952231/keras-how-to-expand-validation-split-to-generate-a-third-set-i-e-test-set)
train_set = train_datagen.flow_from_directory(train_dir,
                                              target_size=image_size,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              class_mode='categorical',
                                              color_mode='grayscale',
                                              subset='training',
                                              seed=seed)
# test_set = test_datagen.flow_from_directory(train_dir,  # Use again train_dir as all data is stored here
#                                             target_size=image_size,
#                                             batch_size=vt_batch_size,
#                                             class_mode='categorical',
#                                             # alternatively, could also use 'binary', which would however affect the model design and compliation
#                                             color_mode='grayscale',
#                                             subset='testing',
#                                             seed=seed)
val_set = test_datagen.flow_from_directory(train_dir,  # Use again train_dir as all data is stored here
                                           target_size=image_size,
                                           batch_size=vt_batch_size,
                                           class_mode='categorical',
                                           color_mode='grayscale',
                                           subset='validation',
                                           seed=seed)

# Build model ------------------------------------------------------------------------------------------------
# Using the Keras function API
img_input = Input(shape=train_set.image_shape, name='img_input')

x = Conv2D(filters=32, kernel_size=6, padding='same', use_bias=False, name='1nd_Conv2D')(
    img_input) # no bias necessary before batch norm
x = BatchNormalization(scale=False, center=True)(x) # no batch norm scaling necessary before "relu"
x = Activation('relu')(x) # activation after batch norm

x = Conv2D(filters=24, kernel_size=3, padding='same', use_bias=False, strides=2, name='2nd_Conv2D')(x)
x = BatchNormalization(scale=False, center=True)(x)
x = Activation('relu')(x)

x = Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False, strides=1, name='3nd_Conv2D')(x)
x = BatchNormalization(scale=False, center=True)(x)
x = Activation('relu')(x)

x = Flatten()(x)
x = Dense(200, use_bias=False)(x)
x = BatchNormalization(scale=False, center=True)(x)
x = Activation('relu')(x)
x = Dense(100, use_bias=False)(x)
x = BatchNormalization(scale=False, center=True)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)  # Dropout on dense layer only
output = Dense(4, activation='softmax', name='img_output')(x)

model = Model(inputs=img_input, outputs=output, name='func_model')

# Set learning rate
set_lr = 0.00015
# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=set_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


neptune_logger = NeptuneLoggerCallback(model=model,
                                       validation_data=val_set)

history = model.fit(train_set,
                    steps_per_epoch=train_set.n // train_set.batch_size,
                    epochs=10,
                    validation_data=val_set,
                    validation_steps=val_set.n // val_set.batch_size,
                   callbacks=[neptune_logger])

# Plot results
def plt_acc_loss(model=history):
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    # Subplot 1: Accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    # Subplot 2: Loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.show()
    plt.savefig(os.path.join(output_dir, 'fig_train-val_acc-loss.png'))
    plt.close()

plt_acc_loss(model=history)

# # Evaluate model ---------------------------------------------------------------------------------------------
# results = model.evaluate(test_set)
# print('Model loss evaluated on the test data: {:.2f}%'.format(results[0] * 100))
# print('Model accuracy evaluated on the test data: {:.2f}%'.format(results[1] * 100))
