#------------------Model Building
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os,re
import cv2,uuid

def trainingdf():
    train_path = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\data\traintestval\train'
    train_datagen = ImageDataGenerator(rescale=1./255,
                   horizontal_flip=0.4,
                   vertical_flip=0.4,
                   rotation_range=40,
                   shear_range=0.2,
                   width_shift_range=0.4,
                   height_shift_range=0.4,
                   fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(train_path,batch_size=32,target_size=(240,240),
                                  class_mode='categorical',
                                  shuffle=True,
                                  seed=42,color_mode='rgb')
    return train_generator

def validationdf():

    val_path = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\data\traintestval\val'
     
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    val_generator = val_datagen.flow_from_directory(val_path,batch_size=32,target_size=(240,240),
                                  class_mode='categorical',
                                  shuffle=True,
                                  seed=42,color_mode='rgb')
    return val_generator

def testdf():
    test_path = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\data\traintestval\test'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_path,batch_size=32,target_size=(240,240),
                                  class_mode='categorical',
                                  shuffle=True,
                                  seed=42,color_mode='rgb')
    return test_generator

def classlabel(data_generator):
    class_labels = data_generator.class_indices
    class_name = {value:key for (key,value) in class_labels.items()}
    print(class_name)

def ModelBuilding(traindf,valdf):

    base_model = VGG19(input_shape=(240,240,3),include_top=False,
                   weights='imagenet')

    for layers in base_model.layers:
        layers.trainable=False
    
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608,activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152,activation='relu')(drop_out)
    output = Dense(4,activation='softmax')(class_2)

    model_01 = Model(base_model.input,output)
    model_01.summary()

# Define the paths and filenames
    path = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\models'
    filepath = os.path.join(path, 'model.keras')

# Define the callbacks
    es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=4)
    cp = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    lrr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.0001)

# Define the optimizer
    sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model_01.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    historymodel = model_01.fit(traindf, steps_per_epoch=2, epochs=4, callbacks=[es, cp, lrr],
                                validation_data=valdf)
    return historymodel


def ModelInfoPlot(info):
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))
  ax[0].plot(info.history['accuracy'], color='green', label='Train Accuracy')
  ax[0].plot(info.history['val_accuracy'], ls='--', label='Val Accuracy')
  ax[0].set_title('Accuracy')
  ax[0].legend()  # Set legend for the accuracy plot

# Plot loss
  ax[1].plot(info.history['loss'], label='Train Loss')
  ax[1].plot(info.history['val_loss'], ls='--', label='Val Loss')
  ax[1].set_title('Loss')
  ax[1].legend()
  plt.savefig(os.path.join(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\reports\figures',f"{uuid.uuid1()}.png"))# Set legend for the loss plot
  plt.show()

