"""
Description: Network training-related programs.
"""
import numpy as np
import tensorflow as tf #import models
from tensorflow.python.keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import simpler_CNN,mini_XCEPTION,tiny_XCEPTION,simple_CNN,big_XCEPTION
from sklearn.model_selection import train_test_split

# set param
batch_size = 32#32
num_epochs = 10000
input_shape = (48, 48, 1)#48
validation_split = .2
verbose = 1
num_classes = 7
patience = 50#Tolerate up to 50 epochs with no boost in val_loss.
base_path = 'models/'



# build a model
model = big_XCEPTION(input_shape, num_classes)#mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer="adam", # The optimizer uses adam
              loss='mse', #He has a built-in softmax operation, which OP computes the cross-entropy loss between the inputinput and the labellabel , which combines the OP computation of LogSoftmax and NLLLoss and can be used to train an n-class classifier.
              #The reason for this is that the CrossEntropyLoss function actually has LogSoftmax and NLLLoss built in, which means that once you use CrossEntropyLoss, it will automatically give you the predictions to LogSoftmax and NLLLoss when calculating the predicted and labeled values
              metrics=['acc'])#loss='categorical_crossentropy'
model.summary()



# Define the callback function earlystop in Callbacks to be used for the training process
log_file_path = base_path + '_emotion_training - big_XCEPTION2+mse64@.log'#log of training process,change when training
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4),
                              verbose=1)
# Model Location and Naming
trained_models_path = base_path + '_big_XCEPTION_2+mse64'#'_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-acc{acc:.2f}.hdf5'#'.{epoch:02d}-{val_acc:.2f}.hdf5'

# Define model weight locations, naming, etc.
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                    save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]



# Load Data Set
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# Divide training, test sets
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)#train_test_split

# Image generator, augmenting data in batch, expanding dataset size
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,   #revolve
                        width_shift_range=0.1,  #pan left and right
                        height_shift_range=0.1,  #pan up and down
                        zoom_range=.1,  #resizing
                        horizontal_flip=True)  #Horizontal Flip

# Training with data augmentation
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),#model.fit
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
