"""
Description: 训练人脸表情识别程序
"""
import numpy as np
import tensorflow as tf #导入模块
from tensorflow.python.keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import simpler_CNN,mini_XCEPTION,tiny_XCEPTION,simple_CNN,big_XCEPTION
from sklearn.model_selection import train_test_split

# 参数
batch_size = 32#32
num_epochs = 10000
input_shape = (48, 48, 1)#48
validation_split = .2
verbose = 1
num_classes = 7
patience = 50#最多可以忍受50次epoch后val_loss没有提升
base_path = 'models/'



# 构建模型
model = big_XCEPTION(input_shape, num_classes)#mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer="adam", # 优化器采用adam
              loss='mse', #他内置了softmax操作，该OP计算输入input和标签label间的交叉熵损失 ，它结合了 LogSoftmax 和 NLLLoss 的OP计算，可用于训练一个 n 类分类器。
              #原因在于CrossEntropyLoss函数实际上内置了LogSoftmax 和 NLLLoss，也就是你一旦使用CrossEntropyLoss，在计算预测值和标签值时就会自动帮你将预测值给LogSoftmax 和 NLLLoss
              metrics=['acc'])#loss='categorical_crossentropy'
model.summary()



# 定义回调函数 Callbacks 中的earlystop用于训练过程
log_file_path = base_path + '_emotion_training - big_XCEPTION2+mse64@.log'#log of training process,change when training
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4),
                              verbose=1)
# 模型位置及命名
trained_models_path = base_path + '_big_XCEPTION_2+mse64'#'_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-acc{acc:.2f}.hdf5'#'.{epoch:02d}-{val_acc:.2f}.hdf5'

# 定义模型权重位置、命名等
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                    save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]



# 载入数据集
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# 划分训练、测试集
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)#train_test_split

# 图片产生器，在批量中对数据进行增强，扩充数据集大小
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,   #旋转
                        width_shift_range=0.1,  #左右平移
                        height_shift_range=0.1,  #上下平移
                        zoom_range=.1,  #缩放
                        horizontal_flip=True)  #水平翻转

# 利用数据增强进行训练
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),#model.fit
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
