#import h5py
import keras
import numpy as np
import random
from scipy.fftpack import fft
from sklearn import preprocessing
from keras.layers import add,LSTM,Input,Reshape,Dense,Dropout,Conv1D,LeakyReLU,MaxPooling1D,GlobalMaxPooling1D,Concatenate,GlobalAveragePooling1D,BatchNormalization
from keras.layers.convolutional import ZeroPadding1D


# LSTM_CNN_para
def get_model(n_class_temp):
    n_classes = n_class_temp
    inp=Input(shape=(3600,),name = 'inp')
    resh_1 = Reshape((3600,1))(inp)
    resh_2 = Reshape((60,60))(inp)
    
    conv7=Conv1D(128, 3)(resh_1)
    conv7_1=LeakyReLU(alpha=0.33)(conv7) 
    m8=MaxPooling1D(3)(conv7_1)
 #   m4 = Concatenate(axis = 1)([m4,conv3_1])
    conv9=Conv1D(64, 3)(m8)
    conv9=LeakyReLU(alpha=0.33)(conv9)
    g2=GlobalAveragePooling1D()(conv9)
    lg2=Dense(512)(g2)
    lg2=LeakyReLU(alpha=0.33)(lg2)
    
    lstm1=LSTM(input_shape=(60,60),output_dim=128, return_sequences=True)(resh_2)
    lstm2 = LSTM(64, return_sequences=False)(lstm1)
    
    g=Concatenate(axis=1)([lstm2,lg2])
    
    l1 = Dense(64)(g)
    l2=LeakyReLU(alpha=0.33)(l1)
    l3 = Dense(32)(l2)
    l4=LeakyReLU(alpha=0.33)(l3)
    l11=Dense(n_classes, activation='softmax')(l4)
    model=keras.Model(input=inp,outputs=l11)
    model.summary()
    #编译model
    adam = keras.optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0,amsgrad=False)
   # adam = keras.optimizers.Adam(lr = 0.0005, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    #adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    #sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
    #reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def NormMinandMax(npdarr, min_temp, max_temp):
    """"
    将数据npdarr 归一化到[min,max]区间的方法
    返回 副本
    """
    Ymax = np.max(npdarr)  # 计算最大值
    Ymin = np.min(npdarr)  # 计算最小值
    k = (max_temp - min_temp) / (Ymax - Ymin)
    last = min_temp + k * (npdarr - Ymin)
    return last 

def main():
    print('main_2() is running!')
    batch_size = 128
    num_classes = 2
    epochs = 100

    datapath = 'D:\\Spectrum_sense\\Sectrum Sense paper\\sample\\sample\\sample\\data FSK\\spectrum_data_train.mat'

    Spectrum_train_data =  h5py.File(datapath,'r')
    Spectrum_train_X = Spectrum_train_data['spectrum_data_trainX'][:]
    Spectrum_train_Y = Spectrum_train_data['spectrum_data_trainY'][:]
    Spectrum_test_X = Spectrum_train_data['spectrum_data_testX'][:]
    Spectrum_test_Y = Spectrum_train_data['spectrum_data_testY'][:]

    Spectrum_train_X = np.transpose(Spectrum_train_X)
    Spectrum_test_X = np.transpose(Spectrum_test_X)
    #Spectrum_train_Y = np.transpose(Spectrum_train_Y)
    #Spectrum_test_Y = np.transpose(Spectrum_test_Y)


    Spectrum_train_data.close()
    for i in range(0,len(Spectrum_train_X)):
       Spectrum_train_X[i] = NormMinandMax(Spectrum_train_X[i],0,1)
    for i in range(0,len(Spectrum_test_X)):
       Spectrum_test_X[i] = NormMinandMax(Spectrum_test_X[i],0,1)
##打乱顺序
    index_train = [i for i in range(len(Spectrum_train_X))]
    index_test = [i for i in range(len(Spectrum_test_X))]

    random.shuffle(index_train)
    random.shuffle(index_test)

    Spectrum_train_X = Spectrum_train_X[index_train]
    Spectrum_test_X = Spectrum_test_X[index_test]

    Spectrum_train_Y = Spectrum_train_Y[index_train]
    Spectrum_test_Y = Spectrum_test_Y[index_test]
##加入卷积层的数据处理方式
#    Spectrum_train_X = Spectrum_train_X.astype('float32')
    
#    Spectrum_test_X = Spectrum_test_X.astype('float32')

# convert class vectors to binary class matrices
    Spectrum_train_X = Spectrum_train_X.astype('float32')
    Spectrum_test_X = Spectrum_test_X.astype('float32')
    Spectrum_train_Y = keras.utils.to_categorical(Spectrum_train_Y, num_classes)
    Spectrum_test_Y = keras.utils.to_categorical(Spectrum_test_Y, num_classes)
    
    moudle_LSTM_CNN_para = get_model(num_classes)
    
    moudle_LSTM_CNN_para.fit(Spectrum_train_X, Spectrum_train_Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
         validation_split=0.2
         )

    scores = moudle_LSTM_CNN_para.evaluate(Spectrum_test_X, Spectrum_test_Y, verbose=0)
    print('RNN test score:', scores[0])
    print('RNN test accuracy:', scores[1])
    moudle_LSTM_CNN_para.save('D:\\Spectrum_sense\\Sectrum Sense paper\\sample\sample\\model\\LSTM_CNN_FSK1_model.h5')
if __name__ == '__main__':
    main()
