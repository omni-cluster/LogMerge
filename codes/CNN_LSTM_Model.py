import numpy as np
from keras.utils import np_utils
from keras.layers import Input,LSTM,Dropout,Dense,Flatten,Conv1D,MaxPooling1D
from keras.models import Sequential,Model,load_model
from keras.callbacks import ModelCheckpoint
import time
from keras import backend as K
from keras.models import load_model,Sequential
import os
from sklearn import metrics
import matplotlib.pyplot as plt

class cnn_lstm:
    def __init__(self,model_dir,epoch,batch_size,lstm_hidden_dims,dense_hidden_dims,vector_size,window_size,lstm_drop_out,dense_drop_out,predict_result,result_save,x,y=[]):
        self.x = x
        self.y = y
        self.model_dir = model_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.lstm_hidden_dims = lstm_hidden_dims
        self.dense_hidden_dims = dense_hidden_dims
        self.vector_size = vector_size
        self.window_size = window_size
        self.lstm_drop_out = lstm_drop_out
        self.dense_drop_out = dense_drop_out
        self.predict_result = predict_result
        self.result_save = result_save

    def train_model(self):
        train_start =time.time()
        isExists=os.path.exists(self.model_dir)
        if not isExists:
            os.makedirs(self.model_dir,)
        x = self.x
        y = np_utils.to_categorical(self.y,num_classes=2)
        print('the shape of train x is {0}'.format(x.shape))
        print('the shape of train y is {0}'.format(y.shape))
#         model_1.add(Flatten())
        model  = Sequential()
        model.add(Conv1D(self.lstm_hidden_dims,1,strides=1,padding='same',activation='tanh',input_shape=(self.window_size,self.vector_size)))
#         model.add(MaxPooling1D(1))
        model.add(Conv1D(self.lstm_hidden_dims,1,strides=1,padding='same',activation='tanh'))
#         model.add(MaxPooling1D(1))
        model.add(Conv1D(self.lstm_hidden_dims,1,strides=1,padding='same',activation='tanh'))
#         model.add(MaxPooling1D(1))
        model.add(LSTM(self.lstm_hidden_dims,return_sequences = True))
#         model.add(LSTM(self.lstm_hidden_dims,return_sequences=True,input_shape=(self.window_size,self.vector_size)))
        model.add(Dropout(self.lstm_drop_out))
        model.add(LSTM(self.lstm_hidden_dims))
        model.add(Dropout(self.dense_drop_out))
        model.add(Dense(self.dense_hidden_dims,activation='relu'))
        model.add(Dense(2, activation = 'softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#         filepath = os.path.join(self.model_dir,'log_weights-rm-{epoch:02d}-{loss:.4f}-bigger.h5')
#         checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#         callbacks_list = [checkpoint]
#         model.fit(x,y,epochs=self.epoch,batch_size=self.batch_size,verbose=0,callbacks=callbacks_list)
        model.fit(x,y,epochs=self.epoch,batch_size=self.batch_size,verbose=0)
        model_path = os.path.join(self.model_dir,'model.h5')
        model.save(model_path)
        train_end = time.time()
        print('training time:',(train_end-train_start)/60,'mins')
        K.clear_session()

    def predict_model(self):
        x = self.x
        print('the shape of predict x is {0}'.format(x.shape))
        model_file = os.path.join(self.model_dir,'model.h5')
        model = load_model(model_file)
        for layer in model.layers:
            layer.trainable = False
        predict_score = []
        model.summary()
        print('========doing predicting......========')
        with open(os.path.join(self.predict_result,'predict_result.txt'),'w') as w:
            for i in x:
                i = np.reshape(i, (1, i.shape[0], i.shape[1]))
                score = model.predict(i, verbose = 0)
                w.write(str(score[0][1])+'\n')
    def predict_model_div(self,result_dir,time):
        x = self.x
        print('the shape of predict x is {0}'.format(x.shape))
        model_file = os.path.join(self.model_dir,'model.h5')
        model = load_model(model_file)
        for layer in model.layers:
            layer.trainable = False
        predict_score = []
        model.summary()
        print('========doing predicting......========')
        save_file = os.path.join(result_dir,time)
        scores = []
        for i in x:
            i = np.reshape(i, (1, i.shape[0], i.shape[1]))
            score = model.predict(i, verbose = 0)
            scores.append(score[0][1])
        scores = np.array(scores)
#         print(save_file)
        np.save(save_file,scores)
   
    def performance_model(self):
        scores = []
        index = []
        precision = []
        recall = []
        f1 = []
        with open(os.path.join(self.predict_result,'predict_result.txt'),'r') as r:
            for l in r:
                scores.append(float(l.split()[0]))
        for i,d in enumerate(np.arange(0.0,1,0.0002)):
            d = round(d,4)
            index.append(d)
            predict = []
            for score in scores:
                if score > d:
                    predict.append(1)
                else:
                    predict.append(0)
            precision.append(metrics.precision_score(self.y,predict))
            recall.append(metrics.recall_score(self.y,predict))
            f1.append(metrics.f1_score(self.y,predict))
        with open(os.path.join(self.predict_result,'predict_score.txt'),'w') as w:
            w.write(str(precision)+'\n')
            w.write(str(recall)+'\n')
            w.write(str(f1)+'\n')
        index = np.array(index)
        precision = np.array(precision)
        recall = np.array(recall)
        f1 = np.array(f1)
        best_index = np.argmax(f1)
        with open(self.result_save,'a') as a:
            a.write('Precision score is '+str(precision[best_index])+'\n')
            a.write('Recall score is '+str(recall[best_index])+'\n')
            a.write('best F1 score is '+str(np.max(f1))+'(threshold is'+str(index[best_index])+')\n')
        print('performancedisplay down!')

