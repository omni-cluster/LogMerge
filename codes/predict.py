#!/usr/bin/python
# -*- coding UTF-8 -*-

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import argparse,os
from utils_count import get_train_test_list,args_summary,get_train_log_label,get_test_log_label
from Log_Kmeans import TRAIN_Kmeans,PREDICT_Kmeans
from CNN_LSTM_Model import cnn_lstm as lstm
import numpy as np
from operator import itemgetter
def performance(score_root,y_root,prc,final_result_save):
    files = os.listdir(score_root)
    tp = [0 for i in range(500)]
    tn = [0 for i in range(500)]
    fp = [0 for i in range(500)]
    fn = [0 for i in range(500)]
    p = [0 for i in range(500)]
    r = [0 for i in range(500)]
    f1 = [0 for i in range(500)]
    for f in files:
        score_file = os.path.join(score_root,f)
        if os.path.isdir(score_file):
            continue
        scores = list(np.load(score_file))
        label_file = os.path.join(y_root,f[:-4]+'test_y.npy')
        labels = list(np.load(label_file))
        for i,d in enumerate(np.arange(0.0,1,0.002)):
            d = round(d,4)
            predict = []
            for score in scores:
                if score > d:
                    predict.append(1)
                else:
                    predict.append(0)
            for p_,l_ in zip(predict,labels):
                if p_ == 1 and l_ == 1:
                    tp[i] += 1
                elif p_ == 0 and l_ == 0:
                    tn[i] += 1
                elif p_ == 0 and l_ == 1:
                    fn[i] += 1
                elif p_ == 1 and l_ == 0:
                    fp[i] += 1
    all_r_p = set()
    for i in range(500):
        if tp[i]+fp[i]!= 0 and tp[i]+fn[i] != 0:
            p[i] = tp[i]/(tp[i]+fp[i])
            r[i] = tp[i]/(tp[i]+fn[i])
            f1[i] = 2*p[i]*r[i]/(p[i]+r[i])
        all_r_p.add(str(r[i])+' '+str(p[i]))
    result = []
    for r_p in all_r_p:
        result.append([r_p.split()[0],r_p.split()[1]])
    result = sorted(result,key=itemgetter(0,1))
    with open (prc,'a') as w:
        for i_r in result:
            w.write(str(i_r)+'\n')
        w.write('#'*30+'\n')
    f1 = np.array(f1)
    with open(final_result_save,'a') as w:
        index = np.argmax(f1)
        w.write('p:'+str(p[index])+'\n')
        w.write('r:'+str(r[index])+'\n')
        w.write('f1:'+str(f1[index])+'\n')
    
if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    KTF.set_session(session)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',help='model_dir.',type=str,default='../model/log')
    parser.add_argument('--window_size',help='window_size.',type=int,default=10)
    parser.add_argument('--test_x_dir',help='window_size.',type=str)
    parser.add_argument('--test_y_dir',help='model_dir.',type=str)
    parser.add_argument('--epoch',help='epoch',type=int,default=10)
    parser.add_argument('--batch_size',help='batch_size.',type=int,default=800)
    parser.add_argument('--lstm_hidden_dims',help='lstm_hidden_dims.',type=int,default=128)
    parser.add_argument('--dense_hidden_dims',help='dense_hidden_dims.',type=int,default=50)
    parser.add_argument('--vector_size',help='vector_size.',type=int,default=150)
    parser.add_argument('--lstm_drop_out',help='lstm_drop_out',type=float,default=0.5)
    parser.add_argument('--dense_drop_out',help='dense_drop_out',type=float,default=0.4)
    parser.add_argument('--predict_result',help='predict_result.',type=str,default='../result')  
    parser.add_argument('--final_result_save',help='result_save',type=str,default='../0_result/B6220_D2020')
    parser.add_argument('--div_result',help='result_save',type=str,default='../0_result/')
    parser.add_argument('--prc_result',help='result_save',type=str,default='../0_result/')
    parser.add_argument('--mode',help='',type=int,default=0)# 0 predict 1 performance
    args = parser.parse_args()

    if args.mode == 0:
        time = args.test_x_dir.split('/')[-1].split('test')[0]
        test_x = np.load(args.test_x_dir)
        test_y = np.load(args.test_y_dir)
    
        predict_lstm = lstm(
            args.model_dir,
            args.epoch,
            args.batch_size,
            args.lstm_hidden_dims,
            args.dense_hidden_dims,
            args.vector_size,
            args.window_size,
            args.lstm_drop_out,
            args.dense_drop_out,
            args.predict_result,
            args.final_result_save,
            test_x,
            test_y        
        )
        predict_lstm.predict_model_div(args.div_result,time)
    else:
        performance(args.div_result,args.test_y_dir,args.prc_result,args.final_result_save)
    