#!/usr/bin/python
# -*- coding UTF-8 -*-

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import argparse,os
import numpy as np
from utils_count import get_train_test_list,args_summary,get_train_log_label,get_test_log_label
from Log_Kmeans import TRAIN_Kmeans,PREDICT_Kmeans
from CNN_LSTM_Model import cnn_lstm as lstm

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    KTF.set_session(session)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a_log',type=str,default='../data/Hadoop_Logs/PageRank')
    parser.add_argument('--model_b_log',type=str,default='../data/Hadoop_Logs/WordCount')
        
    parser.add_argument('--model_a_label',type=str,default='../data/Hadoop_labels/PageRank')
    parser.add_argument('--model_b_label',type=str,default='../data/Hadoop_labels/WordCount')
    
    parser.add_argument('--model_a_template_vec',type=str,default='../data/Hadoop_Logs/template_vec.dat')
    parser.add_argument('--model_b_template_vec',type=str,default='../data/Hadoop_Logs/template_vec.dat')
    parser.add_argument('--class_vec_save',type=str,default='../data/Hadoop_Logs/class_vec')
    parser.add_argument('--class_vec_load',type=str,default='../data/Hadoop_Logs/class_vec')
#     -------------------------------------------------------------
    parser.add_argument('--knn_model_save',type=str,default='../model/kmeans_model.pkl')
    parser.add_argument('--knn_model_load',type=str,default='../model/kmeans_model.pkl')
#     -------------------------------
    parser.add_argument('--train_count',type=int,default=1)
    parser.add_argument('--classes_num',type=int,default=170)
#     -------------------------
    parser.add_argument('--step_size',help='single_step.',type=int,default=3)
    parser.add_argument('--window_size',help='window_size.',type=int,default=10)
    
    parser.add_argument('--epoch',help='epoch',type=int,default=10)
    parser.add_argument('--batch_size',help='batch_size.',type=int,default=800)
    parser.add_argument('--lstm_hidden_dims',help='lstm_hidden_dims.',type=int,default=128)
    parser.add_argument('--dense_hidden_dims',help='dense_hidden_dims.',type=int,default=50)
    parser.add_argument('--vector_size',help='vector_size.',type=int,default=150)
    parser.add_argument('--lstm_drop_out',help='lstm_drop_out',type=float,default=0.5)
    parser.add_argument('--dense_drop_out',help='dense_drop_out',type=float,default=0.4)
#     ----------------------------------------
    parser.add_argument('--model_dir',help='model_dir.',type=str,default='../model/log')
    parser.add_argument('--predict_result',help='predict_result.',type=str,default='../result')  
    parser.add_argument('--result_save',help='result_save',type=str,default='../0_result/B6220_D2020')
    parser.add_argument('--test_dir',help='test_dir.',type=str,default='../')
    args = parser.parse_args()
    args_summary(args)
    train_a_list,train_b_list,test_list = get_train_test_list(args.model_a_log,args.model_a_label,args.model_b_log,args.model_b_label)
#     train
    cluster_train = TRAIN_Kmeans(
        train_a_list,
        train_b_list,
        args.train_count,
        args.classes_num,
        args.model_a_template_vec,
        args.model_b_template_vec,
        args.knn_model_save,
        args.class_vec_save
    )
    cluster_train.train_kmeans()
    get_cluster = PREDICT_Kmeans(
        test_list,
        args.classes_num,
        args.model_b_template_vec,
        args.knn_model_load,
        args.class_vec_load)
    get_cluster.predict_kmeans()
    train_x,train_y,train_a_l,train_b_l = get_train_log_label(train_a_list,train_b_list,args.train_count,args.window_size,args.step_size,args.class_vec_save)
    test_x,test_y,test_l = get_test_log_label(test_list,args.train_count,args.window_size,args.step_size,args.class_vec_load)
    with open (args.result_save,'a') as a:
        a.write('train_a_l:' + train_a_l + '\n')
        a.write('train_b_l:' + train_b_l + '\n')
        a.write('test_l:' + test_l + '\n')
    train_model = lstm(
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
        args.result_save,
        train_x,
        train_y        
    ).train_model() 
#     test_x = list(test_x)
#     test_y = list(test_y)
    length = int(len(test_x)/100)
    for time in range(100):
        save_x = os.path.join(args.test_dir,str(time)+'test_x')
        save_y = os.path.join(args.test_dir,str(time)+'test_y')
        if time == 99:
            np.save(save_x,test_x[time*length:])
            np.save(save_y,test_y[time*length:])
        else:
            np.save(save_x,test_x[time*length:(time+1)*length])
            np.save(save_y,test_y[time*length:(time+1)*length])
                
                    
                
            
#     predict_lstm = lstm(
#         args.model_dir,
#         args.epoch,
#         args.batch_size,
#         args.lstm_hidden_dims,
#         args.dense_hidden_dims,
#         args.vector_size,
#         args.window_size,
#         args.lstm_drop_out,
#         args.dense_drop_out,
#         args.predict_result,
#         args.result_save,
#         test_x,
#         test_y        
#     )
#     predict_lstm.predict_model()
#     predict_lstm.performance_model()
    
