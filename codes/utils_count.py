#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np


def get_log_label_list(log_dir,label_dir):
    log_label = []
    log_dir_add = os.path.join(log_dir,'DIV_ID')
    log_id = os.listdir(log_dir_add)
    log_id = sorted(log_id)
    for i in log_id:
        log_id_dir = os.path.join(log_dir_add,i)
        if os.path.isdir(log_id_dir): 
            label_file = os.path.join(label_dir,i + '_label')
            log_label_tem = []
            log_label_tem.append(log_id_dir)
            log_label_tem.append(label_file)
            log_label.append(log_label_tem)
    return log_label                  
    
def get_train_test_list(model_a_log,model_a_label,model_b_log,model_b_label):
    model_a = get_log_label_list(model_a_log,model_a_label)
    model_b = get_log_label_list(model_b_log,model_b_label)
    train_a = model_a 
    train_b = model_b
    test = model_b
    return train_a,train_b,test
def get_all_log_label(log_label,window_size,step_size,vec_file):
#     [[log,label]...[]]
    x = []
    y = []
    class_vec = {}
    with open(vec_file,'r') as r:
        for l in r:
            l = l.split('//')
            class_vec[l[0]] = list(map(float,l[1][1:-2].split(', ')))
    for i in log_label:
        log_file = os.path.join(i[0],'seq_class')
        label_file = i[1]
        with open(log_file,'r') as log,open(label_file,'r') as label:
            logs = log.readlines()
            labels = label.readlines()
        for j in range(0,len(logs)-window_size,step_size):
            tem_x_seq = logs[j:j + window_size]
            tem_x = []
            y.append(int(labels[j + window_size].split()[0]))
            for t in tem_x_seq:
                t = t.split()[1]
                tem_x.append(class_vec[t])            
            x.append(tem_x)
#     index = np.arange(len(x))
#     np.random.shuffle(index)
#     x = np.array(x)
#     y = np.array(y)
#     x = x[index]
#     y = y[index]    
    return x,y
def get_top_log_label(log_label,train_count,window_size,step_size,vec_file):
    x = []
    y = []
    class_vec = {}
    with open(vec_file,'r') as r:
        for l in r:
            l = l.split('//')
            class_vec[l[0]] = list(map(float,l[1][1:-2].split(', ')))
    for i in log_label:
        with open('./tem.txt','a') as a:
            a.write(i[0]+'\n\n')
        log_file = os.path.join(i[0],'seq_class')
        label_file = i[1]
        with open(log_file,'r') as log,open(label_file,'r') as label:
            logs = log.readlines()
            labels = label.readlines()

        for j in range(0,len(logs)-window_size,step_size):
            tem_x_seq = logs[j:j + window_size]
            tem_x = []
            y.append(int(labels[j + window_size].split()[0]))
            for t in tem_x_seq:
                t = t.split()[1]
                tem_x.append(class_vec[t])            
            x.append(tem_x)
    count_error = 0
    flag = 0
    for i,y_tem in enumerate(y):
        if y_tem == 1:
            count_error += 1
        if count_error >= train_count:
            flag = 1
            break
    if flag == 1:
        x = x[:i+1]
        y = y[:i+1]
    return x,y
def get_last_log_label(log_label,train_count,window_size,step_size,vec_file):
    x = []
    y = []
    class_vec = {}
    with open(vec_file,'r') as r:
        for l in r:
            l = l.split('//')
            class_vec[l[0]] = list(map(float,l[1][1:-2].split(', ')))
    for i in log_label:
        with open('./tem.txt','a') as a:
            a.write(i[0]+'\n\n')
        log_file = os.path.join(i[0],'seq_class')
        label_file = i[1]
        with open(log_file,'r') as log,open(label_file,'r') as label:
            logs = log.readlines()
            labels = label.readlines()

        for j in range(0,len(logs)-window_size,step_size):
            tem_x_seq = logs[j:j + window_size]
            tem_x = []
            y.append(int(labels[j + window_size].split()[0]))
            for t in tem_x_seq:
                t = t.split()[1]
                tem_x.append(class_vec[t])            
            x.append(tem_x)
    count_error = 0
    flag = 0
    for i,y_tem in enumerate(y):
        if y_tem == 1:
            count_error += 1
        if count_error >= train_count:
            flag = 1
            break
    if flag == 1:
        x = x[i+1:]
        y = y[i+1:]
    else:
        x = []
        y = []
    return x,y
    
def get_train_log_label(train_a_list,train_b_list,train_count,window_size,step_size,vec_file):
    a_x,a_y = get_all_log_label(train_a_list,window_size,step_size,vec_file)
    if train_count == 0:
        x = a_x
        y = a_y
        b_x = []
        b_y = []
    else:
        b_x,b_y = get_top_log_label(train_b_list,train_count,window_size,step_size,vec_file)
        x = a_x + b_x
        y = a_y + b_y
    x = np.array(x)
    y = np.array(y)
    return x,y,str(len(a_x)),str(len(b_x))
def get_test_log_label(test_b_list,train_count,window_size,step_size,vec_file):  
    if train_count == 0:
        x,y = get_all_log_label(test_b_list,window_size,step_size,vec_file)
    else:
        x,y = get_last_log_label(test_b_list,train_count,window_size,step_size,vec_file)
    length = str(len(x))
    x = np.array(x)
    y = np.array(y)
    return x,y,length
def args_summary(args):
    with open (args.result_save,'a') as a:
#         a.write('model_a_log:' + args.model_a_log + '\n')
#         a.write('model_b_log:' + args.model_b_log + '\n')
#         a.write('model_a_label:' + args.model_a_label + '\n')
#         a.write('model_b_label:' + args.model_b_label + '\n')
#         a.write('model_a_template_vec:' + args.model_a_template_vec + '\n')
#         a.write('model_b_template_vec:' + args.model_b_template_vec + '\n')
#         a.write('knn_model_save:' + args.knn_model_save + '\n')
#         a.write('knn_model_load:' + args.knn_model_load + '\n')
#         a.write('class_vec_save:' + args.class_vec_save + '\n')
#         a.write('class_vec_load:' + args.class_vec_load + '\n')
        a.write('train_count:' + str(args.train_count) + '\n')
#         a.write('classes_num:' + str(args.classes_num) + '\n')
#         a.write('step_size:' + str(args.step_size) + '\n')
#         a.write('window_size:' + str(args.window_size) + '\n')
#         a.write('epoch:' + str(args.epoch) + '\n')
#         a.write('batch_size:' + str(args.batch_size) + '\n')
#         a.write('lstm_hidden_dims:' + str(args.lstm_hidden_dims) + '\n')
#         a.write('dense_hidden_dims:' + str(args.dense_hidden_dims) + '\n')
#         a.write('vector_size:' + str(args.vector_size) + '\n')
#         a.write('lstm_drop_out:' + str(args.lstm_drop_out) + '\n')
#         a.write('dense_drop_out:' + str(args.dense_drop_out) + '\n')
#         a.write('model_dir:' + args.model_dir + '\n')
#         a.write('predict_result' + args.predict_result + '\n')
       