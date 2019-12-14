#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt

class My_Kmeans(object):
    def __init__(self):
        object.__init__(self)
        
    def find_all_template(self):
        for i in self.log_list:
            files = os.listdir(i[0])
            for file in files:
                if file.endswith('Template_order'):
                    template_file = os.path.join(i[0],file)
                    with open(template_file,'r') as r:
                        for l in r:
                            self.template_set.add(l[:-1].lower())
                            
    def match_template_vec(self):
        tem_list = list()
        with open(self.template_vec,'r') as r:
            for l in r :
                l = l.split('//')
                l[1] = list(map(float,l[1][1:-2].split()))
                if l[0] in self.template_set:
                    self.template_list.append(l[0])
                    self.vec_list.append(l[1])

class TRAIN_Kmeans(My_Kmeans):
    def __init__(self,train_a_list,train_b_list,train_count,classes_num,model_a_template_vec,model_b_template_vec,knn_model_save,class_vec_save):
        self.log_a_list = train_a_list
        self.log_b_list = train_b_list
        self.train_count = train_count
        self.classes_num = classes_num
        self.model_a_template_vec = model_a_template_vec
        self.model_b_template_vec = model_b_template_vec
        self.knn_model_save = knn_model_save
        self.class_vec_save = class_vec_save
        self.model_a_template_set = set()
        self.model_b_template_set = set()
        self.template_list = list()
        self.vec_list = list()
        self.class_list = list() 
    def find_a_template(self):
        for i in self.log_a_list:
            files = os.listdir(i[0])
            for file in files:
                if file.endswith('Template_order'):
                    template_file = os.path.join(i[0],file)
                    with open(template_file,'r') as r:
                        for l in r:
                            self.model_a_template_set.add(l[:-1].lower())
    def find_b_template(self):
        all_count = 0
        stop = 0
        for i in self.log_b_list:
            if stop == 1:
                return
            seq_set = set()
            files = os.listdir(i[0])
            label_file = i[1]
            for file in files:
                y = []
                if file.endswith('seq_with_time'):
                    seq_with_time_file = os.path.join(i[0],file)
                    with open(seq_with_time_file,'r') as seq,open(label_file,'r') as label:
                        content = seq.readlines()
                        label_error = []
                        for tem_label in label:
                            label_error.append(tem_label.split()[0])
                        for j in range(0,len(label_error)-10,3):
#                             print(len(label_error))
#                             print(j)
                            y.append([label_error[j + 10]])
                            
                        flag = 0
                        for index,y_tem in enumerate(y):
                            if y_tem == 1:
                                all_count += 1
                        if all_count >= self.train_count:
                            flag = 1
                            break
                        if flag == 1:
                            logs = content[0:index+2]
                        else:
                            logs = content
                        for line in logs:
                            seq_set.add(int(line.split()[1])-1)
            for file in files:
                if file.endswith('Template_order'):
                    template_file = os.path.join(i[0],file)
                    with open(template_file,'r') as r:
                        templates = r.readlines()
                        for index,l in enumerate(templates):
                            if index in seq_set:
                                self.model_b_template_set.add(l[:-1].lower())
    def find_all_template(self):
        self.find_a_template()
        self.find_b_template()
    def match_a_template_vec(self):
        tem_list = list()
        with open(self.model_a_template_vec,'r') as r:
            for l in r :
                l = l.split('//')
                l[1] = list(map(float,l[1][1:-2].split()))
                if l[0] in self.model_a_template_set:
                    self.template_list.append(l[0])
                    self.vec_list.append(l[1])
    def match_b_template_vec(self):
        tem_list = list()
        with open(self.model_b_template_vec,'r') as r:
            for l in r :
                l = l.split('//')
                l[1] = list(map(float,l[1][1:-2].split()))
                if l[0] in self.model_b_template_set:
                    self.template_list.append(l[0])
                    self.vec_list.append(l[1])
    def match_template_vec(self):
        self.match_a_template_vec()
        self.match_b_template_vec()
    def save_template_class(self):
        seq_class = {}
        for i in self.log_a_list:
            files = os.listdir(i[0])
            for file in files:
                if file.endswith('Template_order'):
                    template_file = os.path.join(i[0],file)
                    with open(template_file,'r') as r,open(os.path.join(i[0],'template_to_class'),'w') as w:
                        contents = r.readlines()
                        for index,l in enumerate(contents): 
                            seq_class[index+1] = str(self.class_list[self.template_list.index(l[:-1].lower())])
                            w.write(l[:-1] + '//' + str(index + 1) + '//' + seq_class[index+1] + '\n') 
            for file in files:
                if file.endswith('seq_with_time'):    
                    seq_file = os.path.join(i[0],file)
                    with open(seq_file,'r') as r_seq,open(os.path.join(i[0],'seq_class'),'w') as w:
                        contents = r_seq.readlines()
                        for l in contents: 
                            time = l.split()[0]
                            l = int(l.split()[1])
                            if l!= -1:
                                w.write(time + ' ' + seq_class[l] + '\n')
                            else:
                                w.write(time + ' ' + str(self.classes_num) + '\n')
    def train_kmeans(self):
        self.find_all_template()
        self.match_template_vec()
        km = KMeans(n_clusters=self.classes_num)#para
        self.class_list = km.fit_predict(self.vec_list)
        joblib.dump(km,self.knn_model_save)
        self.save_template_class()
        centroids = km.cluster_centers_
        with open(self.class_vec_save,'w') as w:
            for index,c in enumerate(centroids):
                w.write(str(index) + '//' + str(c.tolist()) + '\n')
            w.write(str(self.classes_num) + '//' + str([0.0 for t in range(150)]) + '\n')
class PREDICT_Kmeans(My_Kmeans):
    def __init__(self,test_list,classes_num,template_vec,knn_model_load,class_vec):
        self.log_list = test_list
        self.classes_num = classes_num
        self.template_vec = template_vec
        self.knn_model_load = knn_model_load
        self.class_vec = class_vec
        self.template_set = set()
        self.template_list = list()
        self.vec_list = list()
        self.class_list = list()
        
    def save_template_class(self):
        seq_class = {}
        for i in self.log_list:
            files = os.listdir(i[0])
            for file in files:
                if file.endswith('Template_order'):
                    template_file = os.path.join(i[0],file)
                    with open(template_file,'r') as r,open(os.path.join(i[0],'template_to_class'),'w') as w:
                        contents = r.readlines()
                        for index,l in enumerate(contents): 
                            seq_class[index+1] = str(self.class_list[self.template_list.index(l[:-1].lower())])
                            w.write(l[:-1] + '//' + str(index + 1) + '//' + seq_class[index+1]+'\n') 
            for file in files:
                if file.endswith('seq_with_time'):    
                    seq_file = os.path.join(i[0],file)
                    with open(seq_file,'r') as r_seq,open(os.path.join(i[0],'seq_class'),'w') as w:
                        contents = r_seq.readlines()
                        for l in contents: 
                            time = l.split()[0]
                            l = int(l.split()[1])
                            if l!= -1:
                                w.write(time + ' ' + seq_class[l] + '\n')    
                            else:
                                w.write(time + ' ' + str(self.classes_num) + '\n')
                                
    def predict_kmeans(self):
        self.find_all_template()
        self.match_template_vec()
        km = joblib.load(self.knn_model_load)
        self.class_list = km.predict(self.vec_list)
        self.save_template_class()

                    