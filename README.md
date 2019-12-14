LogMerge——面向多语法日志的通用异常检测机制研究
====
论文如下：
题目：面向多语法日志的通用异常检测机制研究

源码细节如下.

Requirement:
======
	Python: 3.6  
	Tenssorflow: 1.9.0 
	keras:2.2.4
	numpy:1.17.2

代码结构:
======
    ├── model                   # 聚类模型权重
    ├── codes                   # 代码
    ├── data                    # 数据集（以Hadoop数据集为例，其他数据整理成相应格式即可）
    ├── result                  # 结果  
    ├── README.md               # README文件
    

输入数据样例：
====
`
2015-10-17 15:37:56,547 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster for application appattempt_1445062781478_0011_000001
2015-10-17 15:37:56,899 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Executing with tokens:
2015-10-17 15:37:56,900 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Kind: YARN_AM_RM_TOKEN, Service: , Ident: (appAttemptId { application_id { id: 11 cluster_timestamp: 1445062781478 } attemptId: 1 } keyId: 471522253)
2015-10-17 15:37:57,036 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Using mapred newApiCommitter.
2015-10-17 15:37:57,634 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: OutputCommitter set in config null
2015-10-17 15:37:57,720 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: OutputCommitter is org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter
`

结果样例：
====

`
p:0.5069444444444444
r:0.6224806201550388
f1:0.558803061934586
`


简单运行
====
1. git clone https://github.com/Logs2019/LogMerge.git
2. cd codes
3. `bash train.sh` or `bash predict.sh`

