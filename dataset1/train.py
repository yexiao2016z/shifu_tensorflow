import tensorflow as tf
import sys
import json
import numpy as np
import pandas as pd
#from hdfs.ext.kerberos import KerberosClient
import os
import io
import gzip

LEARNING_DECAY = 0.5
ADAMBETA1 = 0.9
ADAMBETA2 = 0.999
MOMENTUM = 0.5

model_file_path = sys.argv[1]
loss = sys.argv[2]
optimizer = sys.argv[3]
data_path = sys.argv[4]
select_status = sys.argv[5]
epochs = int(sys.argv[6])
valid_rate = float(sys.argv[7])
learning_rate = float(sys.argv[8])
job_name = str(sys.argv[9])
task_index = int(sys.argv[10])

data=pd.read_csv(data_path,sep='|',header=None)
x=list()
y=list()
for index in range(len(select_status)):
	if select_status[index] == 'T':
		x.append(index)
	elif select_status[index] == 'N':
		y.append(index)
x_data = data[[1,2,3]].values
y_data = data[[0]].values
#index = data_path.split('/').index('ModelSets')
#data_path = '/'.join(data_path.split('/')[index:])

print("\nYour NormalizedData Path is " + data_path + '\n')
#bagging_num = int(sys.argv[9])
#bagging_replace = bool(sys.argv[10])
#bagging_sample_rate = float(sys.argv[11])
if optimizer == "sgd":
	optimizer = tf.keras.optimizers.SGD(lr=learning_rate,momentum=MOMENTUM,decay=LEARNING_DECAY)
elif optimizer == "adam":
	optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=ADAMBETA1, beta_2=ADAMBETA2, decay=LEARNING_DECAY)
elif optimizer == "adagrad":
	optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=LEARNING_DECAY)
	
if loss == "squared":
	loss = "mean_squared_error"

elif loss == "absolute":
	loss = "mean_absolute_error"
elif loss == "log":
	loss = "categorical_crossentropy"

cluster = tf.train.ClusterSpec({'ps': ["master:2322"], 'worker': ["worker1:2322"]})
server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
if job_name == 'ps':
	server.join()
with tf.device(tf.train.replica_device_setter(
            cluster=cluster)):
	global_step = tf.Variable(0, name='global_step', trainable=False)



	
	model = tf.keras.models.model_from_json(json.dumps(json.load(open(model_file_path))))
	model_graph = tf.keras.backend.get_session().graph.as_graph_def()
	tf.import_graph_def(model_graph)
	y = model.output
	x_ = model.input
	y_ = tf.placeholder(tf.float32, [None, 1])
	loss = -tf.reduce_sum(y_ * y)
	opt = tf.train.AdamOptimizer(0.1)

	train_step = opt.minimize(loss,global_step=global_step)
	init_op = tf.global_variables_initializer()
	sv = tf.train.Supervisor(is_chief=True, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)
	sess = sv.prepare_or_wait_for_session(server.target)
	_, step = sess.run([train_step, global_step], feed_dict=train_feed)
	print(" Worker %d: traing step dome (global step:%d)" % (0, step))
#select_status = select_status.split(',')

#data = pd.DataFrame()
#client_hdfs = KerberosClient('http://' + os.environ['IP_HDFS'] + ':50070')
#for file in client_hdfs.list(data_path):
#	if file.endswith('.gz'):
#		with client_hdfs.read(data_path+'/'+file) as reader:
#			reader=gzip.GzipFile(fileobj=io.BytesIO(reader.read()))
#			tmp_data = pd.read_csv(reader,sep='|',header=None,dtype=np.float32)
#			data = pd.concat([tmp_data,data])
#			print(data.info())

sample_weight = data[[len(select_status)]].values.flatten()		
#data_raw = data
for i in range(bagging_num):
	data = data_raw.sample(frac=bagging_sample_rate,replace=bagging_replace)
	x_data = data[x].values
	y_data = data[y].values.flatten()
	sample_weight = data[[len(select_status)]].values.flatten()
	
	model.fit(x=x_data,y=y_data,epochs=epochs,validation_split=valid_rate,sample_weight=sample_weight)
	tf.train.write_graph(tf.keras.backend.get_session().graph.as_graph_def(),".",
						"./models/model_save"+str(i)+".pb",as_text=False)
