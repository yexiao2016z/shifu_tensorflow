import tensorflow as tf
import sys
import json
import numpy as np
import pandas as pd
#from hdfs.ext.kerberos import KerberosClient
import os
#import io
#import gzip
#import distkeras

LEARNING_DECAY = 0.0
ADAMBETA1 = 0.9
ADAMBETA2 = 0.999
MOMENTUM = 0.0

model_file_path = sys.argv[1]
loss = sys.argv[2]
optimizer = sys.argv[3]
data_path = sys.argv[4]
select_status = sys.argv[5]
epochs = int(sys.argv[6])
valid_rate = float(sys.argv[7])
learning_rate = float(sys.argv[8])
bagging_num = int(sys.argv[9])
if sys.argv[10] in ['false', 'False']:
	bagging_replace = False
else:
	bagging_replace = True
bagging_sample_rate = float(sys.argv[11])

print("Your NormalizedData Path is " + data_path + '\n')

if optimizer == "sgd":
	optimizer = tf.keras.optimizers.SGD(lr=learning_rate,momentum=MOMENTUM,decay=LEARNING_DECAY)
elif optimizer == "adam":
	optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=ADAMBETA1, beta_2=ADAMBETA2, decay=LEARNING_DECAY)
elif optimizer == "adagrad":
	optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=LEARNING_DECAY)
elif optimizer == "rmsprop":
	optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, decay=LEARNING_DECAY)
elif optimizer == "nadam":
	optimizer = tf.keras.optimizers.Nadam(lr=learning_rate, beta_1=ADAMBETA1, beta_2=ADAMBETA2)

if loss == "squared":
	loss = "mean_squared_error"
elif loss == "absolute":
	loss = "mean_absolute_error"
elif loss == "log":
	loss = "mean_squared_logarithmic_error"
elif loss == "hinge":
	loss = "hinge"
	
model = tf.keras.models.model_from_json(json.dumps(json.load(open(model_file_path))))
model.compile(optimizer=optimizer,loss=loss)

data = pd.DataFrame()
#client_hdfs = KerberosClient('http://' + os.environ['IP_HDFS'] + ':50070')
#index = 1
for file in os.listdir(data_path):
	#if file.endswith('.gz'):
	#	with client_hdfs.read(data_path+'/'+file) as reader:
	#		reader=gzip.GzipFile(fileobj=io.BytesIO(reader.read()))
	tmp_data = pd.read_csv(data_path+'/'+file, sep='|', header=None, dtype=np.float32)
	data = pd.concat([tmp_data,data])
	print("Loading Data File " + file)
	#index += 1

x=list()
y=list()
select_status = select_status.split(',')			
for index in range(len(select_status)):
	if select_status[index] == 'T':
		x.append(index)
	elif select_status[index] == 'N':
		y.append(index)

index = 1
data_raw = data
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

for i in range(bagging_num):
	print("Start Bagging Turn " + str(index))
	data = data_raw.sample(frac=bagging_sample_rate,replace=bagging_replace)
	x_data = data[x].values
	y_data = data[y].values.flatten()
	sample_weight = data[[len(select_status)]].values.flatten()
	
	print("Start Model Fitting in Bagging Turn " + str(index))
	model.fit(x=x_data,y=y_data,epochs=epochs,validation_split=valid_rate)
	print("End Model Fitting in Bagging Turn " + str(index))
	sess = tf.keras.backend.get_session()
	builder = tf.saved_model.builder.SavedModelBuilder('./modelx')
	builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
	#frozen_graph = freeze_session(tf.keras.backend.get_session(),
    #                          output_names=[out.op.name for out in model.outputs])
	#tf.train.write_graph(frozen_graph,".",
	#					"./models/model_save"+str(i)+".pb",as_text=False)
	builder.save()
	os.popen('mv ./modelx/* ./models')
	os.popen('rm -r ./modelx')
	print("Save Model File Successfully in Bagging Turn" + str(index))
	index += 1
exit(0)