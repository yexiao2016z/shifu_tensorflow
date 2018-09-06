package ml.shifu.shifu.core.alg;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.AbstractTrainer;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.dnn.Common.*;
import ml.shifu.shifu.dnn.Initializer.Constant;
import ml.shifu.shifu.dnn.Initializer.Initializer;
import ml.shifu.shifu.dnn.Initializer.RandomNormal;
import ml.shifu.shifu.dnn.Initializer.RandomUniform;
import ml.shifu.shifu.dnn.Layer.Dense;
import ml.shifu.shifu.dnn.Layer.Dropout;
import ml.shifu.shifu.dnn.Layer.Layer;

/**
 * Neural network trainer
 */
public class DNNTrainerTF extends AbstractTrainer {

     private static final Logger LOG = LoggerFactory.getLogger(DNNTrainerTF.class);
    
    /**
     * Convergence judger instance for convergence criteria checking.
     */
    private String select_status;
    
    public DNNTrainerTF(ModelConfig modelConfig, int trainerID, Boolean dryRun, String select_status) {
        super(modelConfig, trainerID, dryRun);
        this.select_status = select_status;
    }

    private Layer last_layer;
    private int lastLayerNum;
    
    private List<String> buildNetworkStep() throws Exception {
    	List<String> sb = new ArrayList<String>();
    	//addCommon();
    	sb.add("global_step = tf.Variable(0, name='global_step', trainable=False)");
    	sb.addAll(addInputLayer(null));
    	List<String> layer_names = (List<String>) modelConfig.getParams().get("LayerName");
    	List<String> activation = (List<String>) modelConfig.getParams().get("ActivationFunc");
    	List<Integer> hidden_layer_nodes = (List<Integer>) modelConfig.getParams().get("NumHiddenNodes");
    	List<String> kernel_initializers = (List<String>) modelConfig.getParams().get("KernelInitializers");
    	List<String> bias_initializers = (List<String>) modelConfig.getParams().get("BiasInitializers");
    	List<Double> dropout_rate = (List<Double>)modelConfig.getParams().get("DropoutRate");
    	String loss = (String)modelConfig.getParams().get("Loss");
    	String optimizer = (String)modelConfig.getParams().get("Optimizer");
    	double learningRate = (Double)modelConfig.getParams().get("LearningRate");
    	if(layer_names.size() != 0 && (layer_names.size() != activation.size() || layer_names.size() != hidden_layer_nodes.size()
    			|| layer_names.size() != kernel_initializers.size() || layer_names.size() != bias_initializers.size())) {
            throw new RuntimeException(
                    "the number of layer do not equal to some of the parameters");
        }
    	Layer layer;
    	Initializer kernel_ini;
    	Initializer bias_ini;
    	for(int i = 0; i < layer_names.size(); i++) {
    		String layer_name = layer_names.get(i);
    		if(layer_name.equalsIgnoreCase("dense")) {
    			if(bias_initializers.get(i).equalsIgnoreCase(InitializerCatagory.Constant.name())) {
    				bias_ini = new Constant();
    			}else if(bias_initializers.get(i).equalsIgnoreCase(InitializerCatagory.RandomUniform.name())) {
    				bias_ini = new RandomUniform();
    			}else if(bias_initializers.get(i).equalsIgnoreCase(InitializerCatagory.RandomNormal.name())){
    				bias_ini = new RandomNormal();
    			}else {
    				bias_ini = null;
    			}
    			if(kernel_initializers.get(i).equalsIgnoreCase(InitializerCatagory.Constant.name())) {
    				kernel_ini = new Constant();
    			}else if(kernel_initializers.get(i).equalsIgnoreCase(InitializerCatagory.RandomUniform.name())) {
    				kernel_ini = new RandomUniform();
    			}else if(kernel_initializers.get(i).equalsIgnoreCase(InitializerCatagory.RandomNormal.name())){
    				kernel_ini = new RandomNormal();
    			}else {
    				kernel_ini = null;
    			}
    			if(bias_ini != null && kernel_ini != null)
    				layer = new Dense(hidden_layer_nodes.get(i), ActivationCatagory.valueOf(activation.get(i)),
    						kernel_ini,bias_ini);
    			else if(bias_ini != null && kernel_ini == null) {
    				layer = new Dense(hidden_layer_nodes.get(i), ActivationCatagory.valueOf(activation.get(i)),
    						new RandomNormal());
    			}else if(bias_ini == null && kernel_ini != null) {
    				layer = new Dense(hidden_layer_nodes.get(i), ActivationCatagory.valueOf(activation.get(i)),
    						kernel_ini);
    			}else {
    				layer = new Dense(hidden_layer_nodes.get(i), ActivationCatagory.valueOf(activation.get(i)));
    			}
    						
    			
    		}else if(layer_name.equalsIgnoreCase("dropout")) {
    			layer = new Dropout(dropout_rate.get(i));
    		}else {
    			throw new RuntimeException("Not Support Layer Catagory");
    		}
    		sb.addAll(addHiddenLayer(layer));
    		
    	}
    	sb.addAll(addLoss(loss));
    	sb.addAll(addOptimizer(optimizer, learningRate));
    	//addOutputLayer();
		
		return sb;
    }
    //hinge没做
    private List<String> addLoss(String loss) {
    	List<String> sb = new ArrayList<String>();
    	String y_ = this.last_layer.getName() + "_outac";
    	if (loss.equalsIgnoreCase("squared")) {
    		sb.add(String.format("loss = tf.losses.mean_squared_error(y, %s)", y_));
    	}else if(loss.equalsIgnoreCase("absolute")) {
    		sb.add(String.format("loss = tf.losses.absolute_difference(y, %s)", y_));
    	}else if(loss.equalsIgnoreCase("hinge")) {
    		sb.add(String.format("loss = tf.losses.hinge_loss(y, %s)", y_));
    	}else if(loss.equalsIgnoreCase("log")) {
    		sb.add(String.format("loss = tf.losses.log_loss(y, %f)", y_));
    	}
    	return sb;
    }
    private List<String> addOptimizer(String optimizer,double rate){
    	List<String> sb = new ArrayList<String>();
    	if (optimizer.equalsIgnoreCase("sgd")) {
    		sb.add(String.format("train_step = tf.train.GradientDescentOptimizer(%f).minimize(loss,global_step=global_step)", rate));
    	}else if(optimizer.equalsIgnoreCase("adam")) {
    		sb.add(String.format("train_step = tf.train.AdamOptimizer(%f).minimize(loss,global_step=global_step)", rate));
    	}else if(optimizer.equalsIgnoreCase("adamgrad")) {
    		sb.add(String.format("train_step = tf.train.AdagradOptimizer(%f).minimize(loss,global_step=global_step)", rate));
    	}
    	return sb;
    }
    public List<String> addHiddenLayer(Layer layer) throws Exception {
    	List<String> sb = new ArrayList<String>();
    	String layerName = layer.getName();
    	if(layer.getLayerCatagory().equals(LayerCatagory.Dense)){
    		ActivationCatagory ac = ((Dense)layer).getActivationCatagory();
    		int units = ((Dense)layer).getUnits();
    		Initializer k = ((Dense)layer).getKernelInitializer();
    		Initializer b = ((Dense)layer).getBiasInitializer();
    		sb.addAll(addInitializer(null, layerName, units, k, "w"));
    		sb.addAll(addInitializer(null,layerName, units, b, "b"));
    		//params.put("units", ((Dense)layer).getUnits());
		if (this.last_layer != null)
    			sb.add(String.format("%s_out =  tf.nn.xw_plus_b(%s_outac, %s_w ,%s_b)\n", layerName, this.last_layer.getName(),layerName,  layerName));
    		else
			
    			sb.add(String.format("%s_out = tf.nn.xw_plus_b(x, %s_w, %s_b)", layerName, layerName,  layerName));
		if (ac.equals(ActivationCatagory.sigmoid)) {
    			sb.add(String.format("%s_outac = tf.nn.sigmoid(%s_out,name='%s_outac')", layerName, layerName,layerName));
    		}else if(ac.equals(ActivationCatagory.softmax)) {
    			sb.add(String.format("%s_outac = tf.nn.softmax(%s_out,name='%s_outac')", layerName, layerName,layerName));
    		}else if(ac.equals(ActivationCatagory.tanh)) {
    			sb.add(String.format("%s_outac = tf.nn.tanh(%s_out,name='%s_outac')", layerName, layerName,layerName));
    		}else if(ac.equals(ActivationCatagory.relu)) {
    			sb.add(String.format("%s_outac = tf.nn.relu(%s_out,name='%s_outac')", layerName, layerName,layerName));
    		}else if(ac.equals(ActivationCatagory.linear)) {
    			sb.add(String.format("%s_outac = %s_out\n", layerName, layerName));
    		}
    		this.last_layer = layer;
    		this.lastLayerNum = units;
    	}else if(layer.getLayerCatagory().equals(LayerCatagory.Dropout)) {
    		double rate = ((Dropout)layer).getRate();
    		sb.add(String.format("%s_outac = tf.nn.dropout(%s_outac, %f, name='%s_outac')", layerName, this.last_layer.getName(), rate,layerName));
    		this.last_layer = layer;
    	}
    	return sb;
    }
    private List<String> addInitializer(String deviceName, String layerName, int layerNum ,Initializer ini, String wORb){
    	List<String> sb = new ArrayList<String>();
    	String varName = layerName + '_' + wORb;
    	
    	if(ini.getInitializerCatagory().equals(InitializerCatagory.Constant)) {
    		double val = ((Constant)ini).getConstant();
    		if (wORb == "w")
    			sb.add(String.format("%s = tf.Variable(tf.constant(%f,shape=[%d, %d]), name='%s')", varName, val, 
    					this.lastLayerNum, layerNum,  varName));
    		else
    			sb.add(String.format("%s = tf.Variable(tf.constant(%f, shape=[%d]), name='%s')", varName, val, layerNum,  
    					 varName));
    		
    	}else if(ini.getInitializerCatagory().equals(InitializerCatagory.RandomNormal)) {
    		double mean = ((RandomNormal)ini).getMean();
    		double stddev = ((RandomNormal)ini).getStddev();
    		if (wORb == "w")
    			sb.add(String.format("%s = tf.Variable(tf.truncated_normal([%d, %d], mean=%f, stddev=%f), name='%s')", varName, 
    					this.lastLayerNum, layerNum, mean, stddev,  varName));
    		else
    			sb.add(String.format("%s = tf.Variable(tf.truncated_normal([%d], mean=%f, stddev=%f), name='%s')", varName, layerNum,
    					mean, stddev, varName));
    	}else if(ini.getInitializerCatagory().equals(InitializerCatagory.RandomUniform)) {
    		double minVal = ((RandomUniform)ini).getMinVal();
    		double maxVal = ((RandomUniform)ini).getMaxVal();
    		if (wORb == "w")
    			sb.add(String.format("%s = tf.Variable(tf.random_uniform([%d, %d], minval=%f, maxval=%f), name='%s')", varName, this.lastLayerNum, 
    					layerNum,minVal, maxVal, varName));
    		else
    			sb.add(String.format("%s = tf.Variable(tf.random_uniform([%d], minval=%f, maxval=%f), name='%s')", varName, layerNum, minVal, 
    					maxVal, varName));
    	}
    	if (deviceName == null) {
    		;
    	}else {
    		sb.set(0, "\t"+sb.get(0));
    		sb.add(0, "with tf.device(" + deviceName + "):");
    	}
    	return sb;
    }
    private List<String> addInputLayer(String deviceName){
    	List<String> sb = new ArrayList<String>();
    	int columnNumsX = 0;
    	int columnNumsY = 0;
    	for(Character c:select_status.toCharArray()) {
    		if(c.equals('T')) {
    			columnNumsX++;
    		}else if(c.equals('N')) {
    			columnNumsY++;
    		}
    	}
    	String x = String.format("x = tf.placeholder(tf.float32, [None, %d],name='x')", columnNumsX);
    	String y = String.format("y = tf.placeholder(tf.float32, [None, %d],name='y')", columnNumsY);
    	
    	//StringBuilder sb = new StringBuilder();
    	if (deviceName == null) {
    		sb.add(x);
    		sb.add(y);
    	}else {
    		sb.add("with tf.device(" + deviceName + "):");
	    	sb.add("\t" + x);
	    	sb.add("\t" + y); 
    	}
    	//this.lastLayerNum = new Integer(columnNumsX).toString();
    	this.lastLayerNum = columnNumsX;
     	return sb;
    }
    private List<String> addTrainStep(){
    	List<String> sb = new ArrayList<String>();
    	sb.add("init_op = tf.global_variables_initializer()");
    	//sb.add("train_dir = tempfile.mkdtemp()\n");
    	sb.add("#sv = tf.train.Server(is_chief=is_chief, logdir='./tmp/train_logs', init_op=init_op, recovery_wait_secs=1, global_step=global_step)");
    	sb.add("if is_chief:");
			sb.add("\tprint('Worker %d: Initailizing session...' % FLAGS.task_index)");
		sb.add("else:"); 
			sb.add("\tprint('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)"); 
		sb.add("with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),checkpoint_dir='./tmp/train_logs'" + 
				") as sess:");
		//sb.add(while True:
			sb.add("\tprint('Worker %d: Session initialization  complete.' % FLAGS.task_index)"); 
			sb.add("\ttime_begin = time.time()");
			//sb.add("\tprint('Traing begins @ %f' % time_begin)");
			sb.add("\tlocal_step = 0"); 
			sb.add("\twhile True:"); 
				sb.add("\t\ttime_begin = time.time()");
		//sb.add("\tbatch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)\n"); 
		//sb.add("\ttrain_feed = {x: batch_xs, y_: batch_ys}\n");  
				sb.add("\t\t_, step = sess.run([train_step, global_step], feed_dict={x:features, y:labels})"); 
		//	sb.add("\tlocal_step += 1");
				sb.add("\t\tnow = time.time()"); 
				sb.add("\t\tprint('%f: Worker %d: dome (step:%d)' % (now, FLAGS.task_index, local_step+1))"); 
		//	sb.add("\tif step >= FLAGS.train_steps:");
		//		sb.add("\t\tbreak");
				sb.add("\t\ttime_end = time.time()"); 
				//sb.add("\t\tprint('Training ends @ %f' % time_end)"); 
				sb.add("\t\ttrain_time = time_end - time_begin");
				sb.add("\t\tprint('Training elapsed time:%f s' % train_time)");
				sb.add("\t\tlocal_step += 1");
				sb.add("\t\tprint(os.popen('hadoop fs -touchz ' + hdfs_home +  '/w' + str(local_step) + 'k' + str(FLAGS.task_index)).read())");
				sb.add("\t\tstart = time.time()");
				sb.add("\t\twhile os.popen('hadoop fs -ls ' + hdfs_home + ' | grep w' + str(local_step) + 'k | wc -l').read().strip() != str(len(worker_spec)):");
				
				sb.add("\t\t\tprint('wait other worker on step ' + str(local_step) + '....')");
				sb.add("\t\t\ttime.sleep(5)");
				sb.add("\t\t\tif time.time()-start > 60000*30:");
				sb.add("\t\t\t\traise Exception('Time out')");
				sb.add("\t\ttrain_loss = sess.run([loss],feed_dict={x:features,y:labels})[0]");
				sb.add("\t\tvali_loss = sess.run([loss],feed_dict={x:features_vali,y:labels_vali})[0]");
				sb.add("\t\tprint(os.popen('echo \"' + str(train_loss)+';'+str(vali_loss) +'\" | hadoop fs -put - ' + hdfs_home + '/result'     + str(local_step) + '_' + str(FLAGS.task_index)).read())");
				sb.add("\t\twhile os.popen('hadoop fs -ls ' + hdfs_home + ' | grep result' + str(local_step) + '_ | wc -l').read().strip() != str(len(worker_spec)):");
					sb.add("\t\t\tprint('wait other worker on  Err step ' + str(local_step) + '....')");
					sb.add("\t\t\ttime.sleep(5)");
					sb.add("\t\t\tif time.time()-start > 60000*30:");
						sb.add("\t\t\t\traise Exception('Time out')");

				sb.add("\t\tprint(os.popen('hadoop fs -rm ' + hdfs_home + '/w' + str(local_step-1) + 'k' + str(FLAGS.task_index)).read())");
				sb.add("\t\tif local_step == epoch_num:");
					sb.add("\t\t\tprint('self work done')");
			        sb.add("\t\t\tprint(os.popen('hadoop fs -touchz ' + hdfs_home + '/wk' + str(FLAGS.task_index)).read())");
			        sb.add("\t\t\tprint('aware self done')");
			        sb.add("\t\t\tbreak");
			sb.add("\tstart = time.time()");
			sb.add("\twhile os.popen('hadoop fs -ls ' + hdfs_home + ' | grep wk | wc -l').read().strip() != str(len(worker_spec)):");
				
				sb.add("\t\tprint('wait other worker....')");
				sb.add("\t\ttime.sleep(5)");
				sb.add("\t\tif time.time()-start > 60000*30:");
				sb.add("\t\t\traise Exception('Time out')");
			sb.add("\tif is_chief:");
				sb.add("\t\tbuilder = tf.saved_model.builder.SavedModelBuilder('./models')");
				sb.add("\t\tsess.graph._unsafe_unfinalize()");
				sb.add("\t\tbuilder.add_meta_graph_and_variables(sess._sess._sess._sess._sess,[tf.saved_model.tag_constants.SERVING])");
				sb.add("\t\tsess.graph.finalize()");
				sb.add("\t\tbuilder.save()");
				sb.add("\t\tprint(os.popen('hadoop fs -put ./tmp/models/ ' + hdfs_home).read())");
			 	sb.add("\t\tprint('load to hdfs successfully')");
		 sb.add("print('All worker done')");
		// sb.add("if is_chief:");
		// 	sb.add("\tbuilder = tf.saved_model.builder.SavedModelBuilder('./models')");
		// 	sb.add("\twith open('./tmp/train_logs/graph.pbtxt') as f:");
		// 		sb.add("\t\ttxt = f.read()");
		// 	sb.add("\tgdef = text_format.Parse(txt, tf.GraphDef())");
	//	 	sb.add("\ttf.train.write_graph(gdef, './tmp/train_logs', 'graph.pb', as_text=False)");

		 	

		//sb.add(String.format("\tif step > %d", (Integer)modelConfig.getParams().get("numTrainEpochs")));
    	return sb;
    }
    
    private List<String> addClusterStep(){
    	//num_worker = len(worker_spec)
    	List<String> sb = new ArrayList<String>();
    	
    	
    	sb.add("cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})");
    	sb.add("server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)");
    	sb.add("if FLAGS.job_name == 'ps': server.join()");

        sb.add("is_chief = (FLAGS.task_index == 0)");
    
        sb.add("with tf.device(tf.train.replica_device_setter(cluster=cluster)):");
        return sb;
    }
    private List<String> addParametersStep(){
    	List<String> sb = new ArrayList<String>();
    	sb.add("import tensorflow as tf");
    	sb.add("import time");
    	sb.add("import pandas as pd");
    	sb.add("import os");
    	//sb.add("print(os.listdir('.'))");
    	sb.add("import numpy as np");
    	sb.add("import gzip");
		sb.add("import io");
    	sb.add("flags = tf.app.flags");	
    	sb.add("flags.DEFINE_string('ps_hosts', None, 'Comma-separated list of hostname:port pairs')");
    	sb.add("flags.DEFINE_string('worker_hosts', None,'Comma-separated list of hostname:port pairs')");
    	sb.add("flags.DEFINE_string('job_name', None, 'job name: worker or ps')");
    	sb.add("flags.DEFINE_integer('task_index', None, 'Index of task within the job')");
    	sb.add("flags.DEFINE_string('name_node', None, 'NameNode')");
    	sb.add("flags.DEFINE_integer('node_nums', None, 'NameNode')");
    	sb.add("flags.DEFINE_integer('is_chief', None, 'NameNode')");

    	sb.add(String.format("flags.DEFINE_integer('train_steps', %d, 'Number of training steps to perform')"
    			, modelConfig.getNumTrainEpochs()));
    	//FileSystem fs = new FileSystem(new Configuration());
    	try {
			sb.add(String.format("flags.DEFINE_string('data_dir', '%s', 'Directory  for storing mnist data')",
					pathFinder.getNormalizedDataPath().substring(FileSystem.get(new Configuration()).getHomeDirectory().toString().length())));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	sb.add("flags.DEFINE_string('ip_address_dir', '/shifu_tmp/', 'syn the ip_address of every worker')");
    	sb.add("FLAGS = flags.FLAGS");
    	sb.add(String.format("sample_rate = %f", modelConfig.getValidSetRate()));
    	sb.add(String.format("epoch_num = %d", modelConfig.getNumTrainEpochs()));
    	try {
			sb.add(String.format("hdfs_home = '%s'", FileSystem.get(new Configuration()).getHomeDirectory().toString()+"/shifu_tmp/"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	sb.add("os.mkdir('./tmp/train_logs/')");
    	sb.add("def main(unused_argv):");
    	return sb;
    }
    private List<String> readDataStep(){
    	List<String> sb = new ArrayList<String>();
    	List<Integer> featureCol = new ArrayList<Integer>();
    	List<Integer> targetCol = new ArrayList<Integer>();
    	for(int i = 0; i < select_status.split(",").length; i++) {
    		if(select_status.toCharArray()[2*i] == 'T') {
    			featureCol.add(i);
    		}else if(select_status.toCharArray()[2*i] == 'N') {
    			targetCol.add(i);
    		}
    	}
    	sb.add("counter = 0");
    	sb.add("while not os.path.exists('ipz') and counter < 3:");
    	sb.add("\ttime.sleep(60)");
    	sb.add("\tcounter += 1");
    	sb.add("if counter == 3: exit(-1)");
    	sb.add("import re");
    	sb.add("ip = re.findall('10\\.\\d+?\\.\\d+?\\.\\d+',os.popen('ifconfig').read().strip())[0]");
    	sb.add("file_names = list()");
    	sb.add("with open('ipz', 'r') as f:");
    		sb.add("\tfile_names = f.readline().strip().split(',')");
    	sb.add("ps_spec = list([file_names[0]])");
    	sb.add("worker_spec = file_names");
    	sb.add("worker_spec[0] = ':'.join([worker_spec[0].split(':')[0],str(int(worker_spec[0].split(':')[1])+1)])");
    	sb.add("print(ps_spec)");
    	sb.add("print(worker_spec)");
    	sb.add("if ip == ps_spec[0].split(':')[0]:");
    		sb.add("\twith open('./syn','a+') as f:");
    			sb.add("\t\tf.write(str(os.getpid())+';')");
    		sb.add("\twhile True:");
    		    sb.add("\t\twith open('./syn','r') as f:");
    		    	sb.add("\t\t\tk = f.readline().strip().split(';')");
    		                                  //print(k)
    		        sb.add("\t\t\tif len(k) == 3:break");
    		 
    		sb.add("\tif os.getpid() == int(open('./syn','r').readline().strip().split(';')[0]):");
    		    sb.add("\t\tFLAGS.job_name = 'ps'");
    		    sb.add("\t\tFLAGS.task_index = 0");

    		sb.add("\telse:");
    			sb.add("\t\tFLAGS.job_name == 'worker'");
    			sb.add("\t\tFLAGS.task_index = 0");
    	sb.add("else:");
    		sb.add("\tFLAGS.job_name = 'worker'");
    		sb.add("\tFLAGS.task_index = list([i.split(':')[0] for i in worker_spec]).index(ip)");
    		sb.add("\tif is_cheif == 1:exit(0)");
    	sb.add("count_file = len([1 for file in os.listdir('.') if file.endswith('.gz')])");

    	sb.add("count_ip = len(worker_spec)");
    	
    	sb.add("data = pd.DataFrame()");
    	sb.add("if FLAGS.job_name == 'worker':");
    		sb.add("\tfor file in os.listdir('.'):");
    			sb.add("\t\tif file.endswith('.gz'):");
    				sb.add("\t\t\ttmp_data = pd.read_csv(file, sep='|',compression='gzip', header=None, dtype=np.float32)");
    				sb.add("\t\t\tdata = pd.concat([tmp_data,data])");
    		sb.add("\tif len(data) == 0:exit(-1)");
    		sb.add(String.format("\tfeatures = data[%s].values*data[[len(data.columns)-1]].values", featureCol.toString()));
    		sb.add(String.format("\tlabels = data[%s].values", targetCol.toString()));
    		sb.add("\tt = int(len(features)*sample_rate)");
    		sb.add("\tfeatures_vali = features[:t]");
    		sb.add("\tlabels_vali = labels[:t]");
    		sb.add("\tfeatures = features[t:]");
    		sb.add("\tlabels = labels[t:]");

    		
    	return sb;
    }
    private List<String> addRunStep(){
    	List<String> sb = new ArrayList<String>();
    
    	sb.add("if __name__ == '__main__':");
    		sb.add("\timport re");
    	    sb.add("\tip = re.findall('10\\.\\d+?\\.\\d+?\\.\\d+',os.popen('ifconfig').read().strip())[0]");
    	    sb.add("\tfile_names = list()");
    	    sb.add("\twith open('ipz', 'r') as f:");
            	sb.add("\t\tfile_names = f.readline().strip().split(',')");
    	    sb.add("\tps_spec = list([file_names[0]])");
    	    sb.add("\tif ip == ps_spec[0].split(':')[0]:");
    	    	sb.add("\t\timport multiprocessing as mul");
    	        sb.add("\t\tp1 = mul.Process(target=tf.app.run)");
    	        sb.add("\t\tp2 = mul.Process(target=tf.app.run)");
    	        sb.add("\t\tp1.start()");
    	        sb.add("\t\tp2.start()");
    	    sb.add("\telse:");
    	        sb.add("\t\ttf.app.run()");

    	return sb;
    }
    
    private List<String> makeScript() throws Exception{
    	List<String> temp;
    	List<String> sb = new ArrayList<String>();
    	sb.addAll(addParametersStep());
    	temp = readDataStep();
    	for(String s:temp)
    		sb.add("\t" + s);
    	temp = addClusterStep();
    	for(String s : temp) {
    		sb.add("\t" + s);
    	}
    	
    	temp = buildNetworkStep();
    	for(String s : temp) {
    		sb.add("\t\t" + s);
    	}
    	temp = addTrainStep();
    	for(String s : temp) {
    		sb.add("\t\t" + s);
    	}
    	
    	sb.addAll(addRunStep());
    	return sb;
    }
    @Override
    public double train() throws IOException {
    	try {
    		File models = new File("models");
            FileUtils.forceMkdir(models);
    		List<String> tmp = this.makeScript();
    		Path file = Paths.get("./models/script.py");
    		Files.write(file, tmp, Charset.forName("UTF-8"));
    		LOG.info("Successfully generate tensorflow script in models/script.py!");
    	}catch(Exception e) {
    		throw new IOException(e.getMessage().toString());
    	}
        return 0;
    }

    //public DNNNetwork getNetwork() {
    //    return network;
    //}

    

    /**
     * @param network
     *            the network to set
     */
    //public void setNetwork(DNNNetwork network) {
        //this.network = network;
    //}

    

    

    
}

   
    
