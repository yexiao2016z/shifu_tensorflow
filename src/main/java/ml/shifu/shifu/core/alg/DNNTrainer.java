package ml.shifu.shifu.core.alg;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.shifu.container.ModelInitInputObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.ConvergeJudger;
import ml.shifu.shifu.core.MSEWorker;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.keras.KerasExecutor;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.io.FileUtils;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.IntRange;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.concurrency.DetermineWorkload;
import org.encog.util.concurrency.EngineConcurrency;
import org.encog.util.concurrency.TaskGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;

import ml.shifu.shifu.dnn.Common;
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
public class DNNTrainer extends AbstractTrainer {

     private static final Logger LOG = LoggerFactory.getLogger(DNNTrainer.class);
    //private final static double Epsilon = 1.0; // set the weight range in [-INIT_EPSILON INIT_EPSILON];

    //public static final Map<String, Double> defaultLearningRate;
    //public static final Map<String, String> learningAlgMap;

    //private DNNNetwork network;
    //private volatile boolean toPersistentModel = true;
    //private volatile boolean toLoggingProcess = true;

    /**
     * Convergence judger instance for convergence criteria checking.
     */
    private String select_status;
    
    public DNNTrainer(ModelConfig modelConfig, int trainerID, Boolean dryRun, String select_status) {
        super(modelConfig, trainerID, dryRun);
        this.select_status = select_status;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> network;
    private Map<String, Object> config;
    private List<Map<String, Object>> layers;
    private Layer last_layer;
    
    public void buildNetwork() throws Exception {
    	config = new HashMap<String, Object>();
    	layers = new ArrayList<Map<String, Object>>();
    	network = new HashMap<String, Object>();
    	network.put("config", config);
    	config.put("layers", layers);
    	addCommon();
    	addInputLayer();
    	//@SuppressWarnings("unchecked")
    	
    	List<String> layer_names = (List<String>) modelConfig.getParams().get("LayerName");
    	List<String> activation = (List<String>) modelConfig.getParams().get("ActivationFunc");
    	List<Integer> hidden_layer_nodes = (List<Integer>) modelConfig.getParams().get("NumHiddenNodes");
    	List<String> kernel_initializers = (List<String>) modelConfig.getParams().get("KernelInitializers");
    	List<String> bias_initializers = (List<String>) modelConfig.getParams().get("BiasInitializers");
    	List<Double> dropout_rate = (List<Double>)modelConfig.getParams().get("DropoutRate");
    	if(layer_names.size() != 0 && (layer_names.size() != activation.size() || layer_names.size() != hidden_layer_nodes.size()
    			|| layer_names.size() != kernel_initializers.size() || layer_names.size() != bias_initializers.size() || 
    			layer_names.size() != dropout_rate.size())) {
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
    		addHiddenLayer(layer);
    	}
    	addOutputLayer();
    }
    private void addCommon() {
    	this.network.put("class_name", "Model");
    	this.network.put("keras_version", Common.KERAS_VERSION);
    	this.network.put("backend", Common.BACKEND);
    	this.config.put("name", Common.DEFAULT_MODEL_NAME);
    }
    public void addHiddenLayer(Layer layer) throws Exception {
    	Map<String, Object> layer_config = new HashMap<String, Object>();
    	Map<String, Object> params = new HashMap<String, Object>();
    	layer_config.put("name", layer.getName());
    	//LOG.info(layer.toString());
    	//LOG.info((LayerHasAc));
    	layer_config.put("class_name", layer.getLayerCatagory().name());
    	params.put("name", layer.getName());
		params.put("trainable", Common.DEFAULT_TRAINABLE);
		params.put("dtype", Common.DEFAULT_DTYPE);
    	if(layer.getLayerCatagory().equals(LayerCatagory.Dense)){
    		params.put("units", ((Dense)layer).getUnits());
    		params.put("activation", ((Dense)layer).getActivationCatagory().name().toLowerCase());
    		params.put("use_bias", ((Dense)layer).isUseBias());
    		params.put("kernel_initializer", addInitializer(((Dense)layer).getKernelInitializer()));
    		params.put("bias_initializer", addInitializer(((Dense)layer).getBiasInitializer()));
    		//params.put("bias_initializer", addInitializer((Dense)layer).getBiasInitializer()));
    		params.put("kernel_regularizer", null);
    		params.put("bias_regularizer", null);
    		params.put("activity_regularizer", null);
    		params.put("kernel_constraint", null);
    		params.put("bias_constraint", null);
    	}else if(layer.getLayerCatagory().equals(LayerCatagory.Dropout)) {
    		params.put("rate", ((Dropout)layer).getRate());
    		params.put("noise_shape", null);
    		params.put("seed", null);
    	}
    	layer_config.put("config", params);
    	List<Object> tmp1 = new ArrayList<Object>();
    	List<Object> tmp2 = new ArrayList<Object>();
    	List<Object> tmp3 = new ArrayList<Object>();
    	tmp1.add(tmp2);
    	tmp2.add(tmp3);
    	if(this.layers.isEmpty())
    		tmp3.add("input_1");
    	else
    		tmp3.add(this.layers.get(layers.size()-1).get("name"));
    	tmp3.add(0);
    	tmp3.add(0);
    	tmp3.add(new HashMap<String, Object>());
    	layer_config.put("inbound_nodes", tmp1);
    	this.last_layer = layer;
    	this.layers.add(layer_config);
    }
    private Map<String, Object> addInitializer(Initializer ini){
    	Map<String, Object> initializer_config = new HashMap<String, Object>();
    	Map<String, Object> params = new HashMap<String, Object>();
    	if(ini.getInitializerCatagory().equals(InitializerCatagory.Constant)) {
    		initializer_config.put("class_name", "Constant");
    		params.put("value", ((Constant) ini).getConstant());
    		params.put("dtype", Common.DEFAULT_DTYPE);
    	}else if(ini.getInitializerCatagory().equals(InitializerCatagory.RandomNormal)) {
    		initializer_config.put("class_name", "RandomNormal");
    		params.put("mean", ((RandomNormal)ini).getMean());
    		params.put("stddev", ((RandomNormal)ini).getStddev());
    		params.put("seed", null);
    		params.put("dtype", Common.DEFAULT_DTYPE);
    	}else if(ini.getInitializerCatagory().equals(InitializerCatagory.RandomUniform)) {
    		initializer_config.put("class_name", "RandomUniform");
    		params.put("minval", ((RandomUniform)ini).getMinVal());
    		params.put("maxval", ((RandomUniform)ini).getMaxVal());
    		params.put("seed", null);
    		params.put("dtype", Common.DEFAULT_DTYPE);
    	}
    	initializer_config.put("config", params);
    	return initializer_config;
    }
    private void addInputLayer(){
    	int columnNums = 0;
    	Map<String, Object> layer_config = new HashMap<String, Object>();
    	Map<String, Object> params = new HashMap<String, Object>();
    	layer_config.put("name", "input_1");
    	layer_config.put("class_name", "InputLayer");
    	List<Object> tmp = new ArrayList<Object>();
    	tmp.add(null);
    	for(Character c:select_status.toCharArray()) {
    		if(c.equals('T')) {
    			columnNums++;
    		}
    	}
    	tmp.add(columnNums);
    	params.put("batch_input_shape", tmp);
    	params.put("dtype", Common.DEFAULT_DTYPE);
    	params.put("sparse", Common.DEFAULT_SPARSE);
    	params.put("name", "input_1");
    	layer_config.put("inbound_nodes", new ArrayList<Object>());
    	List<Object> tmp1 = new ArrayList<Object>();
    	List<Object> tmp2 = new ArrayList<Object>();
    	tmp1.add(tmp2);
    	tmp2.add("input_1");
    	tmp2.add(0);
    	tmp2.add(0);
     	this.config.put("input_layers", tmp1);
     	layer_config.put("config", params);
     	this.layers.add(layer_config);
     	//LOG.info("Successfully add input layer!");
    }
    private void addOutputLayer() throws Exception{
    	//int columnNums = 0;
    	List<Object> tmp1 = new ArrayList<Object>();
    	List<Object> tmp2 = new ArrayList<Object>();
    	
     	tmp1.add(tmp2);
    	tmp2.add(this.last_layer.getName());
    	tmp2.add(0);
    	tmp2.add(0);
     	this.config.put("output_layers", tmp1);
    }
    
    @Override
    public double train() throws IOException {
    	File models = new File("models");
        FileUtils.forceMkdir(models);
    	try {
    		JSONUtils.writeValue(new File("models/model0.dnn"), this.network);
    		String kerasScriptPath = Environment.getProperty(Environment.SHIFU_HOME) + "/scripts/train.py";
    		KerasExecutor.getExecutor().submitJob(modelConfig, kerasScriptPath, 
    				SourceType.LOCAL, pathFinder, "./models/model0.dnn", select_status);
    	}catch(Exception e) {
    		throw new IOException(e.getMessage().toString());
    	}
        return 0;
    }

    //public DNNNetwork getNetwork() {
    //    return network;
    //}
}

   
    
