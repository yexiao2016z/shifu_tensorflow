package ml.shifu.shifu.dnn;

public class Common {
	public static enum LayerCatagory{Dense, Dropout, Conv1D, Input, Activation}
	public static enum ActivationCatagory{softmax, relu, sigmoid, tanh, linear}
	public static enum InitializerCatagory{Constant, RandomNormal, RandomUniform}
	public static enum LossCatagory{Mean_squared, Hinge, Crossentropy}
	public static enum OptimizerCatagory{SGD, Adagrad, Adam}
	public static enum PaddingCatagory{Valid, Causal, Same}
	public static final String KERAS_VERSION = "2.1.5-tf";
	public static final String BACKEND = "tensorflow";
	public static final String MODEL_NAME = "model";
	public static boolean DEFAULT_TRAINABLE = true;
	public static String DEFAULT_DTYPE = "float32"; 
	public static boolean DEFAULT_SPARSE = false;
	public static double DEFAULT_MEAN = 0.0;
	public static double DEFAULT_STDDEV = 1.0;
	public static Double DEFAULT_MINVAL = 0.0;
	public static Double DEFAULT_MAXVAL = 1.0;
	public static double DEFAULT_CONSTANT_VALUE = 0.0;
	public static double DEFAULT_RATE = 0.5;
	public static String DEFAULT_MODEL_NAME = "model_1"; 
}
