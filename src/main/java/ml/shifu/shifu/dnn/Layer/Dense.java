package ml.shifu.shifu.dnn.Layer;

import ml.shifu.shifu.dnn.Initializer.*;
import ml.shifu.shifu.dnn.Common.ActivationCatagory;
import ml.shifu.shifu.dnn.Common.LayerCatagory;

public class Dense extends Layer{
	private Initializer kernelInitializer;
	private Initializer biasInitializer;
	private int units;
	private int myIndex;
	private static int index = 1;
	private ActivationCatagory ac;
	boolean useBias = true;
	public Dense(int units, ActivationCatagory ac, Initializer kernel){
		this(units, ac, kernel,new Constant());
		this.useBias = false;
	}
	public Dense(int units, ActivationCatagory ac, Initializer kernel, Initializer bias) {
		super(LayerCatagory.Dense);
		this.kernelInitializer = kernel;
		this.biasInitializer = bias;
		this.myIndex = index++;
		this.units = units;
		this.ac = ac;
		this.useBias = true;
	}
	public Dense(int units, Initializer kernel) {
		this(units, ActivationCatagory.linear ,kernel);
		this.useBias = false;
	}
	public Dense(int units, ActivationCatagory ac) {
		this(units, ac, new Constant());
	}
	public Initializer getKernelInitializer() {
		return this.kernelInitializer;
	}
	public Initializer getBiasInitializer() {
		return this.biasInitializer;
	}
	public int getUnits() {
		return this.units;
	}
	public ActivationCatagory getActivationCatagory() {
		return this.ac;
	}
	public String getName() {
		return this.getLayerCatagory().name().toLowerCase() + "_" + this.myIndex;
	}
	
	public boolean isUseBias() {
		return this.useBias;
	}
}
