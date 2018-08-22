package ml.shifu.shifu.dnn.Layer;

import ml.shifu.shifu.dnn.Common.ActivationCatagory;
import ml.shifu.shifu.dnn.Common.LayerCatagory;
import ml.shifu.shifu.dnn.Common.PaddingCatagory;
import ml.shifu.shifu.dnn.Initializer.Initializer;

public class Conv1D extends Layer {
	private int myIndex;
	private static int index = 1;
	private Initializer kernel;
	private Initializer bias;
	private ActivationCatagory ac;
	private PaddingCatagory pad;
	private int kernel_size;
	private int filters;
	private int stride;
	public Conv1D(int filters, int kernel_size, int stride, PaddingCatagory pad, 
			ActivationCatagory ac, Initializer kernel, Initializer bias) {
		super(LayerCatagory.Conv1D);
		this.filters = filters;
		this.bias = bias;
		this.kernel = kernel;
		this.ac = ac;
		this.kernel_size = kernel_size;
		this.myIndex = index++;
		// TODO Auto-generated constructor stub
	}
	public String getName() {
		return this.getLayerCatagory().name() + "_" + this.myIndex;
	}
	
}
