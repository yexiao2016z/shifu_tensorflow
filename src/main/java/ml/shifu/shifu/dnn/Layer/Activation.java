package ml.shifu.shifu.dnn.Layer;

import ml.shifu.shifu.dnn.Common.LayerCatagory;
import ml.shifu.shifu.dnn.Common.ActivationCatagory;

public class Activation extends Layer {
	private ActivationCatagory ac;
	private static int index = 1;
	private int myIndex;
	public Activation(ActivationCatagory ac) {
		super(LayerCatagory.Activation);
		this.ac = ac;
		this.myIndex = index;
		index++;
	}
	public ActivationCatagory getActivationCatagory() {
		return this.ac;
	}
	public String getName() {
		return this.getActivationCatagory().name().toLowerCase() + "_" + this.myIndex;
	}
}
