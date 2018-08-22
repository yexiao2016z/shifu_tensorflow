package ml.shifu.shifu.dnn.Layer;

import ml.shifu.shifu.dnn.Common.LayerCatagory;

public class Dropout extends Layer {
	private double rate;
	private int myIndex;
	private static int index = 1;
	public Dropout() {
		this(0);
	}
	public Dropout(double rate){
		super(LayerCatagory.Dropout);
		if(rate > 1 || rate < 0) {
			this.rate = 0;
		}else {
			this.rate = rate;
		}
		this.myIndex = index++;
	}
	public double getRate() {
		return this.rate;
	}
	public String getName() {
		return this.getLayerCatagory().name().toLowerCase() + "_" + this.myIndex;
	}
}
