package ml.shifu.shifu.dnn.Initializer;

import ml.shifu.shifu.dnn.Common.InitializerCatagory;

public class RandomUniform extends Initializer{
	private double minVal;
	private double maxVal;
	public RandomUniform(double minVal, double maxVal) {
		super(InitializerCatagory.RandomUniform);
		this.minVal = minVal;
		this.maxVal = maxVal;
	}
	public RandomUniform(double minVal) {
		this(minVal,0.05);
	}
	public RandomUniform() {
		this(-0.05);
	}
	public double getMinVal() {
		return this.minVal;
	}
	public double getMaxVal() {
		return this.maxVal;
	}
}
