package ml.shifu.shifu.dnn.Initializer;

import ml.shifu.shifu.dnn.Common.InitializerCatagory;

public class RandomNormal extends Initializer {
	private double mean;
	private double stddev;
	public RandomNormal() {
		//super(InitializerCatagory.RandomNormal);
		this(0,0.05);
	}
	public RandomNormal(double mean, double stddev) {
		super(InitializerCatagory.RandomNormal);
		this.mean = mean;
		this.stddev = stddev;
	}
	public double getMean() {
		return this.mean;
	}
	public double getStddev() {
		return this.stddev;
	}
}
