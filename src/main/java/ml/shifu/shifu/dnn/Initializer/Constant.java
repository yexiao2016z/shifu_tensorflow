package ml.shifu.shifu.dnn.Initializer;

import ml.shifu.shifu.dnn.Common.InitializerCatagory;

public class Constant extends Initializer {
	private double consts;
	public Constant() {
		this(0);
		// TODO Auto-generated constructor stub
	}
	public Constant(double consts) {
		super(InitializerCatagory.Constant);
		this.consts = consts;
	}
	public double getConstant() {
		return this.consts;
	}
}
