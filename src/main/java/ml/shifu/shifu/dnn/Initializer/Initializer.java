package ml.shifu.shifu.dnn.Initializer;

import ml.shifu.shifu.dnn.Common.InitializerCatagory;

public abstract class Initializer {
	private InitializerCatagory ini;
	public Initializer(InitializerCatagory ini){
		this.ini = ini;
	}
	public InitializerCatagory getInitializerCatagory(){
		return this.ini;
	}
}
