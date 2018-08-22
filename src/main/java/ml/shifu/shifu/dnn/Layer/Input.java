package ml.shifu.shifu.dnn.Layer;

import ml.shifu.shifu.dnn.Common.LayerCatagory;

public class Input extends Layer{
	private int shape;
	private boolean sparse;
	private int myIndex;
	private static int index = 1;
	public Input(int shape) {
		this(shape,false);
		// TODO Auto-generated constructor stub
	}
	public Input(int shape, boolean sparse) {
		super(LayerCatagory.Input);
		this.shape = shape;
		this.sparse = sparse;
		this.myIndex = index++;
	}
	public int getShape() {
		return this.shape;
	}
	public boolean isSparse() {
		return sparse;
	}
	public String getName() {
		return this.getLayerCatagory().name().toLowerCase() + "_" + this.myIndex;
	}
}
