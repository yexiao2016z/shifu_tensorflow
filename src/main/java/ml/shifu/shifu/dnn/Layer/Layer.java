package ml.shifu.shifu.dnn.Layer;
import ml.shifu.shifu.dnn.Common.LayerCatagory;

public abstract class  Layer {
	protected LayerCatagory layerCatagory;
	public Layer(LayerCatagory layer){
		//this.layer_index = Layer.index++;
		this.layerCatagory = layer;
	}
	public LayerCatagory getLayerCatagory() {
		return this.layerCatagory;
	}
	public abstract String getName();
}
