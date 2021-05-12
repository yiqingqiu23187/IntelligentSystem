import java.io.Serializable;

public class Network implements Serializable {
    public static double wLearningRate = 0.005;
    public static double bLearningRate = 0.002;
    Layer[] layers;

    public Network(int[] nums) {
        layers = new Layer[nums.length];
        layers[0] = new Layer(nums[0], null);
        for (int i = 1; i < nums.length; i++) {
            layers[i] = new Layer(nums[i], layers[i - 1]);
        }
        for (int i = 0; i < nums.length - 1; i++) {
            layers[i].right = layers[i + 1];
        }
    }

    void trainSin(double[] inputs, double[] outputs) {
        forward(inputs);

        //最后一层的forward不一样
        Layer lastLayer = layers[layers.length - 1];
        for (Node thisNode : lastLayer.nodes) {
            thisNode.output = thisNode.input;
        }
        backward(outputs);
    }

    double testSin(double[] inputs, double[] outputs) {
        forward(inputs);

        //最后一层的forward不一样
        Layer lastLayer = layers[layers.length - 1];
        for (Node thisNode : lastLayer.nodes) {
            thisNode.output = thisNode.input;
        }

        return outputs[0] - layers[layers.length - 1].nodes[0].output;
    }

    void trainClassfi(double[] inputs, double[] outputs) {
        forward(inputs);

        softmax();

        backward(outputs);
    }


    boolean testClassfi(double[] inputs, double[] outputs) {
        forward(inputs);
        softmax();

        int index = -1;
        double max = 0;
        Layer lastLayer = layers[layers.length - 1];
        for (int i = 0; i < outputs.length; i++) {
            if (lastLayer.nodes[i].output > max) {
                index = i;
                max = lastLayer.nodes[i].output;
            }
        }
        if (outputs[index] == 1) return true;
        else return false;
    }

    double lossForClassfi(double[] inputs, double[] outputs) {
        forward(inputs);
        softmax();

        int index = -1;
        double max = 0;
        Layer lastLayer = layers[layers.length - 1];
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += outputs[i] * Math.log(lastLayer.nodes[i].output) / Math.log(Math.getExponent(1));
        }
        return -loss;
    }

    int predictClassifi(double[] inputs) {
        forward(inputs);
        softmax();

        int index = -1;
        double max = 0;
        Layer lastLayer = layers[layers.length - 1];
        for (int i = 0; i < lastLayer.nodes.length; i++) {
            if (lastLayer.nodes[i].output > max) {
                index = i;
                max = lastLayer.nodes[i].output;
            }
        }
        return (index + 1);
    }

    void forward(double[] inputs) {
        for (int n = 0; n < inputs.length; n++) {
            layers[0].nodes[n].output = inputs[n];
        }

        for (int i = 1; i < layers.length; i++) {
            layers[i].forward();
        }
    }

    void backward(double[] outputs) {
        Layer lastLayer = layers[layers.length - 1];
        for (int i = 0; i < lastLayer.nodes.length; i++) {
            Node node = lastLayer.nodes[i];
            node.delta = outputs[i] - node.output;
            for (int j = 0; j < node.weightSize; j++) {
                node.inputWeights[j] += lastLayer.left.nodes[j].output * node.delta * Network.wLearningRate;
            }
            node.b += node.delta * Network.bLearningRate;
        }

        for (int i = layers.length - 2; i > 0; i--) {
            layers[i].backward();
        }
    }


    void softmax() {
        //最后一层的forward不一样
        Layer lastLayer = layers[layers.length - 1];
        double total = 0;
        for (Node thisNode : lastLayer.nodes) {
            total += Math.exp(thisNode.input);
        }
        for (Node thisNode : lastLayer.nodes) {
            thisNode.output = Math.exp(thisNode.input) / total;
        }
    }

}
