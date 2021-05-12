import java.io.Serializable;

public class Layer  implements Serializable {
    Node[] nodes;
    int nodeNumber;
    Layer left;
    Layer right;

    //从左到右建立层
    public Layer(int nodeNumber, Layer left)  {
        this.nodeNumber = nodeNumber;
        this.left = left;
        nodes = new Node[nodeNumber];
        for (int i = 0; i < nodeNumber; i++) {
            if (left != null) nodes[i] = new Node(left.nodeNumber);
            else nodes[i] = new Node(0);
        }
    }


    void forward() {
        for (Node thisNode : nodes) {
            double temp = 0;
            for (int i = 0; i < left.nodes.length; i++) {
                temp += left.nodes[i].output * thisNode.inputWeights[i];
            }
            temp += thisNode.b;

            thisNode.input = temp;
            thisNode.forward();

        }
    }


    void backward() {
        for (int i = 0; i < nodeNumber; i++) {
            double temp = 0;
            for (int j = 0; j < right.nodeNumber; j++) {
                Node node = right.nodes[j];
                temp += node.delta * node.derivatives() * node.inputWeights[i];
            }
            nodes[i].delta = temp;
        }
        for (int i = 0; i < nodeNumber; i++) {
            Node node = nodes[i];
            for (int j = 0; j < node.weightSize; j++) {
                node.inputWeights[j] += left.nodes[j].output * node.derivatives() * node.delta * Network.wLearningRate;// + -0.0005 + Math.random()/1000;
            }
            node.b += node.derivatives() * node.delta * Network.bLearningRate;//-0.0005 + Math.random()/1000;
        }

    }
}
