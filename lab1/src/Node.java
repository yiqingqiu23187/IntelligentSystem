import java.io.Serializable;

public class Node implements Serializable {
    int weightSize;
    double[] inputWeights;
    double b;
    double input;
    double output;
    double delta;//并不是真正的delta w ,只是被求和项

    public Node(int weightSize) {
        this.weightSize = weightSize;
        inputWeights = new double[weightSize];

        for (int i = 0; i < weightSize; i++) {
            inputWeights[i] = Math.random() / 1000;
        }
        b = -Math.random() / 1000;
    }

    double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    double derivatives() {
        return output * (1 - output);
    }

    void forward() {
        output = sigmoid(input);
    }

}
