public class Sin {
    public static void main(String[] args) {
        int sampleNum = 40;
        double[][] input = new double[sampleNum][1];
        double[][] output = new double[sampleNum][1];
        for (int i = 0; i < sampleNum; i++) {
            input[i][0] = Math.PI * 2 * i / sampleNum - Math.PI;
            output[i][0] = Math.sin(input[i][0]);
        }

        int testSize = 4000;
        double[][] test_inputs = new double[testSize][1];
        double[][] test_outputs = new double[testSize][1];
        for (int i = 0; i < testSize; i++) {
            // test_inputs[i][0] = Math.PI * 2 * Math.random() - Math.PI;
            test_inputs[i][0] = Math.PI * 2 * i / testSize - Math.PI;
            test_outputs[i][0] = Math.sin(test_inputs[i][0]);
        }
        Network bp = new Network(new int[]{1, 50, 1});
        for (int j = 0; j < 100000; j++) {
            for (int i = 0; i < sampleNum; i++) {
                bp.trainSin(input[i], output[i]);
            }
        }

        double totalError = 0;

        for (int i = 0; i < testSize; i++) {
            totalError += Math.pow(bp.testSin(test_inputs[i], test_outputs[i]), 2);
        }
        System.out.println("totalerror for test set:" + totalError / testSize);


    }
}
