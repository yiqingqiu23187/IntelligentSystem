import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Classification {

    public static void main(String[] args) {
        int trainSize = 450;
        int testSize = 620 - trainSize;
        double[][][] input = new double[trainSize][12][28 * 28];//这样的顺序是为了提高局部性
        double[][] output = new double[12][12];
        for (int i = 0; i < trainSize; i++) {
            for (int j = 0; j < 12; j++) {
                input[i][j] = imgInfo("train/" + (j + 1) + "/" + (i + 1) + ".bmp");
            }
        }

        double[][][] test_input = new double[testSize][12][28 * 28];//这样的顺序是为了提高局部性
        for (int i = 0; i < testSize; i++) {
            for (int j = 0; j < 12; j++) {
                test_input[i][j] = imgInfo("train/" + (j + 1) + "/" + (i + 1 + trainSize) + ".bmp");
            }
        }
        for (int i = 0; i < 12; i++) {
            output[i][i] = 1;
        }

        Network network = new Network(new int[]{28 * 28, 64, 12});
        int totalEp = 0;
        double lastRate = -1;
        double rate = 0;
        int ep = 5;
        while (true) {
            lastRate = rate;
            totalEp += ep;
            for (int i = 0; i < ep; i++) {
                for (int j = 0; j < trainSize; j++) {
                    for (int k = 0; k < 12; k++) {
                        network.trainClassfi(input[j][k], output[k]);
                    }
                }
            }
            int right = 0;
            for (int j = 0; j < testSize; j++) {
                for (int k = 0; k < 12; k++) {
                    if (network.testClassfi(test_input[j][k], output[k])) right++;
                }
            }
            rate = right / (testSize * 12.0);
            if (rate < lastRate) {
                totalEp -= ep;
                rate = lastRate;
                break;
            }
        }
        System.out.println("跑了epoch: "+totalEp+"准确率: "+rate);

        
        System.out.println();
        for (int i = 0; i < 12; i++) {
            System.out.println(network.predictClassifi(test_input[50][i]));
        }
        System.out.println(network.predictClassifi(test_input[100][8]));
        System.out.println(network.predictClassifi(test_input[50][11]));
        System.out.println(network.predictClassifi(test_input[50][6]));
        System.out.println(network.predictClassifi(test_input[50][2]));
        System.out.println(network.predictClassifi(test_input[50][5]));
        System.out.println(network.predictClassifi(test_input[50][9]));



    }


    public static double[] imgInfo(String src) {
        // 读取图片到BufferedImage
        BufferedImage bf = readImage(src);//绝对路径+文件名
        // 将图片转换为二维数组
        int[] rgbArray1 = convertImageToArray(bf);
        double[] imgInfo = new double[rgbArray1.length];
        for (int i = 0; i < rgbArray1.length; i++) {
            if (rgbArray1[i] == -1) {
                imgInfo[i] = 0;
            } else {
                imgInfo[i] = 1;
            }
        }

        return imgInfo;
    }

    public static BufferedImage readImage(String imageFile) {
        File file = new File(imageFile);
        BufferedImage bf = null;
        try {
            bf = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bf;
    }

    public static int[] convertImageToArray(BufferedImage bf) {
        // 获取图片宽度和高度
        int width = bf.getWidth();
        int height = bf.getHeight();
        // 将图片sRGB数据写入一维数组
        int[] data = new int[width * height];
        bf.getRGB(0, 0, width, height, data, 0, width);
        return data;
    }
}
