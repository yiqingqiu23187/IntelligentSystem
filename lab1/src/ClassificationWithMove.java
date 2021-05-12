import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.logging.Logger;

public class ClassificationWithMove {

    public static void main(String[] args) {
        //trainnet();
        testnet();
    }

    public static void trainnet() {
        int trainSize = 620;
        //int testSize = 620 - trainSize;
        double[][][][][] input = new double[trainSize][12][5][3][28 * 28];//这样的顺序是为了提高局部性
        double[][] output = new double[12][12];
        for (int i = 0; i < trainSize; i++) {
            for (int j = 0; j < 12; j++) {
                for (int k = 0; k < 5; k++)
                    for (int m = 0; m < 2; m++)
                        input[i][j][k][m] = imgInfo("train/" + (j + 1) + "/" + (i + 1) + ".bmp", k, m + 1);
            }
        }

//        double[][][] test_input = new double[testSize][12][28 * 28];//这样的顺序是为了提高局部性
//        for (int i = 0; i < testSize; i++) {
//            for (int j = 0; j < 12; j++) {
//                test_input[i][j] = imgInfo("train/" + (j + 1) + "/" + (i + 1 + trainSize) + ".bmp", 0, 0);
//            }
//        }
        for (int i = 0; i < 12; i++) {
            output[i][i] = 1;
        }

        Network network = new Network(new int[]{28 * 28, 80, 12});
        int totalEp = 0;
        double lastRate = -1;
        double rate = 0;
        int ep = 10;
        while (rate > lastRate || totalEp <= 20) {
            lastRate = rate;
            totalEp += ep;
            for (int i = 0; i < ep; i++) {
                for (int j = 0; j < trainSize; j++) {
                    for (int k = 0; k < 12; k++) {
                        for (int m = 0; m < 5; m++)
                            for (int n = 0; n < 2; n++)
                                network.trainClassfi(input[j][k][m][n], output[k]);
                    }
                }
            }
//            int right = 0;
//            for (int j = 0; j < testSize; j++) {
//                for (int k = 0; k < 12; k++) {
//                    if (network.testClassfi(test_input[j][k], output[k])) right++;
//                }
//            }
//            rate = right / (testSize * 12.0);
//            System.out.println("正确率：" + lastRate);
//            System.out.println("跑了ep: " + (totalEp - ep));
        }

          writeObject(network, "D:\\PJ\\IntelliLabs\\lab1\\fullnetwork.save");
    }


    public static void testnet() {
        Network network = (Network) readObject("D:\\PJ\\IntelliLabs\\lab1\\fullnetwork.save");

        int testSize = 1800;
        double[][] input = new double[testSize][12];
        for (int i = 0; i < testSize; i++) {
            for (int j = 0; j < 12; j++) {
                input[i] = imgInfo("test/" + (i + 1) + ".bmp", 0, 0);
            }
        }

        try {
            File file = new File("D:\\PJ\\IntelliLabs\\Lab1Part2\\pred.txt");
            if (file.exists()){
                file.delete();
                file.createNewFile();
            }
            BufferedWriter out = new BufferedWriter(new FileWriter(file));

            for (int i = 0; i < testSize; i++) {
                int preout = network.predictClassifi(input[i]);
                out.write(String.valueOf(preout)+"\r\n");
            }
            out.close();
        }catch (IOException e){
            e.printStackTrace();
        }


    }


    public static Object readObject(String path) {
        FileInputStream fis = null;
        ObjectInputStream ois = null;
        try {
            fis = new FileInputStream(path);
            ois = new ObjectInputStream(fis);
            return ois.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void writeObject(Object object, String path) {
        FileOutputStream fos = null;
        BufferedOutputStream bos = null;
        ObjectOutputStream oos = null;
        try {
            fos = new FileOutputStream(path);

            bos = new BufferedOutputStream(fos);
            oos = new ObjectOutputStream(bos);
            oos.writeObject(object);
            //清空缓冲区
            oos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double[] imgInfo(String src, int position, int offset) {
        BufferedImage bf = readImage(src);
        int[] rgbArray = convertImageToArray(bf, position, offset);
        return getImgInfo(rgbArray);
    }

    private static BufferedImage readImage(String imageFile) {
        File file = new File(imageFile);
        BufferedImage bf = null;
        try {
            bf = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bf;
    }

    private static int[] convertImageToArray(BufferedImage bf, int position, int offset) {
        int width = bf.getWidth();
        int height = bf.getHeight();
        int[] data = new int[width * height];
        switch (position) {
            case 0:
                //中心
                bf.getRGB(0, 0, width, height, data, 0, width);
                break;
            case 1:
                //左边便宜
                bf.getRGB(offset, 0, width - offset, height, data, offset, width);
                break;
            case 2:
                //向右偏移
                bf.getRGB(0, 0, width - offset, height, data, 0, width);
                break;
            case 3:
                //向上偏移
                bf.getRGB(0, offset, width, height - offset, data, 0, width);
                break;
            case 4:
                //向下偏移
                bf.getRGB(0, 0, width, height - offset, data, 0, width);
                break;
        }
        return data;
    }

    private static double[] getImgInfo(int[] rgbArray) {
        double[] imgInfo = new double[rgbArray.length];
        for (int i = 0; i < rgbArray.length; i++) {
            if (rgbArray[i] == -1) {
                imgInfo[i] = 0;
            } else {
                imgInfo[i] = 1;
            }
        }
        return imgInfo;
    }
}
